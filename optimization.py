import numpy as np
from scipy.optimize import minimize, differential_evolution
from orbit_core import (
    MU_EARTH, R_EARTH, KeplerElements,
    kepler_to_rv, rv_to_kepler, propagate_orbit,
    get_orbit_points
)
from maneuvers import solve_lambert


def objective_velocity_changes(params, initial_elements, final_elements, n_pulses, t_total):
    dvs = params.reshape(n_pulses, 3)
    total_dv = np.sum(np.linalg.norm(dvs, axis=1))
    return total_dv


def constraint_final_orbit(params, initial_elements, final_elements, n_pulses, t_total):
    dvs = params.reshape(n_pulses, 3)
    dt = t_total / n_pulses
    
    current_elements = initial_elements
    
    for i in range(n_pulses):
        r, v = kepler_to_rv(current_elements)
        v_new = v + dvs[i]
        current_elements = rv_to_kepler(r, v_new)
        
        if i < n_pulses - 1:
            current_elements = propagate_orbit(current_elements, 0, dt)
    
    r_final, v_final = kepler_to_rv(current_elements)
    r_target, v_target = kepler_to_rv(final_elements)
    
    error = np.concatenate([
        r_final - r_target,
        v_final - v_target
    ])
    
    return np.linalg.norm(error) - 1e-3


def optimize_maneuver(initial_elements, final_elements, n_pulses, t_total, method='SLSQP'):
    n_vars = n_pulses * 3
    x0 = np.zeros(n_vars)
    
    bounds = [(-5, 5)] * n_vars
    
    constraints = [{
        'type': 'eq',
        'fun': constraint_final_orbit,
        'args': (initial_elements, final_elements, n_pulses, t_total)
    }]
    
    result = minimize(
        objective_velocity_changes,
        x0,
        args=(initial_elements, final_elements, n_pulses, t_total),
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    dvs_optimal = result.x.reshape(n_pulses, 3)
    total_dv = np.sum(np.linalg.norm(dvs_optimal, axis=1))
    
    return {
        'dvs': dvs_optimal,
        'total_dv': total_dv,
        'success': result.success,
        'message': result.message,
        'n_pulses': n_pulses,
        't_total': t_total
    }


def lambert_transfer_optimization(r1, r2, tof_min, tof_max, n_points=100):
    tofs = np.linspace(tof_min, tof_max, n_points)
    dvs = []
    
    for tof in tofs:
        try:
            v1, v2 = solve_lambert(r1, r2, tof)
            v_circular = np.sqrt(MU_EARTH / np.linalg.norm(r1))
            dv = np.linalg.norm(v1 - np.array([0, v_circular, 0]))
            dvs.append(dv)
        except:
            dvs.append(np.inf)
    
    return tofs, np.array(dvs)


def multi_objective_maneuver(initial_elements, final_elements, t_total):
    def objective(params):
        dv_mag, burn_time = params
        return dv_mag + 0.01 * burn_time
    
    result = minimize(
        objective,
        [3.0, t_total],
        bounds=[(0, 10), (100, t_total)],
        method='Nelder-Mead'
    )
    
    return result


def optimize_orbital_parameters(target_altitude, target_inclination, 
                                 constraints=None):
    def objective(params):
        a, e, i = params
        return abs(a - (R_EARTH + target_altitude)) + 0.1 * abs(i - target_inclination)
    
    bounds = [
        (R_EARTH + 160, R_EARTH + 36000),
        (0, 0.9),
        (0, 180)
    ]
    
    result = minimize(
        objective,
        [R_EARTH + target_altitude, 0, target_inclination],
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    a_opt, e_opt, i_opt = result.x
    
    return KeplerElements(a_opt, e_opt, i_opt, 0, 0, 0, units='km_deg')


def station_keeping_delta_v(elements, perturbations, duration_days=30):
    from orbit_propagation import j2_precession_rates
    
    rates = j2_precession_rates(elements)
    
    delta_raan = rates['raan_deg_per_day'] * duration_days
    delta_argp = rates['argp_deg_per_day'] * duration_days
    
    v = np.sqrt(MU_EARTH / elements.a_km)
    delta_i = min(0.1, abs(rates['raan_deg_per_day']) * duration_days * 0.1)
    
    dv_raan = 2 * v * np.sin(np.deg2rad(abs(delta_raan)) / 2)
    dv_argp = v * np.deg2rad(abs(delta_argp))
    dv_inclination = 2 * v * np.sin(np.deg2rad(delta_i) / 2)
    
    total_dv = dv_raan + dv_argp + dv_inclination
    
    return {
        'total_dv_per_year': total_dv * 365 / duration_days,
        'delta_raan_deg': delta_raan,
        'delta_argp_deg': delta_argp,
        'dv_breakdown': {
            'raan_correction': dv_raan,
            'argp_correction': dv_argp,
            'inclination_correction': dv_inclination
        }
    }


def minimize_fuel_usage(initial_elements, target_elements, max_time):
    from maneuvers import hohmann_transfer
    
    r1 = initial_elements.a_km
    r2 = target_elements.a_km
    
    hohmann = hohmann_transfer(r1, r2)
    hohmann_dv = hohmann['total_dv']
    
    def objective(tof):
        try:
            r1_vec, _ = kepler_to_rv(initial_elements)
            r2_vec, _ = kepler_to_rv(target_elements)
            v1, v2 = solve_lambert(r1_vec, r2_vec, tof[0])
            _, v1_initial = kepler_to_rv(initial_elements)
            _, v2_final = kepler_to_rv(target_elements)
            return np.linalg.norm(v1 - v1_initial) + np.linalg.norm(v2_final - v2)
        except:
            return 1e6
    
    result = minimize(
        objective,
        [hohmann['transfer_time'] * 2],
        bounds=[(hohmann['transfer_time'] * 0.5, max_time)],
        method='L-BFGS-B'
    )
    
    optimal_tof = result.x[0]
    optimal_dv = result.fun
    
    return {
        'optimal_tof': optimal_tof,
        'optimal_dv': optimal_dv,
        'hohmann_dv': hohmann_dv,
        'dv_ratio': optimal_dv / hohmann_dv if hohmann_dv > 0 else 1.0
    }
