import numpy as np
from scipy.optimize import fsolve
from orbit_core import (
    MU_EARTH, R_EARTH, KeplerElements,
    kepler_to_rv, rv_to_kepler, vis_viva,
    get_orbit_points
)


def hohmann_transfer(r1, r2, i=0, raan=0, argp=0):
    a_transfer = (r1 + r2) / 2
    e_transfer = (r2 - r1) / (r2 + r1)
    
    v1 = np.sqrt(MU_EARTH / r1)
    v2 = np.sqrt(MU_EARTH / r2)
    
    vp_transfer = vis_viva(a_transfer, r1)
    va_transfer = vis_viva(a_transfer, r2)
    
    dv1 = abs(vp_transfer - v1)
    dv2 = abs(v2 - va_transfer)
    total_dv = dv1 + dv2
    
    transfer_time = np.pi * np.sqrt(a_transfer**3 / MU_EARTH)
    
    transfer_orbit = KeplerElements(
        a_transfer, e_transfer, i, raan, argp, 0,
        units='km_deg'
    )
    
    initial_orbit = KeplerElements(
        r1, 0, i, raan, argp, 0,
        units='km_deg'
    )
    
    final_orbit = KeplerElements(
        r2, 0, i, raan, argp, 180,
        units='km_deg'
    )
    
    return {
        'initial_orbit': initial_orbit,
        'transfer_orbit': transfer_orbit,
        'final_orbit': final_orbit,
        'dv1': dv1,
        'dv2': dv2,
        'total_dv': total_dv,
        'transfer_time': transfer_time,
        'burn1_position': '近地点' if r1 < r2 else '远地点',
        'burn2_position': '远地点' if r1 < r2 else '近地点'
    }


def bielliptic_transfer(r1, r2, rb, i=0, raan=0, argp=0):
    a1 = (r1 + rb) / 2
    a2 = (r2 + rb) / 2
    
    v1 = np.sqrt(MU_EARTH / r1)
    v2 = np.sqrt(MU_EARTH / r2)
    
    vp1 = vis_viva(a1, r1)
    va1 = vis_viva(a1, rb)
    vp2 = vis_viva(a2, rb)
    va2 = vis_viva(a2, r2)
    
    dv1 = abs(vp1 - v1)
    dv2 = abs(vp2 - va1)
    dv3 = abs(v2 - va2)
    total_dv = dv1 + dv2 + dv3
    
    t1 = np.pi * np.sqrt(a1**3 / MU_EARTH)
    t2 = np.pi * np.sqrt(a2**3 / MU_EARTH)
    transfer_time = t1 + t2
    
    transfer_orbit1 = KeplerElements(
        a1, (rb - r1) / (rb + r1), i, raan, argp, 0,
        units='km_deg'
    )
    
    transfer_orbit2 = KeplerElements(
        a2, (rb - r2) / (rb + r2), i, raan, argp, 180,
        units='km_deg'
    )
    
    hohmann = hohmann_transfer(r1, r2, i, raan, argp)
    
    return {
        'transfer_orbit1': transfer_orbit1,
        'transfer_orbit2': transfer_orbit2,
        'dv1': dv1,
        'dv2': dv2,
        'dv3': dv3,
        'total_dv': total_dv,
        'transfer_time': transfer_time,
        'hohmann_dv': hohmann['total_dv'],
        'dv_saving': hohmann['total_dv'] - total_dv,
        'is_better': total_dv < hohmann['total_dv']
    }


def inclination_change(r, delta_i, i_initial=0):
    v = np.sqrt(MU_EARTH / r)
    delta_i_rad = np.deg2rad(delta_i)
    dv = 2 * v * np.sin(delta_i_rad / 2)
    
    return {
        'delta_v': dv,
        'velocity': v,
        'delta_i_deg': delta_i,
        'burn_position': '升交点或降交点',
        'efficiency_note': '倾角改变应在速度最小的远地点进行以节省燃料'
    }


def combined_maneuver(r, dv_tangential, delta_i):
    v = np.sqrt(MU_EARTH / r)
    delta_i_rad = np.deg2rad(delta_i)
    
    v_final = v + dv_tangential
    dv = np.sqrt(v**2 + v_final**2 - 2 * v * v_final * np.cos(delta_i_rad))
    
    return {
        'total_dv': dv,
        'separate_dv': abs(dv_tangential) + 2 * v * np.sin(delta_i_rad / 2),
        'saving': abs(dv_tangential) + 2 * v * np.sin(delta_i_rad / 2) - dv,
        'note': '同时进行切向和法向机动可以节省燃料'
    }


def phasing_maneuver(r_original, delta_phase_deg, direction='forward'):
    delta_phase = np.deg2rad(delta_phase_deg)
    v_original = np.sqrt(MU_EARTH / r_original)
    T_original = 2 * np.pi * np.sqrt(r_original**3 / MU_EARTH)
    n_original = 2 * np.pi / T_original
    
    if direction == 'forward':
        delta_n = -delta_phase / T_original
    else:
        delta_n = delta_phase / T_original
    
    n_phasing = n_original + delta_n
    T_phasing = 2 * np.pi / n_phasing
    a_phasing = (MU_EARTH * (T_phasing / (2 * np.pi))**2)**(1/3)
    
    if delta_phase_deg < 180:
        n_loops = 1
    else:
        n_loops = int(np.ceil(delta_phase_deg / 180))
    
    r_p = 2 * a_phasing - r_original
    
    if r_p <= R_EARTH:
        n_loops += 1
        T_phasing = T_original * n_loops / (n_loops - delta_phase_deg / 360)
        a_phasing = (MU_EARTH * (T_phasing / (2 * np.pi))**2)**(1/3)
        r_p = 2 * a_phasing - r_original
    
    dv1 = abs(vis_viva(a_phasing, r_original) - v_original)
    dv2 = dv1
    total_dv = dv1 + dv2
    
    transfer_time = n_loops * T_phasing
    
    phasing_orbit = KeplerElements(
        a_phasing, abs(r_original - r_p) / (r_original + r_p),
        0, 0, 0, 0, units='km_deg'
    )
    
    return {
        'original_orbit_r': r_original,
        'phasing_orbit': phasing_orbit,
        'delta_phase_deg': delta_phase_deg,
        'dv1': dv1,
        'dv2': dv2,
        'total_dv': total_dv,
        'transfer_time': transfer_time,
        'n_loops': n_loops,
        'r_perigee': r_p,
        'direction': direction
    }


def solve_lambert(r1, r2, tof, prograde=True, max_iter=200, tol=1e-8):
    from scipy.optimize import brentq
    
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cross_prod = np.cross(r1, r2)
    sin_dnu = np.linalg.norm(cross_prod) / (r1_mag * r2_mag)
    
    if not prograde:
        sin_dnu = -sin_dnu
    
    dnu = np.arctan2(sin_dnu, cos_dnu)
    if dnu < 0:
        dnu += 2 * np.pi
    
    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / max(1 - cos_dnu, 1e-15))
    
    def compute_tof(psi):
        if abs(psi) < 1e-6:
            c2 = 0.5
            c3 = 1.0 / 6.0
        elif psi > 0:
            sqrt_psi = np.sqrt(psi)
            c2 = (1 - np.cos(sqrt_psi)) / psi
            c3 = (sqrt_psi - np.sin(sqrt_psi)) / (psi**1.5)
        else:
            sqrt_npsi = np.sqrt(-psi)
            c2 = (1 - np.cosh(sqrt_npsi)) / psi
            c3 = (np.sinh(sqrt_npsi) - sqrt_npsi) / ((-psi)**1.5)
        
        y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(max(c2, 1e-15))
        
        if y <= 0:
            return 1e15
        
        chi = np.sqrt(y / max(c2, 1e-15))
        return (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(MU_EARTH)
    
    def tof_residual(psi):
        return compute_tof(psi) - tof
    
    a_min = (r1_mag + r2_mag) / 4.0
    t_min = 2 * np.pi * np.sqrt(a_min**3 / MU_EARTH)
    
    psi_low = -4 * np.pi**2
    psi_high = 4 * np.pi**2
    
    try:
        f_low = tof_residual(psi_low)
        f_high = tof_residual(psi_high)
        
        if f_low * f_high > 0:
            for psi_guess in np.linspace(-10, 10, 200):
                try:
                    f_guess = tof_residual(psi_guess)
                    if abs(f_guess) < 1000:
                        if f_guess * f_low < 0:
                            psi_high = psi_guess
                            break
                        elif f_guess * f_high < 0:
                            psi_low = psi_guess
                            break
                except:
                    continue
        
        psi_sol = brentq(tof_residual, psi_low, psi_high, 
                         maxiter=max_iter, xtol=tol)
    except:
        best_psi = 0.0
        best_error = np.inf
        for psi_guess in np.linspace(-20, 20, 400):
            try:
                error = abs(tof_residual(psi_guess))
                if error < best_error:
                    best_error = error
                    best_psi = psi_guess
            except:
                continue
        psi_sol = best_psi
    
    psi = psi_sol
    if abs(psi) < 1e-6:
        c2 = 0.5
        c3 = 1.0 / 6.0
    elif psi > 0:
        sqrt_psi = np.sqrt(psi)
        c2 = (1 - np.cos(sqrt_psi)) / psi
        c3 = (sqrt_psi - np.sin(sqrt_psi)) / (psi**1.5)
    else:
        sqrt_npsi = np.sqrt(-psi)
        c2 = (1 - np.cosh(sqrt_npsi)) / psi
        c3 = (np.sinh(sqrt_npsi) - sqrt_npsi) / ((-psi)**1.5)
    
    y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(max(c2, 1e-15))
    
    if y <= 0:
        y = r1_mag + r2_mag
    
    f = 1.0 - y / r1_mag
    g = A * np.sqrt(max(y, 1e-10) / MU_EARTH)
    g_dot = 1.0 - y / r2_mag
    
    if abs(g) < 1e-10:
        v1 = np.sqrt(MU_EARTH / r1_mag) * np.array([0, 1, 0])
        v2 = np.sqrt(MU_EARTH / r2_mag) * np.array([-np.sin(dnu), np.cos(dnu), 0])
    else:
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g
    
    return v1, v2


def lambert_transfer(elements1, elements2, tof, prograde=True):
    r1, _ = kepler_to_rv(elements1)
    r2, _ = kepler_to_rv(elements2)
    
    v1, v2 = solve_lambert(r1, r2, tof, prograde)
    
    _, v1_initial = kepler_to_rv(elements1)
    _, v2_final = kepler_to_rv(elements2)
    
    dv1 = np.linalg.norm(v1 - v1_initial)
    dv2 = np.linalg.norm(v2_final - v2)
    total_dv = dv1 + dv2
    
    transfer_elements = rv_to_kepler(r1, v1)
    
    return {
        'transfer_orbit': transfer_elements,
        'v1_departure': v1,
        'v2_arrival': v2,
        'dv1': dv1,
        'dv2': dv2,
        'total_dv': total_dv,
        'tof': tof,
        'transfer_points': get_orbit_points(transfer_elements)
    }


def cw_matrix(n, dt):
    n_dt = n * dt
    sin_nt = np.sin(n_dt)
    cos_nt = np.cos(n_dt)
    
    phi = np.zeros((6, 6))
    
    phi[0, 0] = 4 - 3 * cos_nt
    phi[0, 1] = 0
    phi[0, 2] = 0
    phi[0, 3] = sin_nt / n
    phi[0, 4] = 2 * (1 - cos_nt) / n
    phi[0, 5] = 0
    
    phi[1, 0] = 6 * (sin_nt - n_dt)
    phi[1, 1] = 1
    phi[1, 2] = 0
    phi[1, 3] = 2 * (cos_nt - 1) / n
    phi[1, 4] = (4 * sin_nt - 3 * n_dt) / n
    phi[1, 5] = 0
    
    phi[2, 0] = 0
    phi[2, 1] = 0
    phi[2, 2] = cos_nt
    phi[2, 3] = 0
    phi[2, 4] = 0
    phi[2, 5] = sin_nt / n
    
    phi[3, 0] = 3 * n * sin_nt
    phi[3, 1] = 0
    phi[3, 2] = 0
    phi[3, 3] = cos_nt
    phi[3, 4] = 2 * sin_nt
    phi[3, 5] = 0
    
    phi[4, 0] = 6 * n * (cos_nt - 1)
    phi[4, 1] = 0
    phi[4, 2] = 0
    phi[4, 3] = -2 * sin_nt
    phi[4, 4] = 4 * cos_nt - 3
    phi[4, 5] = 0
    
    phi[5, 0] = 0
    phi[5, 1] = 0
    phi[5, 2] = -n * sin_nt
    phi[5, 3] = 0
    phi[5, 4] = 0
    phi[5, 5] = cos_nt
    
    return phi


def cw_propagate(state0, n, dt):
    phi = cw_matrix(n, dt)
    return phi @ state0


def cw_rendezvous(state0, t_total, n_burns=2):
    x0, y0, z0, vx0, vy0, vz0 = state0
    n = np.sqrt(MU_EARTH / (R_EARTH + 400)**3)
    
    if n_burns == 2:
        dt = t_total / 2
        phi1 = cw_matrix(n, dt)
        phi2 = cw_matrix(n, dt)
        phi_total = phi2 @ phi1
        
        target = np.zeros(6)
        
        B1 = np.zeros((6, 3))
        B1[3:6, :] = np.eye(3)
        B1 = phi2 @ B1
        
        B2 = np.zeros((6, 3))
        B2[3:6, :] = np.eye(3)
        
        A = np.hstack([B1[:, :3], B2[:, :3]])
        b = target - phi_total @ state0
        
        dv = np.linalg.lstsq(A, b, rcond=None)[0]
        dv1 = dv[:3]
        dv2 = dv[3:]
        
        return {
            'dv1': dv1,
            'dv2': dv2,
            'total_dv': np.linalg.norm(dv1) + np.linalg.norm(dv2),
            't1': 0,
            't2': dt,
            'initial_state': state0,
            'n': n
        }
    else:
        return None


def multi_turn_lambert(r1, r2, n_revs, tof_min=None, tof_max=None, n_points=50):
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    a_min = (r1_mag + r2_mag) / 4
    t_min = 2 * np.pi * np.sqrt(a_min**3 / MU_EARTH)
    
    if tof_min is None:
        tof_min = t_min * (n_revs + 0.1)
    if tof_max is None:
        tof_max = t_min * (n_revs + 2)
    
    tofs = np.linspace(tof_min, tof_max, n_points)
    dvs = []
    
    for tof in tofs:
        try:
            v1, v2 = solve_lambert(r1, r2, tof)
            v_circular = np.sqrt(MU_EARTH / r1_mag)
            dv = np.linalg.norm(v1 - np.array([0, v_circular, 0]))
            dvs.append(dv)
        except:
            dvs.append(np.inf)
    
    return tofs, np.array(dvs)
