import numpy as np
from orbit_core import R_EARTH, MU_EARTH, OMEGA_EARTH
from scipy.integrate import solve_ivp


def standard_atmosphere_1976(altitude_km):
    h = altitude_km * 1000.0
    
    R = 287.0
    g0 = 9.80665
    
    if h > 120000:
        h = 120000
    
    if h >= 86000:
        if h <= 91000:
            T = 186.8673
            p = 0.0158557 * np.exp(-(h - 86000) / 5624.0)
        elif h <= 100000:
            T = 210.65
            p = 0.0030087 * np.exp(-(h - 91000) / 6329.0)
        elif h <= 110000:
            T = 210.65 + 0.007 * (h - 100000)
            p_base = 0.00032011
            T_base = 210.65
            L = 0.007
            p = p_base * (T_base / T) ** (g0 / (R * L))
        else:
            T = 280.65 + 0.0045 * (h - 110000)
            p_base = 0.0000735
            T_base = 280.65
            L = 0.0045
            p = p_base * (T_base / T) ** (g0 / (R * L))
    elif h >= 47000:
        if h <= 51000:
            T = 270.65
            p_base = 1.10906
            T_base = 270.65
            p = p_base * np.exp(-(h - 47000) / 7922.0)
        elif h <= 71000:
            T = 270.65 - 0.0028 * (h - 51000)
            p_base = 0.66939
            T_base = 270.65
            L = -0.0028
            p = p_base * (T_base / T) ** (g0 / (R * L))
        else:
            T = 214.65 - 0.002 * (h - 71000)
            p_base = 0.039564
            T_base = 214.65
            L = -0.002
            p = p_base * (T_base / T) ** (g0 / (R * L))
    elif h >= 32000:
        if h <= 47000:
            T = 228.65 + 0.001 * (h - 32000)
            p_base = 8.680187
            T_base = 228.65
            L = 0.001
            p = p_base * (T_base / T) ** (g0 / (R * L))
    elif h >= 20000:
        if h <= 32000:
            T = 216.65
            p_base = 55.2929
            p = p_base * np.exp(-(h - 20000) / 6459.0)
    elif h >= 11000:
        if h <= 20000:
            T = 216.65
            p_base = 226.3226
            p = p_base * np.exp(-(h - 11000) / 6341.6)
    else:
        if h <= 11000:
            T = 288.15 - 0.0065 * h
            p_base = 1013.25
            T_base = 288.15
            L = -0.0065
            p = p_base * (T_base / T) ** (g0 / (R * L))
        else:
            T = 216.65
            p = 226.3226 * np.exp(-(h - 11000) / 6341.6)
    
    rho = p * 100.0 / (R * T)
    
    return rho, T, p


def mach_number(v, T):
    gamma = 1.4
    R = 287.0
    a = np.sqrt(gamma * R * T)
    return v / a


def aerodynamic_coefficients(M, alpha_deg, Cd0=0.15, CL_alpha=0.10, alpha_max_LD=10.0):
    alpha = np.deg2rad(alpha_deg)
    
    if M < 1.0:
        Cd = Cd0 + 2.0 * np.sin(alpha) ** 2
        CL = CL_alpha * alpha_deg
    elif M < 5.0:
        Cd = Cd0 + (2.0 + 0.5 * (M - 1.0)) * np.sin(alpha) ** 2
        CL = CL_alpha * alpha_deg * (1.0 - 0.05 * (M - 1.0))
    else:
        Cd = Cd0 + 4.0 * np.sin(alpha) ** 2
        CL = CL_alpha * alpha_deg * 0.8
    
    CL = np.clip(CL, -1.5, 1.5)
    
    return Cd, CL


def stagnation_heat_flux(v, rho, R_n=0.1):
    v_m_s = v * 1000.0
    rho_kg_m3 = rho
    R_n_m = R_n
    
    q = 1.83e-8 * np.sqrt(rho_kg_m3 / R_n_m) * (v_m_s ** 3)
    
    return q


def ablation_rate(q, T_surface, T_threshold=1500.0, ablation_coeff=2e-7):
    if T_surface < T_threshold:
        return 0.0
    
    return ablation_coeff * max(0, q - 5e4) / 1000.0


def surface_temperature_derivative(q, T, emissivity=0.85, heat_capacity=800.0, mass_per_area=5.0):
    sigma = 5.67e-8
    
    q_rad = emissivity * sigma * (T ** 4)
    dT_dt = (q - q_rad) / (heat_capacity * mass_per_area)
    
    return dT_dt


class ReentryVehicle:
    def __init__(self, mass=1000.0, reference_area=1.0, nose_radius=0.1,
                 ablation_threshold=1500.0, Cd0=0.05, CL_alpha=0.05, alpha_max_LD=10.0):
        self.mass = mass
        self.reference_area = reference_area
        self.nose_radius = nose_radius
        self.ablation_threshold = ablation_threshold
        self.Cd0 = Cd0
        self.CL_alpha = CL_alpha
        self.alpha_max_LD = alpha_max_LD
        self.initial_mass = mass


class ReentryInitialConditions:
    def __init__(self, altitude=120.0, velocity=7.8, flight_path_angle=-3.0,
                 heading_angle=0.0, latitude=0.0, longitude=0.0):
        self.altitude = altitude
        self.velocity = velocity
        self.flight_path_angle = flight_path_angle
        self.heading_angle = heading_angle
        self.latitude = latitude
        self.longitude = longitude


def reentry_equations(t, state, vehicle, reentry_mode='ballistic', alpha=0.0, bank_angle=0.0):
    r, lon, lat, v, gamma, chi, m, T_surf = state
    
    altitude = r - R_EARTH
    altitude_clamped = np.clip(altitude, 0, 120)
    
    rho, T_atm, p = standard_atmosphere_1976(altitude_clamped)
    
    M = mach_number(v * 1000, T_atm)
    
    if reentry_mode == 'ballistic':
        alpha_eff = 0.0
        bank_eff = 0.0
    else:
        alpha_eff = alpha
        bank_eff = bank_angle
    
    Cd, CL = aerodynamic_coefficients(M, alpha_eff, vehicle.Cd0, vehicle.CL_alpha, vehicle.alpha_max_LD)
    
    q = stagnation_heat_flux(v, rho, vehicle.nose_radius)
    
    dT_surf_dt = surface_temperature_derivative(q, T_surf)
    
    dm_dt = -ablation_rate(q, T_surf, vehicle.ablation_threshold) * vehicle.reference_area
    
    gamma_rad = np.deg2rad(gamma)
    chi_rad = np.deg2rad(chi)
    lat_rad = np.deg2rad(lat)
    
    g = MU_EARTH / (r ** 2)
    
    q_dyn = 0.5 * rho * (v * 1000) ** 2
    D = q_dyn * vehicle.reference_area * Cd / m
    L = q_dyn * vehicle.reference_area * CL / m
    
    D_km = D / 1000.0
    L_km = L / 1000.0
    
    bank_rad = np.deg2rad(bank_eff)
    L_vertical = L_km * np.cos(bank_rad)
    L_horizontal = L_km * np.sin(bank_rad)
    
    dr_dt = v * np.sin(gamma_rad)
    
    dlon_dt = (v * np.cos(gamma_rad) * np.sin(chi_rad)) / (r * np.cos(lat_rad))
    dlat_dt = (v * np.cos(gamma_rad) * np.cos(chi_rad)) / r
    
    omega = OMEGA_EARTH
    a_coriolis_1 = -2 * omega * v * np.cos(gamma_rad) * np.sin(chi_rad) * np.sin(lat_rad)
    a_coriolis_2 = 2 * omega * v * np.cos(gamma_rad) * np.cos(chi_rad) * np.sin(lat_rad)
    a_coriolis_3 = 2 * omega * v * np.cos(gamma_rad) * np.sin(chi_rad) * np.cos(lat_rad)
    
    a_cent_1 = omega ** 2 * r * np.cos(lat_rad) ** 2 * np.sin(gamma_rad)
    a_cent_2 = omega ** 2 * r * np.cos(lat_rad) * np.sin(lat_rad) * np.cos(gamma_rad) * np.cos(chi_rad)
    a_cent_3 = -omega ** 2 * r * np.cos(lat_rad) * np.sin(lat_rad) * np.cos(gamma_rad) * np.sin(chi_rad)
    
    dv_dt = -D_km - g * np.sin(gamma_rad) + a_coriolis_1 + a_cent_1
    
    dgamma_dt = (L_vertical / v - g * np.cos(gamma_rad) / v + 
                 v * np.cos(gamma_rad) / r + 
                 a_coriolis_2 / v + a_cent_2 / v)
    
    dchi_dt = (L_horizontal / (v * np.cos(gamma_rad)) + 
               v * np.cos(gamma_rad) * np.sin(chi_rad) * np.tan(lat_rad) / r +
               a_coriolis_3 / (v * np.cos(gamma_rad)) + a_cent_3 / (v * np.cos(gamma_rad)))
    
    lon_deg = np.rad2deg(dlon_dt)
    lat_deg = np.rad2deg(dlat_dt)
    gamma_deg = np.rad2deg(dgamma_dt)
    chi_deg = np.rad2deg(dchi_dt)
    
    return [dr_dt, lon_deg, lat_deg, dv_dt, gamma_deg, chi_deg, dm_dt, dT_surf_dt]


def simulate_reentry(vehicle, initial_conditions, reentry_mode='ballistic', 
                     alpha=0.0, bank_angle=0.0, t_max=2000.0, dt_eval=1.0):
    r0 = R_EARTH + initial_conditions.altitude
    lon0 = initial_conditions.longitude
    lat0 = initial_conditions.latitude
    v0 = initial_conditions.velocity
    gamma0 = initial_conditions.flight_path_angle
    chi0 = initial_conditions.heading_angle
    m0 = vehicle.mass
    T_surf0 = 250.0
    
    state0 = [r0, lon0, lat0, v0, gamma0, chi0, m0, T_surf0]
    
    def event_impact(t, state):
        return state[0] - R_EARTH - 0.1
    
    event_impact.terminal = True
    event_impact.direction = -1
    
    def event_max_heat(t, state):
        return 0.0
    
    t_eval = np.arange(0, t_max, dt_eval)
    
    sol = solve_ivp(
        reentry_equations,
        [0, t_max],
        state0,
        args=(vehicle, reentry_mode, alpha, bank_angle),
        method='RK45',
        t_eval=t_eval,
        events=event_impact,
        rtol=1e-8,
        atol=1e-10
    )
    
    if not sol.success:
        raise RuntimeError(f"Simulation failed: {sol.message}")
    
    times = sol.t
    states = sol.y.T
    
    r_vals = states[:, 0]
    lon_vals = states[:, 1]
    lat_vals = states[:, 2]
    v_vals = states[:, 3]
    gamma_vals = states[:, 4]
    chi_vals = states[:, 5]
    m_vals = states[:, 6]
    T_surf_vals = states[:, 7]
    
    altitudes = r_vals - R_EARTH
    
    heat_fluxes = []
    overloads = []
    mach_numbers = []
    dynamic_pressures = []
    
    for i in range(len(times)):
        alt_km = altitudes[i]
        alt_clamped = np.clip(alt_km, 0, 120)
        rho, T_atm, p = standard_atmosphere_1976(alt_clamped)
        
        q = stagnation_heat_flux(v_vals[i], rho, vehicle.nose_radius)
        heat_fluxes.append(q)
        
        M = mach_number(v_vals[i] * 1000, T_atm)
        mach_numbers.append(M)
        
        q_dyn = 0.5 * rho * (v_vals[i] * 1000) ** 2
        
        if reentry_mode == 'ballistic':
            alpha_eff = 0.0
        else:
            alpha_eff = alpha
        
        Cd, CL = aerodynamic_coefficients(M, alpha_eff, vehicle.Cd0, vehicle.CL_alpha, vehicle.alpha_max_LD)
        D = q_dyn * vehicle.reference_area * Cd
        L = q_dyn * vehicle.reference_area * CL
        
        overload = np.sqrt(D**2 + L**2) / (m_vals[i] * 9.81)
        overloads.append(overload)
        dynamic_pressures.append(q_dyn)
    
    heat_fluxes = np.array(heat_fluxes)
    overloads = np.array(overloads)
    mach_numbers = np.array(mach_numbers)
    dynamic_pressures = np.array(dynamic_pressures)
    
    max_q_idx = np.argmax(heat_fluxes)
    max_g_idx = np.argmax(overloads)
    
    ablation_start_idx = None
    for i in range(len(times)):
        if T_surf_vals[i] >= vehicle.ablation_threshold and ablation_start_idx is None:
            ablation_start_idx = i
            break
    
    results = {
        'times': times,
        'altitudes': altitudes,
        'velocities': v_vals,
        'longitudes': lon_vals,
        'latitudes': lat_vals,
        'flight_path_angles': gamma_vals,
        'heading_angles': chi_vals,
        'masses': m_vals,
        'surface_temperatures': T_surf_vals,
        'heat_fluxes': heat_fluxes,
        'overloads': overloads,
        'mach_numbers': mach_numbers,
        'dynamic_pressures': dynamic_pressures,
        'max_heat_flux': heat_fluxes[max_q_idx],
        'max_heat_flux_time': times[max_q_idx],
        'max_heat_flux_alt': altitudes[max_q_idx],
        'max_surface_temp': np.max(T_surf_vals),
        'max_overload': overloads[max_g_idx],
        'max_overload_time': times[max_g_idx],
        'max_overload_alt': altitudes[max_g_idx],
        'ablation_start_time': times[ablation_start_idx] if ablation_start_idx is not None else None,
        'ablation_start_alt': altitudes[ablation_start_idx] if ablation_start_idx is not None else None,
        'impact_longitude': lon_vals[-1],
        'impact_latitude': lat_vals[-1],
        'total_time': times[-1],
        'mass_loss': vehicle.initial_mass - m_vals[-1]
    }
    
    return results


def simulate_both_modes(vehicle, initial_conditions, alpha_lifting=5.0, bank_angle=30.0):
    results_ballistic = simulate_reentry(
        vehicle, initial_conditions,
        reentry_mode='ballistic',
        alpha=0.0,
        bank_angle=0.0
    )
    
    vehicle2 = ReentryVehicle(
        mass=vehicle.initial_mass,
        reference_area=vehicle.reference_area,
        nose_radius=vehicle.nose_radius,
        ablation_threshold=vehicle.ablation_threshold,
        Cd0=vehicle.Cd0,
        CL_alpha=vehicle.CL_alpha,
        alpha_max_LD=vehicle.alpha_max_LD
    )
    
    results_lifting = simulate_reentry(
        vehicle2, initial_conditions,
        reentry_mode='lifting',
        alpha=alpha_lifting,
        bank_angle=bank_angle
    )
    
    return results_ballistic, results_lifting
