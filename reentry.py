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


def debris_equations(t, state, Cd, area_mass_ratio):
    r, lon, lat, v, gamma, chi = state
    
    altitude = r - R_EARTH
    altitude_clamped = np.clip(altitude, 0, 120)
    
    rho, T_atm, p = standard_atmosphere_1976(altitude_clamped)
    
    M = mach_number(v * 1000, T_atm)
    
    if M < 1.0:
        Cd_eff = Cd
    elif M < 5.0:
        Cd_eff = Cd * (1.0 + 0.1 * (M - 1.0))
    else:
        Cd_eff = Cd * 1.4
    
    gamma_rad = np.deg2rad(gamma)
    chi_rad = np.deg2rad(chi)
    lat_rad = np.deg2rad(lat)
    
    g = MU_EARTH / (r ** 2)
    
    q_dyn = 0.5 * rho * (v * 1000) ** 2
    D = q_dyn * Cd_eff * area_mass_ratio
    D_km = D / 1000.0
    
    dr_dt = v * np.sin(gamma_rad)
    dlon_dt = (v * np.cos(gamma_rad) * np.sin(chi_rad)) / (r * np.cos(lat_rad))
    dlat_dt = (v * np.cos(gamma_rad) * np.cos(chi_rad)) / r
    
    omega = OMEGA_EARTH
    a_coriolis_1 = -2 * omega * v * np.cos(gamma_rad) * np.sin(chi_rad) * np.sin(lat_rad)
    a_cent_1 = omega ** 2 * r * np.cos(lat_rad) ** 2 * np.sin(gamma_rad)
    
    dv_dt = -D_km - g * np.sin(gamma_rad) + a_coriolis_1 + a_cent_1
    
    a_coriolis_2 = 2 * omega * v * np.cos(gamma_rad) * np.cos(chi_rad) * np.sin(lat_rad)
    a_cent_2 = omega ** 2 * r * np.cos(lat_rad) * np.sin(lat_rad) * np.cos(gamma_rad) * np.cos(chi_rad)
    
    dgamma_dt = (-g * np.cos(gamma_rad) / v + 
                 v * np.cos(gamma_rad) / r + 
                 a_coriolis_2 / v + a_cent_2 / v)
    
    a_coriolis_3 = 2 * omega * v * np.cos(gamma_rad) * np.sin(chi_rad) * np.cos(lat_rad)
    a_cent_3 = -omega ** 2 * r * np.cos(lat_rad) * np.sin(lat_rad) * np.cos(gamma_rad) * np.sin(chi_rad)
    
    dchi_dt = (v * np.cos(gamma_rad) * np.sin(chi_rad) * np.tan(lat_rad) / r +
               a_coriolis_3 / (v * np.cos(gamma_rad)) + a_cent_3 / (v * np.cos(gamma_rad)))
    
    lon_deg = np.rad2deg(dlon_dt)
    lat_deg = np.rad2deg(dlat_dt)
    gamma_deg = np.rad2deg(dgamma_dt)
    chi_deg = np.rad2deg(dchi_dt)
    
    return [dr_dt, lon_deg, lat_deg, dv_dt, gamma_deg, chi_deg]


def simulate_debris(breakup_state, Cd=1.2, area_mass_ratio=0.01, t_max=2000.0, dt_eval=1.0, rtol=1e-6, atol=1e-8):
    r0, lon0, lat0, v0, gamma0, chi0 = breakup_state
    
    state0 = [r0, lon0, lat0, v0, gamma0, chi0]
    
    def event_impact(t, state):
        return state[0] - R_EARTH - 0.01
    
    event_impact.terminal = True
    event_impact.direction = -1
    
    t_eval = np.arange(0, t_max, dt_eval)
    
    sol = solve_ivp(
        debris_equations,
        [0, t_max],
        state0,
        args=(Cd, area_mass_ratio),
        method='RK45',
        t_eval=t_eval,
        events=event_impact,
        rtol=rtol,
        atol=atol
    )
    
    if not sol.success:
        return None
    
    times = sol.t
    states = sol.y.T
    
    r_vals = states[:, 0]
    lon_vals = states[:, 1]
    lat_vals = states[:, 2]
    v_vals = states[:, 3]
    gamma_vals = states[:, 4]
    chi_vals = states[:, 5]
    
    altitudes = r_vals - R_EARTH
    
    result = {
        'times': times,
        'altitudes': altitudes,
        'velocities': v_vals,
        'longitudes': lon_vals,
        'latitudes': lat_vals,
        'flight_path_angles': gamma_vals,
        'heading_angles': chi_vals,
        'impact_longitude': lon_vals[-1],
        'impact_latitude': lat_vals[-1],
        'total_time': times[-1]
    }
    
    return result


def simulate_debris_field(vehicle, initial_conditions, breakup_threshold_pa=50000.0, 
                          num_debris=8, min_amr_factor=0.5, max_amr_factor=3.0,
                          velocity_perturbation=50.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    results_main = simulate_reentry(
        vehicle, initial_conditions,
        reentry_mode='ballistic',
        alpha=0.0,
        bank_angle=0.0
    )
    
    dynamic_pressures = results_main['dynamic_pressures']
    breakup_idx = None
    
    for i, q in enumerate(dynamic_pressures):
        if q >= breakup_threshold_pa:
            breakup_idx = i
            break
    
    if breakup_idx is None:
        return None
    
    breakup_r = R_EARTH + results_main['altitudes'][breakup_idx]
    breakup_lon = results_main['longitudes'][breakup_idx]
    breakup_lat = results_main['latitudes'][breakup_idx]
    breakup_v = results_main['velocities'][breakup_idx]
    breakup_gamma = results_main['flight_path_angles'][breakup_idx]
    breakup_chi = results_main['heading_angles'][breakup_idx]
    
    breakup_state = [breakup_r, breakup_lon, breakup_lat, breakup_v, breakup_gamma, breakup_chi]
    
    base_amr = vehicle.reference_area / vehicle.mass
    
    debris_list = []
    
    for i in range(num_debris):
        amr_factor = np.random.uniform(min_amr_factor, max_amr_factor)
        debris_amr = base_amr * amr_factor
        debris_Cd = np.random.uniform(1.0, 1.8)
        
        dv_pert = np.random.uniform(-velocity_perturbation, velocity_perturbation, 3)
        dv_pert_km_s = dv_pert / 1000.0
        
        gamma_rad = np.deg2rad(breakup_gamma)
        chi_rad = np.deg2rad(breakup_chi)
        
        vx = breakup_v * np.cos(gamma_rad) * np.cos(chi_rad) + dv_pert_km_s[0]
        vy = breakup_v * np.cos(gamma_rad) * np.sin(chi_rad) + dv_pert_km_s[1]
        vz = breakup_v * np.sin(gamma_rad) + dv_pert_km_s[2]
        
        v_pert = np.sqrt(vx**2 + vy**2 + vz**2)
        gamma_pert = np.rad2deg(np.arcsin(vz / v_pert))
        chi_pert = np.rad2deg(np.arctan2(vy, vx))
        
        debris_state = [breakup_r, breakup_lon, breakup_lat, v_pert, gamma_pert, chi_pert]
        
        debris_result = simulate_debris(
            debris_state,
            Cd=debris_Cd,
            area_mass_ratio=debris_amr,
            rtol=1e-5,
            atol=1e-7
        )
        
        if debris_result is not None:
            debris_info = {
                'id': i + 1,
                'area_mass_ratio': debris_amr,
                'ballistic_coefficient': 1.0 / (debris_Cd * debris_amr) if debris_Cd * debris_amr > 0 else 0,
                'Cd': debris_Cd,
                'result': debris_result,
                'impact_lon': debris_result['impact_longitude'],
                'impact_lat': debris_result['impact_latitude']
            }
            debris_list.append(debris_info)
    
    impact_lons = np.array([d['impact_lon'] for d in debris_list])
    impact_lats = np.array([d['impact_lat'] for d in debris_list])
    
    if len(impact_lons) >= 2:
        mean_lon = np.mean(impact_lons)
        mean_lat = np.mean(impact_lats)
        
        lon_span = np.max(impact_lons) - np.min(impact_lons)
        lat_span = np.max(impact_lats) - np.min(impact_lats)
        
        mean_lat_rad = np.deg2rad(mean_lat)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)
        
        east_km = (impact_lons - mean_lon) * km_per_deg_lon
        north_km = (impact_lats - mean_lat) * km_per_deg_lat
        
        coords_km = np.column_stack([east_km, north_km])
        
        if len(coords_km) >= 2:
            cov = np.cov(coords_km.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            n_std = 1.5
            major_axis_km = 2 * np.sqrt(eigenvalues[0]) * n_std if eigenvalues[0] > 0 else 0
            minor_axis_km = 2 * np.sqrt(eigenvalues[1]) * n_std if eigenvalues[1] > 0 else 0
            
            angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            angle_deg = np.rad2deg(angle_rad)
            
            major_axis_length = major_axis_km / km_per_deg_lat
            minor_axis_length = minor_axis_km / km_per_deg_lat
        else:
            major_axis_length = lon_span
            minor_axis_length = lat_span
            angle_deg = 0.0
    else:
        mean_lon = impact_lons[0] if len(impact_lons) > 0 else 0
        mean_lat = impact_lats[0] if len(impact_lats) > 0 else 0
        lon_span = 0
        lat_span = 0
        major_axis_length = 0
        minor_axis_length = 0
        angle_deg = 0
    
    result = {
        'breakup_altitude': results_main['altitudes'][breakup_idx],
        'breakup_time': results_main['times'][breakup_idx],
        'breakup_longitude': breakup_lon,
        'breakup_latitude': breakup_lat,
        'breakup_velocity': breakup_v,
        'breakup_dynamic_pressure': dynamic_pressures[breakup_idx],
        'debris': debris_list,
        'mean_impact_lon': mean_lon,
        'mean_impact_lat': mean_lat,
        'lon_span': lon_span,
        'lat_span': lat_span,
        'major_axis_length': major_axis_length,
        'minor_axis_length': minor_axis_length,
        'ellipse_angle_deg': angle_deg,
        'main_trajectory': results_main
    }
    
    return result


def fast_reentry_equations(t, state, vehicle):
    r, lon, lat, v, gamma, chi = state
    
    altitude = r - R_EARTH
    altitude_clamped = np.clip(altitude, 0, 120)
    
    rho, T_atm, p = standard_atmosphere_1976(altitude_clamped)
    
    M = mach_number(v * 1000, T_atm)
    
    Cd = vehicle.Cd0
    
    gamma_rad = np.deg2rad(gamma)
    chi_rad = np.deg2rad(chi)
    lat_rad = np.deg2rad(lat)
    
    g = MU_EARTH / (r ** 2)
    
    q_dyn = 0.5 * rho * (v * 1000) ** 2
    D = q_dyn * vehicle.reference_area * Cd / vehicle.mass
    D_km = D / 1000.0
    
    dr_dt = v * np.sin(gamma_rad)
    dlon_dt = (v * np.cos(gamma_rad) * np.sin(chi_rad)) / (r * np.cos(lat_rad))
    dlat_dt = (v * np.cos(gamma_rad) * np.cos(chi_rad)) / r
    
    omega = OMEGA_EARTH
    a_coriolis_1 = -2 * omega * v * np.cos(gamma_rad) * np.sin(chi_rad) * np.sin(lat_rad)
    a_cent_1 = omega ** 2 * r * np.cos(lat_rad) ** 2 * np.sin(gamma_rad)
    
    dv_dt = -D_km - g * np.sin(gamma_rad) + a_coriolis_1 + a_cent_1
    
    a_coriolis_2 = 2 * omega * v * np.cos(gamma_rad) * np.cos(chi_rad) * np.sin(lat_rad)
    a_cent_2 = omega ** 2 * r * np.cos(lat_rad) * np.sin(lat_rad) * np.cos(gamma_rad) * np.cos(chi_rad)
    
    dgamma_dt = (-g * np.cos(gamma_rad) / v + 
                 v * np.cos(gamma_rad) / r + 
                 a_coriolis_2 / v + a_cent_2 / v)
    
    a_coriolis_3 = 2 * omega * v * np.cos(gamma_rad) * np.sin(chi_rad) * np.cos(lat_rad)
    a_cent_3 = -omega ** 2 * r * np.cos(lat_rad) * np.sin(lat_rad) * np.cos(gamma_rad) * np.sin(chi_rad)
    
    dchi_dt = (v * np.cos(gamma_rad) * np.sin(chi_rad) * np.tan(lat_rad) / r +
               a_coriolis_3 / (v * np.cos(gamma_rad)) + a_cent_3 / (v * np.cos(gamma_rad)))
    
    lon_deg = np.rad2deg(dlon_dt)
    lat_deg = np.rad2deg(dlat_dt)
    gamma_deg = np.rad2deg(dgamma_dt)
    chi_deg = np.rad2deg(dchi_dt)
    
    return [dr_dt, lon_deg, lat_deg, dv_dt, gamma_deg, chi_deg]


def simulate_reentry_fast(vehicle, initial_conditions, t_max=2000.0):
    r0 = R_EARTH + initial_conditions.altitude
    lon0 = initial_conditions.longitude
    lat0 = initial_conditions.latitude
    v0 = initial_conditions.velocity
    gamma0 = initial_conditions.flight_path_angle
    chi0 = initial_conditions.heading_angle
    
    state0 = [r0, lon0, lat0, v0, gamma0, chi0]
    
    def event_impact(t, state):
        return state[0] - R_EARTH - 0.1
    
    event_impact.terminal = True
    event_impact.direction = -1
    
    sol = solve_ivp(
        fast_reentry_equations,
        [0, t_max],
        state0,
        args=(vehicle,),
        method='RK23',
        events=event_impact,
        rtol=1e-3,
        atol=1e-5
    )
    
    if not sol.success:
        return None
    
    states = sol.y.T
    
    return {
        'impact_longitude': states[-1, 1],
        'impact_latitude': states[-1, 2]
    }


def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0
    
    lon1_rad = np.deg2rad(lon1)
    lat1_rad = np.deg2rad(lat1)
    lon2_rad = np.deg2rad(lon2)
    lat2_rad = np.deg2rad(lat2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def analyze_reentry_window(vehicle, base_initial_conditions, 
                           target_lon, target_lat, allowed_radius_km=100.0,
                           gamma_min=-7.0, gamma_max=-1.0, gamma_step=0.5,
                           chi_delta=30.0, chi_step=5.0):
    base_gamma = base_initial_conditions.flight_path_angle
    base_chi = base_initial_conditions.heading_angle
    
    gamma_range = np.arange(gamma_min, gamma_max + gamma_step / 2, gamma_step)
    chi_min = base_chi - chi_delta
    chi_max = base_chi + chi_delta
    chi_range = np.arange(chi_min, chi_max + chi_step / 2, chi_step)
    
    results = []
    valid_params = []
    
    for gamma in gamma_range:
        for chi in chi_range:
            ic = ReentryInitialConditions(
                altitude=base_initial_conditions.altitude,
                velocity=base_initial_conditions.velocity,
                flight_path_angle=gamma,
                heading_angle=chi,
                latitude=base_initial_conditions.latitude,
                longitude=base_initial_conditions.longitude
            )
            
            result = simulate_reentry_fast(vehicle, ic)
            
            if result is not None:
                dist = haversine_distance(
                    target_lon, target_lat,
                    result['impact_longitude'], result['impact_latitude']
                )
                
                result_entry = {
                    'gamma': gamma,
                    'chi': chi,
                    'impact_lon': result['impact_longitude'],
                    'impact_lat': result['impact_latitude'],
                    'distance_km': dist,
                    'valid': dist <= allowed_radius_km
                }
                results.append(result_entry)
                
                if result_entry['valid']:
                    valid_params.append({
                        'flight_path_angle': gamma,
                        'heading_angle': chi,
                        'impact_longitude': result['impact_longitude'],
                        'impact_latitude': result['impact_latitude'],
                        'distance_km': dist
                    })
    
    gamma_grid, chi_grid = np.meshgrid(gamma_range, chi_range, indexing='ij')
    distance_grid = np.full_like(gamma_grid, np.nan)
    
    for res in results:
        i = np.where(gamma_range == res['gamma'])[0][0]
        j = np.where(chi_range == res['chi'])[0][0]
        distance_grid[i, j] = res['distance_km']
    
    valid_params_sorted = sorted(valid_params, key=lambda x: x['distance_km'])
    
    return {
        'gamma_range': gamma_range,
        'chi_range': chi_range,
        'gamma_grid': gamma_grid,
        'chi_grid': chi_grid,
        'distance_grid': distance_grid,
        'all_results': results,
        'valid_parameters': valid_params_sorted,
        'target_lon': target_lon,
        'target_lat': target_lat,
        'allowed_radius_km': allowed_radius_km
    }
