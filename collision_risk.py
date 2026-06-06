import numpy as np
from orbit_core import (
    MU_EARTH, KeplerElements, kepler_to_rv, propagate_orbit
)


def golden_section_search(func, a, b, tol=1e-6, max_iter=100):
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = func(c)
    fd = func(d)
    
    for _ in range(max_iter):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = func(d)
        if abs(b - a) < tol:
            break
    
    return (a + b) / 2


def compute_relative_distance(elements1, elements2, t):
    el1 = propagate_orbit(elements1, 0, t)
    el2 = propagate_orbit(elements2, 0, t)
    r1, _ = kepler_to_rv(el1)
    r2, _ = kepler_to_rv(el2)
    return np.linalg.norm(r1 - r2)


def find_closest_approach(elements1, elements2, t_start=0, t_end=7*86400, dt_coarse=30):
    times_coarse = np.arange(t_start, t_end, dt_coarse)
    distances = np.array([compute_relative_distance(elements1, elements2, t) for t in times_coarse])
    
    idx_min = np.argmin(distances)
    t_approx = times_coarse[idx_min]
    
    search_window = dt_coarse * 2
    t_low = max(t_start, t_approx - search_window)
    t_high = min(t_end, t_approx + search_window)
    
    def dist_func(t):
        return compute_relative_distance(elements1, elements2, t)
    
    t_closest = golden_section_search(dist_func, t_low, t_high, tol=1e-3)
    min_distance = compute_relative_distance(elements1, elements2, t_closest)
    
    return {
        't_closest': t_closest,
        'min_distance': min_distance,
        't_coarse': times_coarse,
        'distances_coarse': distances
    }


def get_state_at_time(elements, t):
    el = propagate_orbit(elements, 0, t)
    r, v = kepler_to_rv(el)
    return r, v


def project_to_b_plane(r_rel, v_rel):
    v_rel_mag = np.linalg.norm(v_rel)
    if v_rel_mag < 1e-10:
        return np.array([0.0, 0.0]), np.eye(2)
    
    h = np.cross(v_rel, r_rel)
    h_mag = np.linalg.norm(h)
    
    if h_mag < 1e-10:
        return np.array([0.0, 0.0]), np.eye(2)
    
    i_hat = v_rel / v_rel_mag
    j_hat = np.cross(v_rel, h) / (v_rel_mag * h_mag)
    k_hat = h / h_mag
    
    xi = np.dot(r_rel, j_hat)
    zeta = np.dot(r_rel, k_hat)
    
    return np.array([xi, zeta]), np.array([j_hat, k_hat])


def project_covariance_to_b_plane(cov1, cov2, projection_matrix):
    cov_combined = cov1 + cov2
    projected = projection_matrix @ cov_combined @ projection_matrix.T
    return projected


def compute_collision_probability(b_plane_pos, cov_b_plane, collision_radius, grid_size=200):
    det = np.linalg.det(cov_b_plane)
    if det <= 0 or collision_radius <= 0:
        return 0.0
    
    inv_cov = np.linalg.inv(cov_b_plane)
    norm_factor = 1.0 / (2 * np.pi * np.sqrt(det))
    
    max_sigma = 10
    sigma_max = max(np.sqrt(cov_b_plane[0, 0]), np.sqrt(cov_b_plane[1, 1]))
    
    dist_from_center = np.linalg.norm(b_plane_pos)
    
    range_needed = max(collision_radius, dist_from_center + max_sigma * sigma_max)
    
    x = np.linspace(-range_needed, range_needed, grid_size)
    y = np.linspace(-range_needed, range_needed, grid_size)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    X, Y = np.meshgrid(x, y)
    
    in_circle = (X**2 + Y**2) <= collision_radius**2
    
    dX = X - b_plane_pos[0]
    dY = Y - b_plane_pos[1]
    
    exponent = -0.5 * (dX * inv_cov[0, 0] * dX + 
                       2 * dX * inv_cov[0, 1] * dY + 
                       dY * inv_cov[1, 1] * dY)
    
    pdf = norm_factor * np.exp(exponent)
    
    prob = np.sum(pdf[in_circle]) * dx * dy
    
    return min(prob, 1.0)


def collision_risk_analysis(elements1, elements2, cov1, cov2, 
                            radius1=5.0, radius2=5.0, 
                            t_start=0, t_end=7*86400, dt_coarse=30):
    approach = find_closest_approach(elements1, elements2, t_start, t_end, dt_coarse)
    t_ca = approach['t_closest']
    
    r1, v1 = get_state_at_time(elements1, t_ca)
    r2, v2 = get_state_at_time(elements2, t_ca)
    
    r_rel = r1 - r2
    v_rel = v1 - v2
    
    b_plane_pos, proj_matrix = project_to_b_plane(r_rel, v_rel)
    
    cov_b_plane = project_covariance_to_b_plane(cov1, cov2, proj_matrix)
    
    collision_radius = radius1 + radius2
    
    prob = compute_collision_probability(b_plane_pos, cov_b_plane, collision_radius)
    
    return {
        't_closest': t_ca,
        'min_distance': approach['min_distance'],
        'b_plane_pos': b_plane_pos,
        'cov_b_plane': cov_b_plane,
        'collision_radius': collision_radius,
        'collision_probability': prob,
        'r1_ca': r1,
        'v1_ca': v1,
        'r2_ca': r2,
        'v2_ca': v2,
        'v_rel': v_rel,
        'proj_matrix': proj_matrix,
        'approach_data': approach
    }


def apply_maneuver(elements, dv_vec, t_maneuver):
    r0, v0 = get_state_at_time(elements, t_maneuver)
    v_new = v0 + dv_vec
    
    from orbit_core import rv_to_kepler
    new_elements = rv_to_kepler(r0, v_new, units=elements.units)
    return new_elements


def compute_maneuver_effect(elements1, elements2, cov1, cov2, 
                            radius1, radius2, dv_vec, t_maneuver, t_ca):
    new_elements1 = apply_maneuver(elements1, dv_vec, t_maneuver)
    
    result = collision_risk_analysis(
        new_elements1, elements2, cov1, cov2,
        radius1, radius2,
        t_start=t_maneuver, t_end=t_ca + 3600,
        dt_coarse=30
    )
    
    return result


def find_min_dv_maneuver(elements1, elements2, cov1, cov2,
                         radius1=5.0, radius2=5.0,
                         t_ca=None, t_maneuver_offset=6*3600,
                         target_prob=1e-6, max_dv=1.0, dv_step=0.001):
    if t_ca is None:
        approach = find_closest_approach(elements1, elements2)
        t_ca = approach['t_closest']
    
    t_maneuver = max(0, t_ca - t_maneuver_offset)
    
    r1, v1 = get_state_at_time(elements1, t_maneuver)
    v_mag = np.linalg.norm(v1)
    
    radial_dir = r1 / np.linalg.norm(r1)
    along_track_dir = v1 / v_mag
    normal_dir = np.cross(along_track_dir, radial_dir)
    normal_dir = normal_dir / np.linalg.norm(normal_dir)
    
    directions = [
        radial_dir, -radial_dir,
        along_track_dir, -along_track_dir,
        normal_dir, -normal_dir
    ]
    
    best_dv = None
    best_dv_mag = float('inf')
    best_result = None
    
    for direction in directions:
        dv_mag = 0.0
        while dv_mag <= max_dv:
            dv_vec = dv_mag * direction
            result = compute_maneuver_effect(
                elements1, elements2, cov1, cov2,
                radius1, radius2, dv_vec, t_maneuver, t_ca
            )
            
            if result['collision_probability'] <= target_prob:
                if dv_mag < best_dv_mag:
                    best_dv_mag = dv_mag
                    best_dv = dv_vec
                    best_result = result
                break
            
            dv_mag += dv_step
    
    combinations = [
        (radial_dir + along_track_dir),
        (radial_dir - along_track_dir),
        (radial_dir + normal_dir),
        (radial_dir - normal_dir),
        (along_track_dir + normal_dir),
        (along_track_dir - normal_dir),
        (radial_dir + along_track_dir + normal_dir)
    ]
    
    for comb in combinations:
        comb = comb / np.linalg.norm(comb)
        dv_mag = 0.0
        while dv_mag <= max_dv:
            dv_vec = dv_mag * comb
            result = compute_maneuver_effect(
                elements1, elements2, cov1, cov2,
                radius1, radius2, dv_vec, t_maneuver, t_ca
            )
            
            if result['collision_probability'] <= target_prob:
                if dv_mag < best_dv_mag:
                    best_dv_mag = dv_mag
                    best_dv = dv_vec
                    best_result = result
                break
            
            dv_mag += dv_step
    
    if best_dv is not None:
        return {
            'success': True,
            'dv_vec': best_dv,
            'dv_magnitude': best_dv_mag,
            't_maneuver': t_maneuver,
            'result_after_maneuver': best_result,
            'radial_component': np.dot(best_dv, radial_dir),
            'along_track_component': np.dot(best_dv, along_track_dir),
            'normal_component': np.dot(best_dv, normal_dir)
        }
    else:
        return {
            'success': False,
            'message': '在最大Δv限制内未找到有效规避方案'
        }


def batch_screening(main_elements, debris_list, main_cov, debris_cov_list,
                    main_radius=5.0, debris_radii=None,
                    t_start=0, t_end=7*86400, dt_coarse=30,
                    prob_threshold=1e-4,
                    target_prob=1e-6, t_maneuver_offset=6*3600):
    if debris_radii is None:
        debris_radii = [5.0] * len(debris_list)
    
    results = []
    
    for i, (debris_elements, debris_cov, debris_radius) in enumerate(zip(debris_list, debris_cov_list, debris_radii)):
        result = collision_risk_analysis(
            main_elements, debris_elements,
            main_cov, debris_cov,
            main_radius, debris_radius,
            t_start, t_end, dt_coarse
        )
        
        debris_result = {
            'debris_id': i + 1,
            'min_distance': result['min_distance'],
            't_closest': result['t_closest'],
            'collision_probability': result['collision_probability'],
            'exceeds_threshold': result['collision_probability'] > prob_threshold,
            'raw_result': result
        }
        
        if debris_result['exceeds_threshold']:
            maneuver = find_min_dv_maneuver(
                main_elements, debris_elements,
                main_cov, debris_cov,
                main_radius, debris_radius,
                t_ca=result['t_closest'],
                t_maneuver_offset=t_maneuver_offset,
                target_prob=target_prob
            )
            debris_result['maneuver'] = maneuver
        
        results.append(debris_result)
    
    results.sort(key=lambda x: x['collision_probability'], reverse=True)
    
    return results


def generate_collision_statistics(main_elements, debris_list, main_cov, debris_cov_list,
                                  main_radius=5.0, debris_radii=None,
                                  time_windows=None, t_start=0, t_end=7*86400):
    if time_windows is None:
        time_windows = [1, 2, 3, 5, 7]
    
    if debris_radii is None:
        debris_radii = [5.0] * len(debris_list)
    
    cumulative_probs = []
    distances = []
    
    for window_days in time_windows:
        window_end = t_start + window_days * 86400
        window_probs = []
        
        for debris_elements, debris_cov, debris_radius in zip(debris_list, debris_cov_list, debris_radii):
            result = collision_risk_analysis(
                main_elements, debris_elements,
                main_cov, debris_cov,
                main_radius, debris_radius,
                t_start, window_end
            )
            window_probs.append(result['collision_probability'])
            distances.append(result['min_distance'])
        
        prob_any = 1.0 - np.prod(1.0 - np.array(window_probs))
        cumulative_probs.append({
            'window_days': window_days,
            'prob_any_collision': prob_any,
            'individual_probs': window_probs
        })
    
    return {
        'cumulative_probs': cumulative_probs,
        'distances': np.array(distances)
    }


def get_default_covariance(position_sigma_km=0.1, velocity_sigma_km_s=0.001):
    cov = np.zeros((6, 6))
    cov[:3, :3] = np.eye(3) * position_sigma_km**2
    cov[3:, 3:] = np.eye(3) * velocity_sigma_km_s**2
    return cov


def get_position_covariance(position_sigma_km=0.1):
    return np.eye(3) * position_sigma_km**2
