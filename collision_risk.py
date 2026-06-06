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


def compute_collision_probability(b_plane_pos, cov_b_plane, collision_radius, grid_size=200, method='adaptive'):
    det = np.linalg.det(cov_b_plane)
    if det <= 0 or collision_radius <= 0:
        return 0.0
    
    inv_cov = np.linalg.inv(cov_b_plane)
    norm_factor = 1.0 / (2 * np.pi * np.sqrt(det))
    
    dist_from_center = np.linalg.norm(b_plane_pos)
    
    if method == 'analytical' or dist_from_center > 10 * collision_radius:
        sigma1 = np.sqrt(cov_b_plane[0, 0])
        sigma2 = np.sqrt(cov_b_plane[1, 1])
        sigma_avg = np.sqrt(sigma1 * sigma2)
        
        if collision_radius < sigma_avg / 3:
            x, y = b_plane_pos
            exponent = -0.5 * (x * inv_cov[0, 0] * x + 
                               2 * x * inv_cov[0, 1] * y + 
                               y * inv_cov[1, 1] * y)
            pdf_at_center = norm_factor * np.exp(exponent)
            prob = np.pi * collision_radius**2 * pdf_at_center
            return min(prob, 1.0)
    
    max_sigma = 8
    sigma_max = max(np.sqrt(cov_b_plane[0, 0]), np.sqrt(cov_b_plane[1, 1]))
    range_needed = max(collision_radius, dist_from_center + max_sigma * sigma_max)
    
    x = np.linspace(-range_needed, range_needed, grid_size)
    y = np.linspace(-range_needed, range_needed, grid_size)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    X, Y = np.meshgrid(x, y)
    
    in_circle = (X**2 + Y**2) <= collision_radius**2
    
    if not np.any(in_circle):
        return 0.0
    
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


def estimate_dv_effect_fast(elements1, elements2, dv_vec, t_maneuver, t_ca):
    r0, v0 = get_state_at_time(elements1, t_maneuver)
    v_new = v0 + dv_vec
    from orbit_core import rv_to_kepler
    new_elements = rv_to_kepler(r0, v_new, units=elements1.units)
    
    dt = t_ca - t_maneuver
    if dt < 0:
        return None
    
    try:
        el1_new = propagate_orbit(new_elements, 0, dt)
        r1_new, v1_new = kepler_to_rv(el1_new)
        
        el2_at_ca = propagate_orbit(elements2, 0, dt)
        r2, v2 = kepler_to_rv(el2_at_ca)
        
        r_rel = r1_new - r2
        v_rel = v1_new - v2
        
        b_plane_pos, proj_matrix = project_to_b_plane(r_rel, v_rel)
        
        return {
            'b_plane_pos': b_plane_pos,
            'r_rel': r_rel,
            'v_rel': v_rel,
            'min_distance': np.linalg.norm(r_rel)
        }
    except Exception:
        return None


def binary_search_dv(elements1, elements2, cov1, cov2, radius1, radius2,
                     direction, t_maneuver, t_ca, target_prob,
                     max_dv=1.0, tol=1e-5):
    low = 0.0
    high = max_dv
    
    result_zero = compute_maneuver_effect(
        elements1, elements2, cov1, cov2,
        radius1, radius2, np.zeros(3), t_maneuver, t_ca
    )
    
    if result_zero['collision_probability'] <= target_prob:
        return 0.0, result_zero
    
    result_max = compute_maneuver_effect(
        elements1, elements2, cov1, cov2,
        radius1, radius2, high * direction, t_maneuver, t_ca
    )
    if result_max['collision_probability'] > target_prob:
        return None, None
    
    for _ in range(15):
        mid = (low + high) / 2
        result_mid = compute_maneuver_effect(
            elements1, elements2, cov1, cov2,
            radius1, radius2, mid * direction, t_maneuver, t_ca
        )
        
        if result_mid['collision_probability'] <= target_prob:
            high = mid
            best_result = result_mid
        else:
            low = mid
        
        if high - low < tol:
            break
    
    return high, best_result


def find_min_dv_maneuver(elements1, elements2, cov1, cov2,
                         radius1=5.0, radius2=5.0,
                         t_ca=None, t_maneuver_offset=6*3600,
                         target_prob=1e-6, max_dv=1.0):
    if t_ca is None:
        approach = find_closest_approach(elements1, elements2, dt_coarse=60)
        t_ca = approach['t_closest']
    
    t_maneuver = max(0, t_ca - t_maneuver_offset)
    
    r1, v1 = get_state_at_time(elements1, t_maneuver)
    radial_dir = r1 / np.linalg.norm(r1)
    along_track_dir = v1 / np.linalg.norm(v1)
    normal_dir = np.cross(along_track_dir, radial_dir)
    normal_dir = normal_dir / np.linalg.norm(normal_dir)
    
    base_result = compute_maneuver_effect(
        elements1, elements2, cov1, cov2,
        radius1, radius2, np.zeros(3), t_maneuver, t_ca
    )
    
    if base_result['collision_probability'] <= target_prob:
        return {
            'success': True,
            'dv_vec': np.zeros(3),
            'dv_magnitude': 0.0,
            't_maneuver': t_maneuver,
            'result_after_maneuver': base_result,
            'radial_component': 0.0,
            'along_track_component': 0.0,
            'normal_component': 0.0,
            'message': '当前碰撞概率已在目标值以下，无需机动'
        }
    
    directions = [
        ('radial+', radial_dir),
        ('radial-', -radial_dir),
        ('along+', along_track_dir),
        ('along-', -along_track_dir),
        ('normal+', normal_dir),
        ('normal-', -normal_dir),
    ]
    
    best_dv_mag = float('inf')
    best_dv_vec = None
    best_result = None
    best_dir_name = None
    
    for dir_name, direction in directions:
        dv_mag, result = binary_search_dv(
            elements1, elements2, cov1, cov2,
            radius1, radius2, direction,
            t_maneuver, t_ca, target_prob,
            max_dv=max_dv, tol=1e-4
        )
        
        if dv_mag is not None and dv_mag < best_dv_mag:
            best_dv_mag = dv_mag
            best_dv_vec = dv_mag * direction
            best_result = result
            best_dir_name = dir_name
    
    if best_dv_vec is not None:
        return {
            'success': True,
            'dv_vec': best_dv_vec,
            'dv_magnitude': best_dv_mag,
            't_maneuver': t_maneuver,
            'result_after_maneuver': best_result,
            'radial_component': np.dot(best_dv_vec, radial_dir),
            'along_track_component': np.dot(best_dv_vec, along_track_dir),
            'normal_component': np.dot(best_dv_vec, normal_dir),
            'optimal_direction': best_dir_name
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
                    target_prob=1e-6, t_maneuver_offset=6*3600,
                    compute_evolution=True, evolution_interval='day'):
    if debris_radii is None:
        debris_radii = [5.0] * len(debris_list)
    
    results = []
    
    max_evolution_debris = 20
    evolution_count = 0
    
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
        
        if compute_evolution and (result['collision_probability'] > 1e-8 or evolution_count < max_evolution_debris):
            evolution = collision_risk_time_evolution(
                main_elements, debris_elements,
                main_cov, debris_cov,
                main_radius, debris_radius,
                t_start, t_end,
                interval=evolution_interval,
                dt_coarse=dt_coarse
            )
            debris_result['evolution'] = evolution
            debris_result['probability_trend'] = evolution['probabilities']
            evolution_count += 1
        else:
            debris_result['probability_trend'] = np.array([0.0, 0.0])
        
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


def collision_risk_time_evolution(elements1, elements2, cov1, cov2,
                                   radius1=5.0, radius2=5.0,
                                   t_start=0, t_end=7*86400,
                                   interval='day', dt_coarse=30):
    if interval == 'hour':
        interval_seconds = 3600
        adaptive_dt = max(60, dt_coarse * 4)
    else:
        interval_seconds = 86400
        adaptive_dt = dt_coarse
    
    total_duration = t_end - t_start
    num_intervals = max(1, int(np.ceil(total_duration / interval_seconds)))
    
    max_intervals = 50
    if num_intervals > max_intervals:
        if interval == 'hour':
            interval_seconds = int(np.ceil(total_duration / max_intervals))
            num_intervals = max_intervals
    
    quick_dt = 300
    t_quick = np.arange(t_start, t_end, quick_dt)
    min_distances_quick = []
    
    for t in t_quick:
        r1, v1 = get_state_at_time(elements1, t)
        r2, v2 = get_state_at_time(elements2, t)
        dist = np.linalg.norm(r1 - r2)
        min_distances_quick.append(dist)
    
    min_distances_quick = np.array(min_distances_quick)
    overall_min_dist = np.min(min_distances_quick)
    safe_distance = (radius1 + radius2) * 100
    
    intervals = []
    interval_probs = []
    interval_distances = []
    interval_times = []
    
    for i in range(num_intervals):
        interval_start = t_start + i * interval_seconds
        interval_end = min(t_start + (i + 1) * interval_seconds, t_end)
        
        if interval_end <= interval_start:
            continue
        
        mask = (t_quick >= interval_start) & (t_quick < interval_end)
        if np.any(mask):
            interval_min_quick = np.min(min_distances_quick[mask])
        else:
            interval_min_quick = float('inf')
        
        if interval_min_quick > safe_distance:
            intervals.append(i + 1)
            interval_probs.append(1e-15)
            interval_distances.append(interval_min_quick)
            interval_times.append((interval_start + interval_end) / 2)
            continue
        
        result = collision_risk_analysis(
            elements1, elements2, cov1, cov2,
            radius1=radius1, radius2=radius2,
            t_start=interval_start, t_end=interval_end,
            dt_coarse=adaptive_dt
        )
        
        intervals.append(i + 1)
        interval_probs.append(result['collision_probability'])
        interval_distances.append(result['min_distance'])
        interval_times.append((interval_start + interval_end) / 2)
    
    return {
        'intervals': intervals,
        'probabilities': np.array(interval_probs),
        'min_distances': np.array(interval_distances),
        'interval_mid_times': np.array(interval_times),
        'interval_seconds': interval_seconds,
        'num_intervals': num_intervals,
        'interval_type': interval
    }


def monte_carlo_collision_prob(elements1, elements2, cov1, cov2,
                                radius1=5.0, radius2=5.0,
                                t_start=0, t_end=7*86400,
                                n_samples=50, dt_coarse=30):
    nominal_result = collision_risk_analysis(
        elements1, elements2, cov1, cov2,
        radius1, radius2, t_start, t_end, dt_coarse
    )
    
    prob_samples = []
    
    try:
        L1 = np.linalg.cholesky(cov1)
        L2 = np.linalg.cholesky(cov2)
    except np.linalg.LinAlgError:
        return {
            'nominal_prob': nominal_result['collision_probability'],
            'prob_samples': [nominal_result['collision_probability']] * n_samples,
            'lower_90': nominal_result['collision_probability'],
            'upper_90': nominal_result['collision_probability'],
            'mean_prob': nominal_result['collision_probability'],
            'median_prob': nominal_result['collision_probability']
        }
    
    for _ in range(n_samples):
        z1 = np.random.randn(3)
        z2 = np.random.randn(3)
        
        perturb1 = L1 @ z1
        perturb2 = L2 @ z2
        
        r1_orig, v1_orig = get_state_at_time(elements1, nominal_result['t_closest'])
        r2_orig, v2_orig = get_state_at_time(elements2, nominal_result['t_closest'])
        
        r1_perturbed = r1_orig + perturb1
        r2_perturbed = r2_orig + perturb2
        
        r_rel_perturbed = r1_perturbed - r2_perturbed
        v_rel = v1_orig - v2_orig
        
        b_plane_pos_pert, proj_matrix = project_to_b_plane(r_rel_perturbed, v_rel)
        cov_b_plane_pert = project_covariance_to_b_plane(cov1, cov2, proj_matrix)
        
        prob_pert = compute_collision_probability(
            b_plane_pos_pert, cov_b_plane_pert, radius1 + radius2
        )
        prob_samples.append(prob_pert)
    
    prob_samples = np.array(prob_samples)
    prob_samples_sorted = np.sort(prob_samples)
    
    lower_idx = int(n_samples * 0.05)
    upper_idx = int(n_samples * 0.95)
    
    return {
        'nominal_prob': nominal_result['collision_probability'],
        'prob_samples': prob_samples,
        'lower_90': prob_samples_sorted[lower_idx],
        'upper_90': prob_samples_sorted[upper_idx],
        'mean_prob': np.mean(prob_samples),
        'median_prob': np.median(prob_samples),
        'nominal_result': nominal_result
    }


def generate_collision_statistics_with_ci(main_elements, debris_list, main_cov, debris_cov_list,
                                           main_radius=5.0, debris_radii=None,
                                           time_windows=None, t_start=0, t_end=7*86400,
                                           n_mc_samples=50):
    if time_windows is None:
        time_windows = [1, 2, 3, 5, 7, 14, 30]
    
    if debris_radii is None:
        debris_radii = [5.0] * len(debris_list)
    
    max_window = max(time_windows)
    max_window_end = t_start + max_window * 86400
    
    debris_mc_data = []
    for debris_elements, debris_cov, debris_radius in zip(debris_list, debris_cov_list, debris_radii):
        mc_result = monte_carlo_collision_prob(
            main_elements, debris_elements,
            main_cov, debris_cov,
            main_radius, debris_radius,
            t_start, max_window_end,
            n_samples=n_mc_samples
        )
        nominal_max = mc_result['nominal_prob']
        lower_max = mc_result['lower_90']
        upper_max = mc_result['upper_90']
        
        ci_ratio_lower = lower_max / nominal_max if nominal_max > 0 else 1.0
        ci_ratio_upper = upper_max / nominal_max if nominal_max > 0 else 1.0
        
        debris_mc_data.append({
            'nominal_max': nominal_max,
            'ci_ratio_lower': ci_ratio_lower,
            'ci_ratio_upper': ci_ratio_upper
        })
    
    cumulative_probs = []
    
    for window_days in time_windows:
        window_end = t_start + window_days * 86400
        window_probs_nominal = []
        window_probs_lower = []
        window_probs_upper = []
        
        for i, (debris_elements, debris_cov, debris_radius) in enumerate(zip(debris_list, debris_cov_list, debris_radii)):
            result = collision_risk_analysis(
                main_elements, debris_elements,
                main_cov, debris_cov,
                main_radius, debris_radius,
                t_start, window_end,
                dt_coarse=60
            )
            nominal_prob = result['collision_probability']
            window_probs_nominal.append(nominal_prob)
            
            mc_data = debris_mc_data[i]
            window_probs_lower.append(nominal_prob * mc_data['ci_ratio_lower'])
            window_probs_upper.append(nominal_prob * mc_data['ci_ratio_upper'])
        
        prob_any_nominal = 1.0 - np.prod(1.0 - np.array(window_probs_nominal))
        prob_any_lower = 1.0 - np.prod(1.0 - np.array(window_probs_lower))
        prob_any_upper = 1.0 - np.prod(1.0 - np.array(window_probs_upper))
        
        cumulative_probs.append({
            'window_days': window_days,
            'prob_any_collision': prob_any_nominal,
            'prob_lower_90': prob_any_lower,
            'prob_upper_90': prob_any_upper,
            'individual_probs': window_probs_nominal,
            'individual_lower': window_probs_lower,
            'individual_upper': window_probs_upper
        })
    
    distances = []
    for debris_elements, debris_cov, debris_radius in zip(debris_list, debris_cov_list, debris_radii):
        result = collision_risk_analysis(
            main_elements, debris_elements,
            main_cov, debris_cov,
            main_radius, debris_radius,
            t_start, t_end
        )
        distances.append(result['min_distance'])
    
    return {
        'cumulative_probs': cumulative_probs,
        'distances': np.array(distances)
    }
