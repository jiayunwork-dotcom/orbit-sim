import numpy as np
from orbit_core import (
    R_EARTH, MU_EARTH, KeplerElements,
    kepler_to_rv, propagate_orbit
)


def walker_delta_constellation(total_sats, num_planes, phasing, altitude, inclination, e=0):
    sats_per_plane = total_sats // num_planes
    raan_spacing = 360.0 / num_planes
    nu_spacing = 360.0 / sats_per_plane
    
    constellation = []
    
    for plane_idx in range(num_planes):
        raan = plane_idx * raan_spacing
        phase_offset = (plane_idx * phasing * nu_spacing) % 360
        
        for sat_idx in range(sats_per_plane):
            nu = (sat_idx * nu_spacing + phase_offset) % 360
            
            elements = KeplerElements(
                R_EARTH + altitude,
                e,
                inclination,
                raan,
                0,
                nu,
                units='km_deg'
            )
            
            constellation.append({
                'plane': plane_idx,
                'sat_index': sat_idx,
                'elements': elements
            })
    
    return constellation


def walker_alpha_constellation(total_sats, num_planes, phasing, altitude, inclination, e=0):
    return walker_delta_constellation(total_sats, num_planes, phasing, altitude, inclination, e)


def get_constellation_positions(constellation, t=0):
    positions = []
    for sat in constellation:
        elements = sat['elements']
        if t > 0:
            elements = propagate_orbit(elements, 0, t)
        r, v = kepler_to_rv(elements)
        positions.append({
            'position': r,
            'velocity': v,
            'plane': sat['plane'],
            'sat_index': sat['sat_index'],
            'elements': elements
        })
    return positions


def check_visibility(sat_pos, ground_point, min_elevation=10):
    sat_vec = sat_pos - ground_point
    ground_norm = ground_point / np.linalg.norm(ground_point)
    
    elevation = 90 - np.rad2deg(np.arccos(
        np.dot(sat_vec, ground_norm) / (np.linalg.norm(sat_vec) * np.linalg.norm(ground_norm))
    ))
    
    return elevation >= min_elevation


def compute_coverage(constellation, lat_points=36, lon_points=72, 
                     time_steps=100, time_span=86400, min_elevation=10):
    lats = np.linspace(-90, 90, lat_points)
    lons = np.linspace(-180, 180, lon_points)
    times = np.linspace(0, time_span, time_steps)
    
    coverage_matrix = np.zeros((lat_points, lon_points))
    
    for lat_idx, lat in enumerate(lats):
        for lon_idx, lon in enumerate(lons):
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            
            ground_point = R_EARTH * np.array([
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad)
            ])
            
            visible_count = 0
            for t in times:
                positions = get_constellation_positions(constellation, t)
                
                for sat in positions:
                    if check_visibility(sat['position'], ground_point, min_elevation):
                        visible_count += 1
                        break
            
            coverage_matrix[lat_idx, lon_idx] = visible_count / time_steps
    
    return lats, lons, coverage_matrix


def compute_coverage_by_latitude(coverage_matrix, lats):
    avg_coverage = np.mean(coverage_matrix, axis=1)
    return lats, avg_coverage


def compute_revisit_time(constellation, lat=0, lon=0, time_span=86400, 
                          time_step=10, min_elevation=10):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    ground_point = R_EARTH * np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])
    
    times = np.arange(0, time_span, time_step)
    visibility = []
    
    for t in times:
        positions = get_constellation_positions(constellation, t)
        is_visible = False
        for sat in positions:
            if check_visibility(sat['position'], ground_point, min_elevation):
                is_visible = True
                break
        visibility.append(is_visible)
    
    gaps = []
    in_gap = True
    gap_start = 0
    
    for i, vis in enumerate(visibility):
        if vis and in_gap:
            if i > gap_start:
                gaps.append((i - gap_start) * time_step)
            in_gap = False
        elif not vis and not in_gap:
            gap_start = i
            in_gap = True
    
    if len(gaps) > 0:
        return {
            'mean_revisit': np.mean(gaps),
            'max_revisit': np.max(gaps),
            'min_revisit': np.min(gaps),
            'all_gaps': gaps
        }
    else:
        return {
            'mean_revisit': 0,
            'max_revisit': 0,
            'min_revisit': 0,
            'all_gaps': []
        }


def compute_number_of_visible_satellites(constellation, lat=0, lon=0, 
                                          time_span=3600, time_step=10, 
                                          min_elevation=10):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    ground_point = R_EARTH * np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])
    
    times = np.arange(0, time_span, time_step)
    num_visible = []
    
    for t in times:
        positions = get_constellation_positions(constellation, t)
        count = 0
        for sat in positions:
            if check_visibility(sat['position'], ground_point, min_elevation):
                count += 1
        num_visible.append(count)
    
    return {
        'times': times,
        'num_visible': num_visible,
        'mean': np.mean(num_visible),
        'max': np.max(num_visible),
        'min': np.min(num_visible)
    }


def print_constellation_summary(constellation):
    num_sats = len(constellation)
    num_planes = max([sat['plane'] for sat in constellation]) + 1
    
    if num_sats > 0:
        elements = constellation[0]['elements']
        altitude = elements.a_km - R_EARTH
        inclination = elements.i
        
        print(f"Walker 星座配置:")
        print(f"  总卫星数: {num_sats}")
        print(f"  轨道面数: {num_planes}")
        print(f"  每面卫星数: {num_sats // num_planes}")
        print(f"  轨道高度: {altitude:.1f} km")
        print(f"  轨道倾角: {inclination:.1f}°")


def get_constellation_orbit_elements(constellation):
    return [sat['elements'] for sat in constellation]
