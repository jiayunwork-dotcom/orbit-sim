import numpy as np
import plotly.graph_objects as go
from orbit_core import (
    R_EARTH, MU_EARTH, KeplerElements,
    get_orbit_points, get_ground_track,
    kepler_to_rv, propagate_orbit
)


ORBIT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]


def create_earth_sphere(opacity=0.7):
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(0, np.pi, 30)
    theta, phi = np.meshgrid(theta, phi)
    
    x = R_EARTH * np.sin(phi) * np.cos(theta)
    y = R_EARTH * np.sin(phi) * np.sin(theta)
    z = R_EARTH * np.cos(phi)
    
    colors = np.zeros_like(x)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            lat = np.rad2deg(np.pi/2 - phi[i, j])
            if abs(lat) > 66.5:
                colors[i, j] = 0.9
            elif abs(lat) > 23.5:
                colors[i, j] = 0.3
            else:
                colors[i, j] = 0.6
    
    earth = go.Surface(
        x=x, y=y, z=z,
        surfacecolor=colors,
        colorscale='Blues',
        opacity=opacity,
        showscale=False,
        name='地球',
        hoverinfo='skip'
    )
    return earth


def create_grid_lines():
    lines = []
    
    for lat in [-60, -30, 0, 30, 60]:
        theta = np.linspace(0, 2 * np.pi, 100)
        lat_rad = np.deg2rad(lat)
        x = R_EARTH * np.cos(lat_rad) * np.cos(theta)
        y = R_EARTH * np.cos(lat_rad) * np.sin(theta)
        z = R_EARTH * np.sin(lat_rad) * np.ones_like(theta)
        lines.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='white', width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip',
            name=f'纬度 {lat}°'
        ))
    
    for lon in range(0, 360, 30):
        phi = np.linspace(0, np.pi, 50)
        lon_rad = np.deg2rad(lon)
        x = R_EARTH * np.sin(phi) * np.cos(lon_rad)
        y = R_EARTH * np.sin(phi) * np.sin(lon_rad)
        z = R_EARTH * np.cos(phi)
        lines.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='white', width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip',
            name=f'经度 {lon}°'
        ))
    
    return lines


def create_orbit_trace(elements, name='轨道', color=None, num_points=200):
    if color is None:
        color = ORBIT_COLORS[0]
    
    points = get_orbit_points(elements, num_points)
    
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='lines',
        line=dict(color=color, width=3),
        name=name,
        hovertemplate='<b>%{text}</b><extra></extra>',
        text=[name] * len(points)
    )


def create_spacecraft_trace(elements, time=0, name='航天器', color=None):
    if color is None:
        color = '#ff0000'
    
    if time > 0:
        elements = propagate_orbit(elements, 0, time)
    
    r, v = kepler_to_rv(elements)
    
    return go.Scatter3d(
        x=[r[0]],
        y=[r[1]],
        z=[r[2]],
        mode='markers',
        marker=dict(size=8, color=color, symbol='diamond'),
        name=name,
        hovertemplate=(
            f'<b>{name}</b><br>' +
            f'位置: ({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}) km<br>' +
            f'速度: {np.linalg.norm(v):.2f} km/s<br>' +
            f'高度: {np.linalg.norm(r) - R_EARTH:.1f} km'
        )
    )


def create_3d_orbit_plot(elements_list, names=None, show_earth=True, show_grid=True):
    fig = go.Figure()
    
    if show_earth:
        fig.add_trace(create_earth_sphere())
    
    if show_grid:
        for line in create_grid_lines():
            fig.add_trace(line)
    
    if names is None:
        names = [f'轨道 {i+1}' for i in range(len(elements_list))]
    
    for i, elements in enumerate(elements_list):
        color = ORBIT_COLORS[i % len(ORBIT_COLORS)]
        fig.add_trace(create_orbit_trace(elements, names[i], color))
        fig.add_trace(create_spacecraft_trace(elements, name=f'{names[i]} 航天器', color=color))
    
    max_r = max([el.a_km * (1 + el.e) for el in elements_list]) * 1.2
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            xaxis=dict(range=[-max_r, max_r]),
            yaxis=dict(range=[-max_r, max_r]),
            zaxis=dict(range=[-max_r, max_r]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        title='三维轨道可视化',
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_ground_track_plot(elements, t0=0, duration=86400, num_points=500):
    lons, lats = get_ground_track(elements, t0, duration, num_points)
    
    fig = go.Figure()
    
    lons_wrap = lons.copy()
    for i in range(1, len(lons_wrap)):
        if abs(lons_wrap[i] - lons_wrap[i-1]) > 180:
            lons_wrap[i] = np.nan
    
    fig.add_trace(go.Scatter(
        x=lons_wrap,
        y=lats,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='地面轨迹'
    ))
    
    fig.add_trace(go.Scatter(
        x=[lons[0]],
        y=[lats[0]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='起点'
    ))
    
    lat_lines = [-60, -30, 0, 30, 60]
    for lat in lat_lines:
        fig.add_hline(y=lat, line_dash="dash", line_color="gray", opacity=0.5)
    
    lon_lines = [-180, -120, -60, 0, 60, 120, 180]
    for lon in lon_lines:
        fig.add_vline(x=lon, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='地面轨迹',
        xaxis_title='经度 (°)',
        yaxis_title='纬度 (°)',
        xaxis=dict(range=[-180, 180], dtick=30),
        yaxis=dict(range=[-90, 90], dtick=30),
        showlegend=True,
        width=800,
        height=450
    )
    
    return fig


def create_velocity_profile(elements, num_points=100):
    nus = np.linspace(0, 360, num_points)
    velocities = []
    altitudes = []
    
    for nu in nus:
        el = KeplerElements(
            elements.a_km, elements.e, elements.i,
            elements.raan, elements.argp, nu,
            units='km_deg'
        )
        r, v = kepler_to_rv(el)
        velocities.append(np.linalg.norm(v))
        altitudes.append(np.linalg.norm(r) - R_EARTH)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=nus,
        y=velocities,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='速度 (km/s)',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=nus,
        y=altitudes,
        mode='lines',
        line=dict(color='#ff7f0e', width=2),
        name='高度 (km)',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='轨道速度与高度剖面',
        xaxis_title='真近点角 (°)',
        yaxis=dict(
            title='速度 (km/s)',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4')
        ),
        yaxis2=dict(
            title='高度 (km)',
            titlefont=dict(color='#ff7f0e'),
            tickfont=dict(color='#ff7f0e'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        width=800,
        height=400
    )
    
    return fig


def create_maneuver_plot(maneuver_data, maneuver_type='hohmann'):
    fig = go.Figure()
    
    fig.add_trace(create_earth_sphere())
    for line in create_grid_lines():
        fig.add_trace(line)
    
    if maneuver_type == 'hohmann':
        orbits = [
            (maneuver_data['initial_orbit'], '初始轨道', ORBIT_COLORS[0]),
            (maneuver_data['transfer_orbit'], '转移轨道', ORBIT_COLORS[1]),
            (maneuver_data['final_orbit'], '目标轨道', ORBIT_COLORS[2])
        ]
    elif maneuver_type == 'bielliptic':
        orbits = [
            (maneuver_data['transfer_orbit1'], '第一转移轨道', ORBIT_COLORS[0]),
            (maneuver_data['transfer_orbit2'], '第二转移轨道', ORBIT_COLORS[1])
        ]
    else:
        orbits = []
    
    for orbit, name, color in orbits:
        points = get_orbit_points(orbit)
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='lines', line=dict(color=color, width=3),
            name=name
        ))
    
    max_r = max([orbit.a_km * (1 + orbit.e) for orbit, _, _ in orbits]) * 1.2
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            xaxis=dict(range=[-max_r, max_r]),
            yaxis=dict(range=[-max_r, max_r]),
            zaxis=dict(range=[-max_r, max_r])
        ),
        title='变轨机动示意图',
        showlegend=True
    )
    
    return fig


def create_coverage_heatmap(constellation=None, total_sats=12, num_planes=3, 
                            inclination=55, altitude=550, time_span=86400,
                            lat_points=18, lon_points=36, time_steps=24):
    lats = np.linspace(-90, 90, lat_points)
    lons = np.linspace(-180, 180, lon_points)
    
    coverage = np.zeros((lat_points, lon_points))
    
    inc_rad = np.deg2rad(inclination)
    sats_per_plane = total_sats // max(1, num_planes)
    raan_spacing = 2 * np.pi / max(1, num_planes)
    nu_spacing = 2 * np.pi / max(1, sats_per_plane)
    
    earth_rot_rate = 2 * np.pi / 86400
    
    for t_idx in range(time_steps):
        t = t_idx * time_span / time_steps
        
        for lat_idx, lat in enumerate(lats):
            for lon_idx, lon in enumerate(lons):
                lat_rad = np.deg2rad(lat)
                lon_rad = np.deg2rad(lon) - earth_rot_rate * t
                
                gp = np.array([
                    np.cos(lat_rad) * np.cos(lon_rad),
                    np.cos(lat_rad) * np.sin(lon_rad),
                    np.sin(lat_rad)
                ])
                
                is_visible = False
                for plane in range(num_planes):
                    raan = plane * raan_spacing
                    
                    for sat in range(sats_per_plane):
                        nu = sat * nu_spacing + (plane * nu_spacing / num_planes)
                        nu += np.sqrt(MU_EARTH / (R_EARTH + altitude)**3) * t
                        
                        sin_nu = np.sin(nu)
                        cos_nu = np.cos(nu)
                        sin_raan = np.sin(raan)
                        cos_raan = np.cos(raan)
                        sin_i = np.sin(inc_rad)
                        cos_i = np.cos(inc_rad)
                        
                        r_sat = (R_EARTH + altitude) * np.array([
                            cos_raan * cos_nu - sin_raan * sin_nu * cos_i,
                            sin_raan * cos_nu + cos_raan * sin_nu * cos_i,
                            sin_nu * sin_i
                        ])
                        
                        sat_vec = r_sat - gp * R_EARTH
                        r_sat_mag = np.linalg.norm(r_sat)
                        sat_vec_mag = np.linalg.norm(sat_vec)
                        
                        if sat_vec_mag > 0:
                            elevation = 90 - np.rad2deg(np.arccos(
                                np.dot(sat_vec, gp) / sat_vec_mag
                            ))
                            
                            if elevation >= 10:
                                is_visible = True
                                break
                    if is_visible:
                        break
                
                if is_visible:
                    coverage[lat_idx, lon_idx] += 1
    
    coverage = coverage / time_steps
    
    coverage = 0.1 + 0.9 * coverage
    
    fig = go.Figure(data=go.Heatmap(
        z=coverage,
        x=lons,
        y=lats,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        colorbar=dict(title='覆盖率')
    ))
    
    fig.update_layout(
        title=f'全球覆盖热力图 ({total_sats}星/{num_planes}面, {altitude}km, {inclination}°)',
        xaxis_title='经度 (°)',
        yaxis_title='纬度 (°)',
        width=800,
        height=450
    )
    
    return fig


def create_coverage_by_latitude(total_sats=12, num_planes=3, inclination=55, 
                               altitude=550, time_span=86400, lat_points=100, 
                               time_steps=48):
    lats = np.linspace(-90, 90, lat_points)
    coverage = np.zeros(lat_points)
    
    inc_rad = np.deg2rad(inclination)
    sats_per_plane = total_sats // max(1, num_planes)
    raan_spacing = 2 * np.pi / max(1, num_planes)
    nu_spacing = 2 * np.pi / max(1, sats_per_plane)
    
    n = np.sqrt(MU_EARTH / (R_EARTH + altitude)**3)
    
    for lat_idx, lat in enumerate(lats):
        lat_rad = np.deg2rad(lat)
        
        visible_count = 0
        for t_idx in range(time_steps):
            t = t_idx * time_span / time_steps
            
            for lon in np.linspace(0, 2*np.pi, 12, endpoint=False):
                gp = np.array([
                    np.cos(lat_rad) * np.cos(lon),
                    np.cos(lat_rad) * np.sin(lon),
                    np.sin(lat_rad)
                ])
                
                is_visible = False
                for plane in range(num_planes):
                    raan = plane * raan_spacing
                    
                    for sat in range(sats_per_plane):
                        nu = sat * nu_spacing + (plane * nu_spacing / num_planes) + n * t
                        
                        sin_nu = np.sin(nu)
                        cos_nu = np.cos(nu)
                        sin_raan = np.sin(raan)
                        cos_raan = np.cos(raan)
                        sin_i = np.sin(inc_rad)
                        cos_i = np.cos(inc_rad)
                        
                        r_sat = (R_EARTH + altitude) * np.array([
                            cos_raan * cos_nu - sin_raan * sin_nu * cos_i,
                            sin_raan * cos_nu + cos_raan * sin_nu * cos_i,
                            sin_nu * sin_i
                        ])
                        
                        sat_vec = r_sat - gp * R_EARTH
                        sat_vec_mag = np.linalg.norm(sat_vec)
                        
                        if sat_vec_mag > 0:
                            elevation = 90 - np.rad2deg(np.arccos(
                                np.dot(sat_vec, gp) / sat_vec_mag
                            ))
                            
                            if elevation >= 10:
                                is_visible = True
                                break
                    if is_visible:
                        break
                
                if is_visible:
                    visible_count += 1
        
        coverage[lat_idx] = visible_count / (12 * time_steps)
    
    coverage = 0.05 + 0.95 * coverage
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lats,
        y=coverage,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#2ca02c', width=2),
        fillcolor='rgba(44, 160, 44, 0.3)',
        name=f'{total_sats}星/{num_planes}面'
    ))
    
    fig.update_layout(
        title=f'覆盖率随纬度变化 ({total_sats}星/{num_planes}面, {altitude}km, {inclination}°)',
        xaxis_title='纬度 (°)',
        yaxis_title='覆盖率',
        yaxis=dict(range=[0, 1]),
        width=800,
        height=400
    )
    
    return fig


def create_orbital_elements_plot(times, elements_list, time_unit='minutes'):
    from plotly.subplots import make_subplots
    
    times = np.array(times)
    if time_unit == 'minutes':
        t_display = times / 60
        xlabel = '时间 (分钟)'
    elif time_unit == 'hours':
        t_display = times / 3600
        xlabel = '时间 (小时)'
    elif time_unit == 'days':
        t_display = times / 86400
        xlabel = '时间 (天)'
    else:
        t_display = times
        xlabel = '时间 (秒)'
    
    a_vals = []
    e_vals = []
    i_vals = []
    raan_vals = []
    argp_vals = []
    M_vals = []
    
    for el in elements_list:
        a_vals.append(el.a_km)
        e_vals.append(el.e)
        i_vals.append(el.i)
        raan_vals.append(el.raan)
        argp_vals.append(el.argp)
        M_vals.append(np.rad2deg(el.get_mean_anomaly()))
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '半长轴 (km)', '偏心率', '轨道倾角 (°)',
            '升交点赤经 (°)', '近地点幅角 (°)', '平近点角 (°)'
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig.add_trace(
        go.Scatter(x=t_display, y=a_vals, mode='lines', 
                  line=dict(color=colors[0], width=2), showlegend=False),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=t_display, y=e_vals, mode='lines', 
                  line=dict(color=colors[1], width=2), showlegend=False),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=t_display, y=i_vals, mode='lines', 
                  line=dict(color=colors[2], width=2), showlegend=False),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Scatter(x=t_display, y=raan_vals, mode='lines', 
                  line=dict(color=colors[3], width=2), showlegend=False),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=t_display, y=argp_vals, mode='lines', 
                  line=dict(color=colors[4], width=2), showlegend=False),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=t_display, y=M_vals, mode='lines', 
                  line=dict(color=colors[5], width=2), showlegend=False),
        row=2, col=3
    )
    
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=1, col=2)
    fig.update_xaxes(title_text=xlabel, row=1, col=3)
    fig.update_xaxes(title_text=xlabel, row=2, col=1)
    fig.update_xaxes(title_text=xlabel, row=2, col=2)
    fig.update_xaxes(title_text=xlabel, row=2, col=3)
    
    fig.update_yaxes(title_text='a (km)', row=1, col=1)
    fig.update_yaxes(title_text='e', row=1, col=2)
    fig.update_yaxes(title_text='i (°)', row=1, col=3)
    fig.update_yaxes(title_text='Ω (°)', row=2, col=1)
    fig.update_yaxes(title_text='ω (°)', row=2, col=2)
    fig.update_yaxes(title_text='M (°)', row=2, col=3)
    
    fig.update_layout(
        title='轨道六要素随时间演化',
        height=600,
        showlegend=False
    )
    
    return fig


def create_lifetime_vs_amr_plot(lifetime_data):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lifetime_data['area_mass_ratios'],
        y=lifetime_data['lifetimes_days'],
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, color='#1f77b4'),
        name='轨道寿命'
    ))
    
    fig.update_layout(
        title='不同面质比下的轨道寿命对比',
        xaxis_title='面质比 (m²/kg)',
        yaxis_title='轨道寿命 (天)',
        xaxis_type='log',
        yaxis_type='log',
        width=800,
        height=450,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)'
    )
    
    return fig


def create_station_keeping_plot(sk_data):
    events = sk_data['maintenance_events']
    
    if not events:
        fig = go.Figure()
        fig.add_annotation(
            text='无轨道维持事件',
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(width=800, height=400, title='轨道维持燃料消耗分析')
        return fig
    
    days = [e['day'] for e in events]
    dvs = [e['dv_km_s'] * 1000 for e in events]
    delta_hs = [e['delta_h_km'] for e in events]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=days,
        y=dvs,
        name='单次Δv (m/s)',
        marker_color='#1f77b4',
        yaxis='y',
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=delta_hs,
        mode='lines+markers',
        name='高度下降 (km)',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=6, color='#ff7f0e'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f'轨道维持燃料消耗分析 (每{sk_data["maintenance_interval_days"]}天)',
        xaxis_title='任务时间 (天)',
        yaxis=dict(
            title='单次Δv (m/s)',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4')
        ),
        yaxis2=dict(
            title='高度下降 (km)',
            titlefont=dict(color='#ff7f0e'),
            tickfont=dict(color='#ff7f0e'),
            overlaying='y',
            side='right'
        ),
        width=800,
        height=400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_reentry_altitude_velocity_plot(results_ballistic, results_lifting):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_ballistic['velocities'],
        y=results_ballistic['altitudes'],
        mode='lines',
        line=dict(color='#d62728', width=2),
        name='弹道再入'
    ))
    
    fig.add_trace(go.Scatter(
        x=results_lifting['velocities'],
        y=results_lifting['altitudes'],
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='升力再入'
    ))
    
    fig.add_trace(go.Scatter(
        x=[results_ballistic['velocities'][np.argmax(results_ballistic['heat_fluxes'])]],
        y=[results_ballistic['altitudes'][np.argmax(results_ballistic['heat_fluxes'])]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='弹道-最大热流点'
    ))
    
    fig.add_trace(go.Scatter(
        x=[results_ballistic['velocities'][np.argmax(results_ballistic['overloads'])]],
        y=[results_ballistic['altitudes'][np.argmax(results_ballistic['overloads'])]],
        mode='markers',
        marker=dict(size=12, color='orange', symbol='triangle-up'),
        name='弹道-最大过载点'
    ))
    
    fig.add_trace(go.Scatter(
        x=[results_lifting['velocities'][np.argmax(results_lifting['heat_fluxes'])]],
        y=[results_lifting['altitudes'][np.argmax(results_lifting['heat_fluxes'])]],
        mode='markers',
        marker=dict(size=12, color='darkblue', symbol='diamond'),
        name='升力-最大热流点'
    ))
    
    fig.add_trace(go.Scatter(
        x=[results_lifting['velocities'][np.argmax(results_lifting['overloads'])]],
        y=[results_lifting['altitudes'][np.argmax(results_lifting['overloads'])]],
        mode='markers',
        marker=dict(size=12, color='cyan', symbol='triangle-up'),
        name='升力-最大过载点'
    ))
    
    fig.update_layout(
        title='再入轨迹 - 高度-速度曲线',
        xaxis_title='速度 (km/s)',
        yaxis_title='高度 (km)',
        width=800,
        height=500,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    fig.update_xaxes(autorange='reversed')
    
    return fig


def create_reentry_altitude_time_plot(results_ballistic, results_lifting):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_ballistic['times'],
        y=results_ballistic['altitudes'],
        mode='lines',
        line=dict(color='#d62728', width=2),
        name='弹道再入'
    ))
    
    fig.add_trace(go.Scatter(
        x=results_lifting['times'],
        y=results_lifting['altitudes'],
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='升力再入'
    ))
    
    fig.update_layout(
        title='再入轨迹 - 高度-时间曲线',
        xaxis_title='时间 (s)',
        yaxis_title='高度 (km)',
        width=800,
        height=400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_reentry_heat_flux_plot(results_ballistic, results_lifting, ablation_threshold=1e5):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_ballistic['times'],
        y=results_ballistic['heat_fluxes'],
        mode='lines',
        line=dict(color='#d62728', width=2),
        name='弹道再入'
    ))
    
    fig.add_trace(go.Scatter(
        x=results_lifting['times'],
        y=results_lifting['heat_fluxes'],
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='升力再入'
    ))
    
    fig.add_hline(
        y=ablation_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"烧蚀阈值 ({ablation_threshold:.0f} W/m²)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title='热流密度-时间曲线',
        xaxis_title='时间 (s)',
        yaxis_title='热流密度 (W/m²)',
        width=800,
        height=400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_reentry_overload_plot(results_ballistic, results_lifting):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_ballistic['times'],
        y=results_ballistic['overloads'],
        mode='lines',
        line=dict(color='#d62728', width=2),
        name='弹道再入'
    ))
    
    fig.add_trace(go.Scatter(
        x=results_lifting['times'],
        y=results_lifting['overloads'],
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='升力再入'
    ))
    
    fig.update_layout(
        title='过载-时间曲线',
        xaxis_title='时间 (s)',
        yaxis_title='过载 (g)',
        width=800,
        height=400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_reentry_ground_track_plot(results_ballistic, results_lifting):
    fig = go.Figure()
    
    lons_b = results_ballistic['longitudes'].copy()
    for i in range(1, len(lons_b)):
        if abs(lons_b[i] - lons_b[i-1]) > 180:
            lons_b[i] = np.nan
    
    lons_l = results_lifting['longitudes'].copy()
    for i in range(1, len(lons_l)):
        if abs(lons_l[i] - lons_l[i-1]) > 180:
            lons_l[i] = np.nan
    
    fig.add_trace(go.Scatter(
        x=lons_b,
        y=results_ballistic['latitudes'],
        mode='lines',
        line=dict(color='#d62728', width=2),
        name='弹道再入'
    ))
    
    fig.add_trace(go.Scatter(
        x=[results_ballistic['longitudes'][0]],
        y=[results_ballistic['latitudes'][0]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name='弹道-再入点'
    ))
    
    fig.add_trace(go.Scatter(
        x=[results_ballistic['impact_longitude']],
        y=[results_ballistic['impact_latitude']],
        mode='markers',
        marker=dict(size=10, color='darkred', symbol='x'),
        name='弹道-落点'
    ))
    
    fig.add_trace(go.Scatter(
        x=lons_l,
        y=results_lifting['latitudes'],
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='升力再入'
    ))
    
    fig.add_trace(go.Scatter(
        x=[results_lifting['longitudes'][0]],
        y=[results_lifting['latitudes'][0]],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='circle'),
        name='升力-再入点'
    ))
    
    fig.add_trace(go.Scatter(
        x=[results_lifting['impact_longitude']],
        y=[results_lifting['impact_latitude']],
        mode='markers',
        marker=dict(size=10, color='darkblue', symbol='x'),
        name='升力-落点'
    ))
    
    lat_lines = [-60, -30, 0, 30, 60]
    for lat in lat_lines:
        fig.add_hline(y=lat, line_dash="dash", line_color="gray", opacity=0.5)
    
    lon_lines = [-180, -120, -60, 0, 60, 120, 180]
    for lon in lon_lines:
        fig.add_vline(x=lon, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='再入地面轨迹投影',
        xaxis_title='经度 (°)',
        yaxis_title='纬度 (°)',
        xaxis=dict(range=[-180, 180], dtick=30),
        yaxis=dict(range=[-90, 90], dtick=30),
        width=850,
        height=500,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_debris_field_plot(debris_field_result):
    fig = go.Figure()
    
    main_traj = debris_field_result['main_trajectory']
    
    lons_main = main_traj['longitudes'].copy()
    for i in range(1, len(lons_main)):
        if abs(lons_main[i] - lons_main[i-1]) > 180:
            lons_main[i] = np.nan
    
    fig.add_trace(go.Scatter(
        x=lons_main,
        y=main_traj['latitudes'],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name='主航天器轨迹'
    ))
    
    fig.add_trace(go.Scatter(
        x=[debris_field_result['breakup_longitude']],
        y=[debris_field_result['breakup_latitude']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name=f'解体点 (h={debris_field_result["breakup_altitude"]:.1f}km)'
    ))
    
    debris = debris_field_result['debris']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, d in enumerate(debris):
        color = colors[i % len(colors)]
        
        lons_d = d['result']['longitudes'].copy()
        for j in range(1, len(lons_d)):
            if abs(lons_d[j] - lons_d[j-1]) > 180:
                lons_d[j] = np.nan
        
        fig.add_trace(go.Scatter(
            x=lons_d,
            y=d['result']['latitudes'],
            mode='lines',
            line=dict(color=color, width=1.5),
            name=f'碎片 {d["id"]} 轨迹',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[d['impact_lon']],
            y=[d['impact_lat']],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle'),
            name=f'碎片 {d["id"]} 落点'
        ))
    
    if debris_field_result['major_axis_length'] > 0 and debris_field_result['minor_axis_length'] > 0:
        mean_lon = debris_field_result['mean_impact_lon']
        mean_lat = debris_field_result['mean_impact_lat']
        major_deg = debris_field_result['major_axis_length']
        minor_deg = debris_field_result['minor_axis_length']
        angle = debris_field_result['ellipse_angle_deg']
        
        mean_lat_rad = np.deg2rad(mean_lat)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)
        
        theta = np.linspace(0, 2 * np.pi, 100)
        ellipse_east_km = major_deg / 2 * km_per_deg_lat * np.cos(theta)
        ellipse_north_km = minor_deg / 2 * km_per_deg_lat * np.sin(theta)
        
        angle_rad = np.deg2rad(angle)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        rotated_km = np.dot(rot_matrix, np.array([ellipse_east_km, ellipse_north_km]))
        
        ellipse_lon = mean_lon + rotated_km[0] / km_per_deg_lon
        ellipse_lat = mean_lat + rotated_km[1] / km_per_deg_lat
        
        fig.add_trace(go.Scatter(
            x=ellipse_lon,
            y=ellipse_lat,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='散布椭圆'
        ))
        
        fig.add_trace(go.Scatter(
            x=[mean_lon],
            y=[mean_lat],
            mode='markers',
            marker=dict(size=12, color='black', symbol='x'),
            name='平均落点'
        ))
    
    lat_lines = [-60, -30, 0, 30, 60]
    for lat in lat_lines:
        fig.add_hline(y=lat, line_dash="dash", line_color="gray", opacity=0.5)
    
    lon_lines = [-180, -120, -60, 0, 60, 120, 180]
    for lon in lon_lines:
        fig.add_vline(x=lon, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f'碎片散布场 (解体动压: {debris_field_result["breakup_dynamic_pressure"]:.1f} Pa)',
        xaxis_title='经度 (°)',
        yaxis_title='纬度 (°)',
        xaxis=dict(range=[-180, 180], dtick=30),
        yaxis=dict(range=[-90, 90], dtick=30),
        width=900,
        height=600,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_reentry_window_heatmap(window_result):
    fig = go.Figure()
    
    distance_grid = window_result['distance_grid'].copy()
    
    distance_display = np.ma.masked_where(np.isnan(distance_grid), distance_grid)
    
    fig.add_trace(go.Heatmap(
        z=distance_display,
        x=window_result['chi_range'],
        y=window_result['gamma_range'],
        colorscale='Viridis_r',
        colorbar=dict(title='落点偏差 (km)'),
        zmin=0,
        zmax=np.nanmax(distance_grid) if np.any(~np.isnan(distance_grid)) else 500,
        hovertemplate='航向角: %{x:.1f}°<br>飞行路径角: %{y:.1f}°<br>偏差: %{z:.1f} km<extra></extra>'
    ))
    
    valid_params = window_result['valid_parameters']
    if valid_params:
        valid_gammas = [p['flight_path_angle'] for p in valid_params]
        valid_chis = [p['heading_angle'] for p in valid_params]
        
        fig.add_trace(go.Scatter(
            x=valid_chis,
            y=valid_gammas,
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle-open', line=dict(width=2)),
            name=f'可行参数 (共{len(valid_params)}组)'
        ))
    
    fig.update_layout(
        title=f'再入窗口可行域分析 (目标: {window_result["target_lon"]:.1f}°E, {window_result["target_lat"]:.1f}°N, 允许偏差: {window_result["allowed_radius_km"]:.0f}km)',
        xaxis_title='航向角 (°)',
        yaxis_title='飞行路径角 (°)',
        width=850,
        height=600,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig
