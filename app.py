import streamlit as st
import numpy as np
import pandas as pd

from orbit_core import (
    R_EARTH, MU_EARTH, KeplerElements,
    kepler_to_rv, rv_to_kepler,
    propagate_orbit, vis_viva
)
from orbit_propagation import (
    Perturbations, propagate_numerical,
    j2_precession_rates, sun_synchronous_inclination,
    perigee_event, apogee_event, altitude_event,
    estimate_orbit_lifetime, orbit_station_keeping_analysis,
    lifetime_vs_area_mass_ratio
)
from maneuvers import (
    hohmann_transfer, bielliptic_transfer,
    inclination_change, phasing_maneuver,
    lambert_transfer, multi_turn_lambert
)
from visualization import (
    create_3d_orbit_plot, create_ground_track_plot,
    create_velocity_profile, create_maneuver_plot,
    create_coverage_heatmap, create_coverage_by_latitude,
    create_orbital_elements_plot, create_lifetime_vs_amr_plot,
    create_station_keeping_plot,
    create_reentry_altitude_velocity_plot,
    create_reentry_altitude_time_plot,
    create_reentry_heat_flux_plot,
    create_reentry_overload_plot,
    create_reentry_ground_track_plot,
    create_debris_field_plot,
    create_reentry_window_heatmap,
    create_b_plane_plot, create_approach_distance_plot,
    create_collision_cumulative_plot, create_distance_distribution_plot,
    create_batch_screening_table
)
from collision_risk import (
    collision_risk_analysis, find_min_dv_maneuver,
    batch_screening, generate_collision_statistics,
    get_position_covariance
)
from reentry import (
    ReentryVehicle, ReentryInitialConditions,
    simulate_both_modes, standard_atmosphere_1976,
    simulate_debris_field, analyze_reentry_window
)
from constellation import (
    walker_delta_constellation, get_constellation_orbit_elements,
    get_constellation_positions, compute_coverage_by_latitude
)
from optimization import (
    optimize_maneuver, station_keeping_delta_v,
    minimize_fuel_usage
)

st.set_page_config(
    page_title="航天器轨道力学分析与变轨仿真工具",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ 航天器轨道力学分析与变轨仿真工具")
st.markdown("---")

with st.sidebar:
    st.header("导航")
    page = st.radio(
        "选择功能模块",
        [
            "📊 轨道要素分析",
            "🌍 轨道可视化",
            "🛫 变轨机动设计",
            "📡 轨道摄动分析",
            "✨ 星座设计与覆盖",
            "🔢 数值积分传播",
            "⚡ 机动优化",
            "🔥 再入轨迹分析",
            "☄️ 碰撞风险评估"
        ]
    )
    st.markdown("---")
    st.caption("基于 Python + NumPy + SciPy + Plotly + Streamlit")

if page == "📊 轨道要素分析":
    st.header("轨道要素分析")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("开普勒六要素输入")
        
        unit_system = st.selectbox(
            "单位制",
            ["公里 + 角度 (km + deg)", "地球半径 + 弧度 (Re + rad)"]
        )
        
        a_input = st.number_input("半长轴 (a)", value=7000.0, min_value=6400.0, max_value=50000.0, step=100.0)
        e_input = st.number_input("偏心率 (e)", value=0.0, min_value=0.0, max_value=0.9, step=0.01)
        i_input = st.number_input("轨道倾角 (i)", value=45.0, min_value=0.0, max_value=180.0, step=1.0)
        raan_input = st.number_input("升交点赤经 (RAAN)", value=0.0, min_value=0.0, max_value=360.0, step=1.0)
        argp_input = st.number_input("近地点幅角 (ω)", value=0.0, min_value=0.0, max_value=360.0, step=1.0)
        nu_input = st.number_input("真近点角 (ν)", value=0.0, min_value=0.0, max_value=360.0, step=1.0)
        
        elements = KeplerElements(
            a_input, e_input, i_input,
            raan_input, argp_input, nu_input,
            units='km_deg'
        )
    
    with col2:
        st.subheader("轨道参数计算结果")
        
        r, v = kepler_to_rv(elements)
        
        st.metric("轨道周期", f"{elements.get_period()/60:.2f} 分钟")
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("近地点高度", f"{elements.get_hp():.2f} km")
            st.metric("近地点速度", f"{vis_viva(elements.a_km, elements.get_rp()):.3f} km/s")
        with col2b:
            st.metric("远地点高度", f"{elements.get_ha():.2f} km")
            st.metric("远地点速度", f"{vis_viva(elements.a_km, elements.get_ra()):.3f} km/s")
        
        st.metric("比轨道能量", f"{elements.get_specific_energy():.2f} km²/s²")
        st.metric("角动量", f"{elements.get_angular_momentum():.2f} km²/s")
        
        st.markdown("#### 位置速度向量")
        st.write(f"位置: ({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}) km")
        st.write(f"速度: ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}) km/s")
    
    st.markdown("---")
    st.subheader("位置速度 ↔ 开普勒要素转换")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("输入位置速度向量:")
        rv_x = st.number_input("X (km)", value=r[0], format="%.2f")
        rv_y = st.number_input("Y (km)", value=r[1], format="%.2f")
        rv_z = st.number_input("Z (km)", value=r[2], format="%.2f")
        vx = st.number_input("Vx (km/s)", value=v[0], format="%.4f")
        vy = st.number_input("Vy (km/s)", value=v[1], format="%.4f")
        vz = st.number_input("Vz (km/s)", value=v[2], format="%.4f")
        
        if st.button("转换为开普勒要素"):
            converted = rv_to_kepler(
                [rv_x, rv_y, rv_z],
                [vx, vy, vz]
            )
            st.success("转换成功!")
            st.write(converted)
    
    with col4:
        st.write("速度-高度关系 (Vis-Viva方程):")
        r_test = st.slider("到地心距离 (km)", int(R_EARTH+100), 50000, int(elements.a_km))
        v_test = vis_viva(elements.a_km, r_test)
        st.metric(f"r = {r_test} km 处的速度", f"{v_test:.3f} km/s")
        
        st.markdown("#### 开普勒方程验证")
        M = st.slider("平近点角 (deg)", 0, 360, 0)
        from orbit_core import kepler_equation_solver, mean_to_true
        E = kepler_equation_solver(np.deg2rad(M), elements.e)
        nu_calc = mean_to_true(np.deg2rad(M), elements.e)
        st.write(f"偏近点角 E = {np.rad2deg(E):.2f}°")
        st.write(f"真近点角 ν = {np.rad2deg(nu_calc):.2f}°")

elif page == "🌍 轨道可视化":
    st.header("轨道可视化")
    
    tab1, tab2, tab3 = st.tabs(["三维轨道", "地面轨迹", "速度剖面"])
    
    with tab1:
        st.subheader("三维地心惯性坐标系轨道")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("轨道参数:")
            a1 = st.number_input("半长轴 1", value=7000.0, min_value=6400.0, step=100.0, key="a1")
            e1 = st.number_input("偏心率 1", value=0.1, min_value=0.0, max_value=0.9, step=0.01, key="e1")
            i1 = st.number_input("倾角 1", value=45.0, min_value=0.0, max_value=180.0, step=5.0, key="i1")
            
            add_orbit2 = st.checkbox("添加第二条轨道")
            
            if add_orbit2:
                a2 = st.number_input("半长轴 2", value=10000.0, min_value=6400.0, step=100.0, key="a2")
                e2 = st.number_input("偏心率 2", value=0.2, min_value=0.0, max_value=0.9, step=0.01, key="e2")
                i2 = st.number_input("倾角 2", value=60.0, min_value=0.0, max_value=180.0, step=5.0, key="i2")
        
        with col2:
            elements_list = [
                KeplerElements(a1, e1, i1, 0, 0, 0, units='km_deg')
            ]
            names = ["轨道 1"]
            
            if add_orbit2:
                elements_list.append(
                    KeplerElements(a2, e2, i2, 30, 0, 0, units='km_deg')
                )
                names.append("轨道 2")
            
            fig = create_3d_orbit_plot(elements_list, names)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("地面轨迹投影")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            a_gt = st.number_input("半长轴", value=6700.0, min_value=6400.0, step=100.0, key="gt_a")
            e_gt = st.number_input("偏心率", value=0.0, min_value=0.0, max_value=0.9, step=0.01, key="gt_e")
            i_gt = st.number_input("倾角", value=55.0, min_value=0.0, max_value=180.0, step=5.0, key="gt_i")
            duration_hours = st.slider("仿真时长 (小时)", 1, 48, 24)
            
            ss_inc = sun_synchronous_inclination(a_gt - R_EARTH)
            st.info(f"太阳同步轨道倾角约为: {ss_inc:.2f}°")
        
        with col2:
            elements_gt = KeplerElements(a_gt, e_gt, i_gt, 0, 0, 0, units='km_deg')
            fig = create_ground_track_plot(
                elements_gt,
                duration=duration_hours * 3600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("轨道速度与高度剖面")
        
        a_vp = st.slider("半长轴", 6400.0, 40000.0, 7000.0, step=100.0)
        e_vp = st.slider("偏心率", 0.0, 0.7, 0.2, step=0.01)
        
        elements_vp = KeplerElements(a_vp, e_vp, 0, 0, 0, 0, units='km_deg')
        fig = create_velocity_profile(elements_vp)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **说明:**
        - 速度在近地点最大，远地点最小
        - 高度与速度呈反比关系（能量守恒）
        - 圆轨道的速度和高度保持恒定
        """)

elif page == "🛫 变轨机动设计":
    st.header("变轨机动设计")
    
    maneuver_type = st.selectbox(
        "选择机动类型",
        ["霍曼转移", "双椭圆转移", "倾角改变", "调相机动", "Lambert 转移"]
    )
    
    if maneuver_type == "霍曼转移":
        st.subheader("霍曼转移 (Hohmann Transfer)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            r1 = st.number_input("初始轨道半径 (km)", value=6700.0, min_value=6400.0, step=100.0)
            r2 = st.number_input("目标轨道半径 (km)", value=42164.0, min_value=6400.0, step=100.0)
            i_man = st.number_input("轨道倾角 (deg)", value=0.0, min_value=0.0, max_value=180.0, step=5.0)
        
        with col2:
            result = hohmann_transfer(r1, r2, i_man)
            
            st.success(f"总 Δv: {result['total_dv']:.3f} km/s")
            st.write(f"第一次脉冲 (近地点): {result['dv1']:.3f} km/s")
            st.write(f"第二次脉冲 (远地点): {result['dv2']:.3f} km/s")
            st.write(f"转移时间: {result['transfer_time']/3600:.2f} 小时")
            
            if r2 > r1:
                ratio = r2 / r1
                if ratio > 11.9:
                    st.info("💡 提示: 高度比 > 11.9 时，双椭圆转移可能更省燃料")
        
        fig = create_maneuver_plot(result, 'hohmann')
        st.plotly_chart(fig, use_container_width=True)
    
    elif maneuver_type == "双椭圆转移":
        st.subheader("双椭圆转移 (Bi-Elliptic Transfer)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            r1_be = st.number_input("初始轨道半径 (km)", value=6700.0, min_value=6400.0, step=100.0, key="be_r1")
            r2_be = st.number_input("目标轨道半径 (km)", value=100000.0, min_value=6400.0, step=1000.0, key="be_r2")
            rb = st.number_input("中间轨道半径 (km)", value=200000.0, min_value=max(r1_be, r2_be)+1000, step=1000.0)
        
        with col2:
            result_be = bielliptic_transfer(r1_be, r2_be, rb)
            
            if result_be['is_better']:
                st.success(f"双椭圆更优! 节省 Δv: {result_be['dv_saving']:.3f} km/s")
            else:
                st.warning(f"霍曼转移更优。双椭圆多消耗: {-result_be['dv_saving']:.3f} km/s")
            
            st.write(f"总 Δv (双椭圆): {result_be['total_dv']:.3f} km/s")
            st.write(f"总 Δv (霍曼): {result_be['hohmann_dv']:.3f} km/s")
            st.write(f"转移时间: {result_be['transfer_time']/3600:.2f} 小时")
        
        fig = create_maneuver_plot(result_be, 'bielliptic')
        st.plotly_chart(fig, use_container_width=True)
    
    elif maneuver_type == "倾角改变":
        st.subheader("轨道倾角改变")
        
        r_inc = st.number_input("轨道半径 (km)", value=6700.0, min_value=6400.0, step=100.0)
        delta_i = st.slider("倾角改变量 (deg)", 0.0, 90.0, 10.0, step=1.0)
        
        result_inc = inclination_change(r_inc, delta_i)
        
        st.metric("所需 Δv", f"{result_inc['delta_v']:.3f} km/s")
        st.write(f"轨道速度: {result_inc['velocity']:.3f} km/s")
        st.info(result_inc['efficiency_note'])
        
        delta_is = np.linspace(0, 90, 100)
        dvs = [inclination_change(r_inc, di)['delta_v'] for di in delta_is]
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=delta_is, y=dvs, mode='lines', name='Δv'))
        fig.update_layout(
            title='倾角改变与 Δv 关系',
            xaxis_title='倾角改变量 (deg)',
            yaxis_title='Δv (km/s)',
            width=800,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif maneuver_type == "调相机动":
        st.subheader("平面内调相机动")
        
        r_ph = st.number_input("原始轨道半径 (km)", value=6700.0, min_value=6400.0, step=100.0)
        delta_phase = st.slider("相位改变量 (deg)", 1.0, 359.0, 45.0, step=1.0)
        direction = st.selectbox("方向", ["forward", "backward"])
        
        result_ph = phasing_maneuver(r_ph, delta_phase, direction)
        
        st.metric("总 Δv", f"{result_ph['total_dv']:.4f} km/s")
        st.write(f"调相轨道近地点高度: {result_ph['r_perigee'] - R_EARTH:.1f} km")
        st.write(f"转移圈数: {result_ph['n_loops']}")
        st.write(f"转移时间: {result_ph['transfer_time']/3600:.2f} 小时")
        
        st.info(f"方向: {'前进 (追上目标)' if direction == 'forward' else '后退 (等待目标)'}")
    
    elif maneuver_type == "Lambert 转移":
        st.subheader("Lambert 轨道转移")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            r1_lam = st.number_input("初始半径 r1 (km)", value=7000.0, min_value=6400.0, step=100.0)
            r2_lam = st.number_input("目标半径 r2 (km)", value=8000.0, min_value=6400.0, step=100.0)
            nu_diff = st.slider("真近点角差 (deg)", 10.0, 350.0, 90.0, step=10.0)
            tof_hours = st.slider("转移时间 (小时)", 0.1, 48.0, 1.5, step=0.1)
            
            a_transfer_est = (r1_lam + r2_lam) / 2
            t_hohmann_est = np.pi * np.sqrt(a_transfer_est**3 / MU_EARTH) / 3600
            st.info(f"💡 霍曼转移时间约为: {t_hohmann_est:.2f} 小时")
        
        with col2:
            r1_vec = np.array([r1_lam, 0, 0])
            nu_rad = np.deg2rad(nu_diff)
            r2_vec = np.array([r2_lam * np.cos(nu_rad), r2_lam * np.sin(nu_rad), 0])
            
            el1 = KeplerElements(r1_lam, 0, 0, 0, 0, 0, units='km_deg')
            el2 = KeplerElements(r2_lam, 0, 0, 0, 0, nu_diff, units='km_deg')
            
            try:
                result_lam = lambert_transfer(el1, el2, tof_hours * 3600)
                
                st.success(f"总 Δv: {result_lam['total_dv']:.3f} km/s")
                st.write(f"出发 Δv: {result_lam['dv1']:.3f} km/s")
                st.write(f"到达 Δv: {result_lam['dv2']:.3f} km/s")
                st.write(f"转移时间: {tof_hours:.1f} 小时")
                
                elements_list = [el1, result_lam['transfer_orbit'], el2]
                names = ["初始轨道", "转移轨道", "目标轨道"]
                fig = create_3d_orbit_plot(elements_list, names)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as ex:
                st.error(f"求解失败: {str(ex)}")
                st.info("尝试调整转移时间（建议在霍曼转移时间附近）或增大转移时间范围")

elif page == "📡 轨道摄动分析":
    st.header("轨道摄动分析")
    
    tab1, tab2, tab3 = st.tabs(["J2 摄动", "大气阻力", "太阳同步轨道"])
    
    with tab1:
        st.subheader("J2 摄动（地球扁率）")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            a_j2 = st.number_input("半长轴 (km)", value=7000.0, min_value=6400.0, step=100.0, key="j2_a")
            e_j2 = st.number_input("偏心率", value=0.1, min_value=0.0, max_value=0.9, step=0.01, key="j2_e")
            i_j2 = st.number_input("倾角 (deg)", value=45.0, min_value=0.0, max_value=180.0, step=5.0, key="j2_i")
        
        with col2:
            elements_j2 = KeplerElements(a_j2, e_j2, i_j2, 0, 0, 0, units='km_deg')
            rates = j2_precession_rates(elements_j2)
            
            st.metric("升交点赤经进动率", f"{rates['raan_deg_per_day']:.4f} °/天")
            st.metric("近地点幅角旋转率", f"{rates['argp_deg_per_day']:.4f} °/天")
            
            if rates['raan_deg_per_day'] < 0:
                st.write("→ 升交点西退（顺行轨道）")
            elif rates['raan_deg_per_day'] > 0:
                st.write("→ 升交点东进（逆行轨道）")
            else:
                st.write("→ 极轨道，无RAAN进动")
        
        st.markdown("#### 倾角对进动率的影响")
        inclinations = np.linspace(0, 180, 100)
        raan_rates = []
        argp_rates = []
        
        for i in inclinations:
            el = KeplerElements(a_j2, e_j2, i, 0, 0, 0, units='km_deg')
            r = j2_precession_rates(el)
            raan_rates.append(r['raan_deg_per_day'])
            argp_rates.append(r['argp_deg_per_day'])
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inclinations, y=raan_rates, mode='lines', name='RAAN进动率 (°/天)'))
        fig.add_trace(go.Scatter(x=inclinations, y=argp_rates, mode='lines', name='近地点旋转率 (°/天)'))
        fig.update_layout(
            title='J2摄动进动率与倾角的关系',
            xaxis_title='轨道倾角 (deg)',
            yaxis_title='变化率 (°/天)',
            width=800,
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **说明:**
        - 当 i = 90°（极轨道），RAAN 进动率为 0
        - 当 i ≈ 63.4°（临界倾角），近地点旋转率为 0
        - 太阳同步轨道利用 RAAN 进动率 = 0.9856°/天
        """)
    
    with tab2:
        st.subheader("大气阻力摄动")
        
        col_drag1, col_drag2 = st.columns([1, 1])
        
        with col_drag1:
            a_drag = st.number_input("初始半长轴 (km)", value=6600.0, min_value=6400.0, step=10.0, key="drag_a")
            e_drag = st.number_input("初始偏心率", value=0.05, min_value=0.0, max_value=0.5, step=0.01, key="drag_e")
            area_mass = st.slider("面质比 (m²/kg)", 0.001, 0.1, 0.01, step=0.001)
        
        with col_drag2:
            sim_days = st.slider("仿真天数", 1, 365, 30)
            reentry_alt = st.number_input("再入高度阈值 (km)", value=120.0, min_value=80.0, max_value=200.0, step=10.0)
        
        pert_drag = Perturbations(
            use_j2=False,
            use_drag=True,
            use_srp=False,
            area_mass_ratio=area_mass
        )
        
        elements_drag = KeplerElements(a_drag, e_drag, 0, 0, 0, 0, units='km_deg')
        
        with st.spinner("正在进行轨道衰减仿真..."):
            times_drag = np.linspace(0, sim_days * 86400, 100)
            altitudes = []
            eccentricities = []
            
            current_el = elements_drag
            for i, t in enumerate(times_drag):
                if i > 0:
                    dt = times_drag[i] - times_drag[i-1]
                    current_el = propagate_numerical(
                        current_el, 0, dt,
                        dt=60.0,
                        method='rk4',
                        perturbations=pert_drag
                    )
                altitudes.append(current_el.a_km * (1 - current_el.e) - R_EARTH)
                eccentricities.append(current_el.e)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times_drag / 86400,
                y=altitudes,
                mode='lines',
                name='近地点高度 (km)'
            ))
            fig.add_hline(
                y=reentry_alt,
                line_dash="dash",
                line_color="red",
                annotation_text=f"再入高度 ({reentry_alt}km)",
                annotation_position="right"
            )
            fig.update_layout(
                title='轨道高度衰减曲线',
                xaxis_title='时间 (天)',
                yaxis_title='近地点高度 (km)',
                width=800,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=times_drag / 86400,
                y=eccentricities,
                mode='lines',
                name='偏心率',
                line=dict(color='orange')
            ))
            fig2.update_layout(
                title='偏心率变化',
                xaxis_title='时间 (天)',
                yaxis_title='偏心率',
                width=800,
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            if altitudes[-1] < 100:
                st.warning("⚠️ 轨道已衰减到 100km 以下，航天器将再入大气层")
        
        st.markdown("---")
        st.subheader("🚀 轨道寿命预测")
        
        with st.expander("轨道寿命预测参数", expanded=True):
            col_life1, col_life2 = st.columns([1, 1])
            with col_life1:
                life_area_mass = st.number_input(
                    "面质比 (m²/kg)", 
                    value=0.01, 
                    min_value=0.001, 
                    max_value=0.5, 
                    step=0.001,
                    key="life_amr"
                )
                cd_life = st.number_input("阻力系数 Cd", value=2.2, min_value=1.0, max_value=4.0, step=0.1)
            with col_life2:
                life_reentry_alt = st.number_input(
                    "再入高度阈值 (km)", 
                    value=120.0, 
                    min_value=80.0, 
                    max_value=200.0, 
                    step=10.0,
                    key="life_reentry"
                )
        
        if st.button("计算轨道寿命", key="calc_lifetime"):
            with st.spinner("正在进行轨道寿命预测..."):
                lifetime_result = estimate_orbit_lifetime(
                    elements_drag,
                    area_mass_ratio=life_area_mass,
                    cd=cd_life,
                    reentry_altitude=life_reentry_alt
                )
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric(
                        "预估轨道寿命",
                        f"{lifetime_result['lifetime_days']:.0f} 天",
                        delta=f"{lifetime_result['lifetime_days']/365:.1f} 年"
                    )
                with col_res2:
                    st.metric(
                        "初始近地点高度",
                        f"{elements_drag.get_hp():.1f} km"
                    )
                with col_res3:
                    st.metric(
                        "最终近地点高度",
                        f"{lifetime_result['hp_history'][-1]:.1f} km"
                    )
                
                fig_life = go.Figure()
                fig_life.add_trace(go.Scatter(
                    x=lifetime_result['time_history'],
                    y=lifetime_result['hp_history'],
                    mode='lines',
                    name='近地点高度',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_life.add_hline(
                    y=life_reentry_alt,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"再入高度 ({life_reentry_alt}km)",
                    annotation_position="right"
                )
                fig_life.add_vline(
                    x=lifetime_result['lifetime_days'],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"寿命终点: {lifetime_result['lifetime_days']:.0f}天",
                    annotation_position="top"
                )
                fig_life.update_layout(
                    title='轨道寿命衰减曲线',
                    xaxis_title='时间 (天)',
                    yaxis_title='近地点高度 (km)',
                    width=800,
                    height=400
                )
                st.plotly_chart(fig_life, use_container_width=True)
        
        st.markdown("---")
        st.subheader("⚡ 轨道维持燃料消耗分析")
        
        with st.expander("轨道维持参数", expanded=True):
            col_sk1, col_sk2 = st.columns([1, 1])
            with col_sk1:
                sk_interval = st.number_input(
                    "轨道维持间隔 (天)", 
                    value=30, 
                    min_value=7, 
                    max_value=180, 
                    step=7,
                    key="sk_interval"
                )
                sk_area_mass = st.number_input(
                    "面质比 (m²/kg)", 
                    value=0.01, 
                    min_value=0.001, 
                    max_value=0.5, 
                    step=0.001,
                    key="sk_amr"
                )
            with col_sk2:
                sk_mission_years = st.number_input(
                    "任务周期 (年)", 
                    value=5.0, 
                    min_value=0.5, 
                    max_value=30.0, 
                    step=0.5,
                    key="sk_mission"
                )
        
        if st.button("计算轨道维持燃料消耗", key="calc_sk"):
            with st.spinner("正在计算轨道维持燃料消耗..."):
                sk_result = orbit_station_keeping_analysis(
                    elements_drag,
                    area_mass_ratio=sk_area_mass,
                    maintenance_interval_days=sk_interval,
                    mission_duration_years=sk_mission_years
                )
                
                col_sk_res1, col_sk_res2, col_sk_res3 = st.columns(3)
                with col_sk_res1:
                    st.metric(
                        "总Δv消耗",
                        f"{sk_result['total_dv_km_s'] * 1000:.1f} m/s",
                        delta=f"{sk_result['total_dv_km_s']:.4f} km/s"
                    )
                with col_sk_res2:
                    st.metric(
                        "维持机动次数",
                        f"{sk_result['total_maneuvers']} 次"
                    )
                with col_sk_res3:
                    avg_dv = (sk_result['total_dv_km_s'] * 1000 / max(1, sk_result['total_maneuvers'])) if sk_result['total_maneuvers'] > 0 else 0
                    st.metric(
                        "平均单次Δv",
                        f"{avg_dv:.2f} m/s"
                    )
                
                fig_sk = create_station_keeping_plot(sk_result)
                st.plotly_chart(fig_sk, use_container_width=True)
                
                if sk_result['maintenance_events']:
                    st.markdown("#### 轨道维持事件详情")
                    events_df = pd.DataFrame(sk_result['maintenance_events'])
                    events_df['dv_m_s'] = events_df['dv_km_s'] * 1000
                    events_df_display = events_df[['day', 'delta_h_km', 'dv_m_s', 'hp_before_km']].rename(columns={
                        'day': '任务天数',
                        'delta_h_km': '高度下降 (km)',
                        'dv_m_s': 'Δv (m/s)',
                        'hp_before_km': '维持前近地点高度 (km)'
                    })
                    st.dataframe(events_df_display, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 不同面质比下的寿命对比")
        
        if st.button("生成寿命对比图", key="gen_lifetime_comparison"):
            with st.spinner("正在计算不同面质比下的寿命..."):
                amr_values = np.logspace(-3, -1, 15)
                lifetime_comp = lifetime_vs_area_mass_ratio(
                    elements_drag,
                    area_mass_ratios=amr_values,
                    cd=cd_life if 'cd_life' in locals() else 2.2
                )
                
                fig_comp = create_lifetime_vs_amr_plot(lifetime_comp)
                st.plotly_chart(fig_comp, use_container_width=True)
                
                comp_df = pd.DataFrame({
                    '面质比 (m²/kg)': lifetime_comp['area_mass_ratios'],
                    '轨道寿命 (天)': lifetime_comp['lifetimes_days'],
                    '轨道寿命 (年)': lifetime_comp['lifetimes_days'] / 365
                })
                st.dataframe(comp_df, use_container_width=True)
    
    with tab3:
        st.subheader("太阳同步轨道设计")
        
        altitude_ss = st.slider("轨道高度 (km)", 200, 1200, 500, step=10)
        ss_inclination = sun_synchronous_inclination(altitude_ss)
        
        st.success(f"所需轨道倾角: {ss_inclination:.2f}°")
        st.info(
            f"该轨道的升交点赤经进动率约为 0.9856°/天，\n"
            f"与地球绕太阳公转的角速度相同，因此轨道面与太阳方向保持固定关系。"
        )
        
        altitudes = np.linspace(200, 1200, 100)
        inclinations = [sun_synchronous_inclination(h) for h in altitudes]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=altitudes,
            y=inclinations,
            mode='lines',
            fill='tozeroy',
            line=dict(color='goldenrod', width=2)
        ))
        fig.update_layout(
            title='太阳同步轨道 - 高度与倾角关系',
            xaxis_title='轨道高度 (km)',
            yaxis_title='轨道倾角 (deg)',
            width=800,
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "✨ 星座设计与覆盖":
    st.header("星座设计与覆盖分析")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Walker 星座参数")
        total_sats = st.number_input("总卫星数", value=12, min_value=3, max_value=100, step=1)
        num_planes = st.number_input("轨道面数", value=3, min_value=1, max_value=12, step=1)
        phasing = st.number_input("相位因子", value=1, min_value=0, max_value=10, step=1)
        altitude_const = st.slider("轨道高度 (km)", 200, 2000, 550, step=50)
        inclination_const = st.slider("轨道倾角 (deg)", 0, 180, 55, step=5)
    
    with col2:
        constellation = walker_delta_constellation(
            total_sats, num_planes, phasing,
            altitude_const, inclination_const
        )
        
        elements_list = get_constellation_orbit_elements(constellation)
        names = [f"轨道面 {s['plane']+1} - 星 {s['sat_index']+1}" for s in constellation]
        
        fig = create_3d_orbit_plot(elements_list[:min(12, len(elements_list))], 
                                   names[:min(12, len(names))])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("覆盖分析")
    
    with st.spinner("正在计算覆盖分析..."):
        fig_heatmap = create_coverage_heatmap(
            constellation,
            total_sats=total_sats,
            num_planes=num_planes,
            inclination=inclination_const,
            altitude=altitude_const,
            time_steps=12,
            lat_points=12,
            lon_points=24
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        fig_lat = create_coverage_by_latitude(
            total_sats=total_sats,
            num_planes=num_planes,
            inclination=inclination_const,
            altitude=altitude_const,
            time_steps=24,
            lat_points=60
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    
    with st.expander("星座参数详情"):
        st.write(f"**总卫星数:** {total_sats}")
        st.write(f"**轨道面数:** {num_planes}")
        st.write(f"**每面卫星数:** {total_sats // num_planes}")
        st.write(f"**轨道高度:** {altitude_const} km")
        st.write(f"**轨道倾角:** {inclination_const}°")
        st.write(f"**相位因子:** {phasing}")
        st.write(f"**Walker Delta 符号:** {total_sats}:{num_planes}:{phasing}")

elif page == "🔢 数值积分传播":
    st.header("数值积分轨道传播")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("积分参数设置")
        
        a_prop = st.number_input("半长轴 (km)", value=6700.0, min_value=6400.0, step=100.0)
        e_prop = st.number_input("偏心率", value=0.1, min_value=0.0, max_value=0.9, step=0.01)
        i_prop = st.number_input("倾角 (deg)", value=45.0, min_value=0.0, max_value=180.0, step=5.0)
        
        method = st.selectbox("积分方法", ["RK4 (四阶龙格库塔)", "RK78 (自适应步长)"])
        use_j2 = st.checkbox("启用 J2 摄动", value=True)
        use_drag = st.checkbox("启用大气阻力", value=False)
        
        dt = st.number_input("积分步长 (秒)", value=60.0, min_value=1.0, max_value=600.0, step=10.0)
        duration_min = st.number_input("传播时长 (分钟)", value=180.0, min_value=1.0, step=10.0)
    
    with col2:
        elements_prop = KeplerElements(a_prop, e_prop, i_prop, 0, 0, 0, units='km_deg')
        
        pert = Perturbations(
            use_j2=use_j2,
            use_drag=use_drag,
            use_srp=False
        )
        
        method_str = 'rk4' if 'RK4' in method else 'rk78'
        
        with st.spinner("正在进行数值积分..."):
            times, states = propagate_numerical(
                elements_prop,
                t_start=0,
                t_end=duration_min * 60,
                dt=dt,
                method=method_str,
                perturbations=pert,
                return_all=True
            )
            
            positions = states[:, :3]
            velocities = states[:, 3:]
            
            r_mags = np.linalg.norm(positions, axis=1)
            v_mags = np.linalg.norm(velocities, axis=1)
            altitudes = r_mags - R_EARTH
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='lines',
                line=dict(color='#1f77b4', width=3),
                name='轨道'
            ))
            
            max_r = np.max(r_mags) * 1.2
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
                title='数值积分轨道',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("传播结果分析")
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=times / 60,
            y=altitudes,
            mode='lines',
            name='轨道高度',
            line=dict(color='#2ca02c')
        ))
        fig2.update_layout(
            title='高度随时间变化',
            xaxis_title='时间 (分钟)',
            yaxis_title='高度 (km)',
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col4:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=times / 60,
            y=v_mags,
            mode='lines',
            name='速度',
            line=dict(color='#d62728')
        ))
        fig3.update_layout(
            title='速度随时间变化',
            xaxis_title='时间 (分钟)',
            yaxis_title='速度 (km/s)',
            height=350
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    st.info(
        f"积分步数: {len(times)} | "
        f"初始高度: {altitudes[0]:.1f} km | "
        f"最终高度: {altitudes[-1]:.1f} km | "
        f"高度变化: {altitudes[-1] - altitudes[0]:.2f} km"
    )
    
    st.markdown("---")
    st.subheader("轨道六要素随时间演化")
    
    with st.spinner("正在计算轨道要素..."):
        elements_list = []
        for state in states:
            r = state[:3]
            v = state[3:]
            el = rv_to_kepler(r, v, units='km_deg')
            elements_list.append(el)
        
        time_unit = 'minutes'
        if duration_min >= 1440:
            time_unit = 'days'
        elif duration_min >= 60:
            time_unit = 'hours'
        
        fig_elements = create_orbital_elements_plot(
            times, elements_list, time_unit=time_unit
        )
        st.plotly_chart(fig_elements, use_container_width=True)

elif page == "⚡ 机动优化":
    st.header("机动优化")
    
    opt_type = st.selectbox(
        "优化类型",
        ["燃料最优转移", "轨道保持Δv估算", "多脉冲机动优化"]
    )
    
    if opt_type == "燃料最优转移":
        st.subheader("燃料最优转移时间搜索")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            r1_opt = st.number_input("初始轨道半径 (km)", value=6700.0, min_value=6400.0, step=100.0, key="opt_r1")
            r2_opt = st.number_input("目标轨道半径 (km)", value=10000.0, min_value=6400.0, step=100.0, key="opt_r2")
            max_time_hours = st.slider("最大转移时间 (小时)", 1, 48, 12)
        
        with col2:
            el1_opt = KeplerElements(r1_opt, 0, 0, 0, 0, 0, units='km_deg')
            el2_opt = KeplerElements(r2_opt, 0, 0, 0, 0, 180, units='km_deg')
            
            with st.spinner("正在优化..."):
                result_opt = minimize_fuel_usage(
                    el1_opt, el2_opt, max_time_hours * 3600
                )
                
                st.success(f"最优转移时间: {result_opt['optimal_tof']/3600:.2f} 小时")
                st.write(f"最优 Δv: {result_opt['optimal_dv']:.3f} km/s")
                st.write(f"霍曼转移 Δv: {result_opt['hohmann_dv']:.3f} km/s")
                st.write(f"Δv 比值: {result_opt['dv_ratio']:.3f}")
    
    elif opt_type == "轨道保持Δv估算":
        st.subheader("轨道保持燃料需求估算")
        
        a_sk = st.number_input("半长轴 (km)", value=7000.0, min_value=6400.0, step=100.0, key="sk_a")
        e_sk = st.number_input("偏心率", value=0.0, min_value=0.0, max_value=0.9, step=0.01, key="sk_e")
        i_sk = st.number_input("倾角 (deg)", value=98.0, min_value=0.0, max_value=180.0, step=5.0, key="sk_i")
        duration_months = st.slider("任务时长 (月)", 1, 60, 12)
        
        elements_sk = KeplerElements(a_sk, e_sk, i_sk, 0, 0, 0, units='km_deg')
        pert_sk = Perturbations(use_j2=True)
        
        sk_result = station_keeping_delta_v(elements_sk, pert_sk, duration_days=30)
        
        st.metric("年Δv需求", f"{sk_result['total_dv_per_year']:.3f} km/s/年")
        st.metric(f"{duration_months}月任务总Δv", 
                 f"{sk_result['total_dv_per_year'] * duration_months / 12:.3f} km/s")
        
        st.write("Δv 构成:")
        for k, v in sk_result['dv_breakdown'].items():
            st.write(f"  - {k}: {v:.4f} km/s")
    
    elif opt_type == "多脉冲机动优化":
        st.subheader("多脉冲机动序列优化")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            r1_mp = st.number_input("初始半径 (km)", value=6700.0, min_value=6400.0, key="mp_r1")
            r2_mp = st.number_input("目标半径 (km)", value=8000.0, min_value=6400.0, key="mp_r2")
            n_pulses = st.slider("脉冲数量", 2, 4, 2)
            t_total_hours = st.slider("总转移时间 (小时)", 1, 24, 4)
        
        with col2:
            el1_mp = KeplerElements(r1_mp, 0, 0, 0, 0, 0, units='km_deg')
            el2_mp = KeplerElements(r2_mp, 0, 0, 0, 0, 0, units='km_deg')
            
            with st.spinner("正在进行多脉冲优化..."):
                try:
                    mp_result = optimize_maneuver(
                        el1_mp, el2_mp,
                        n_pulses=n_pulses,
                        t_total=t_total_hours * 3600
                    )
                    
                    if mp_result['success']:
                        st.success(f"优化成功! 总 Δv: {mp_result['total_dv']:.3f} km/s")
                        
                        st.write("各脉冲 Δv:")
                        for i, dv in enumerate(mp_result['dvs']):
                            st.write(f"  脉冲 {i+1}: {np.linalg.norm(dv):.4f} km/s")
                    else:
                        st.warning(f"优化未收敛: {mp_result['message']}")
                except Exception as ex:
                    st.error(f"优化失败: {str(ex)}")

elif page == "🔥 再入轨迹分析":
    st.header("🔥 航天器再入大气层轨迹分析")
    
    st.markdown("""
    本模块实现完整的再入大气层轨迹数值模拟，考虑气动加热、气动力和质量损失的耦合效应，
    支持弹道再入和升力再入两种模式的对比分析。
    """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["参数设置", "仿真结果", "大气模型验证", "碎片散布场", "再入窗口"])
    
    with tab1:
        st.subheader("再入初始条件")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alt_reentry = st.number_input("再入高度 (km)", value=120.0, min_value=80.0, max_value=200.0, step=5.0)
            vel_reentry = st.number_input("再入速度 (km/s)", value=7.8, min_value=5.0, max_value=12.0, step=0.1)
            gamma_reentry = st.number_input("飞行路径角 (°)", value=-5.0, min_value=-8.0, max_value=0.0, step=0.5)
        
        with col2:
            chi_reentry = st.number_input("航向角 (°)", value=0.0, min_value=-180.0, max_value=180.0, step=5.0)
            lat_reentry = st.number_input("初始纬度 (°)", value=0.0, min_value=-90.0, max_value=90.0, step=5.0)
            lon_reentry = st.number_input("初始经度 (°)", value=0.0, min_value=-180.0, max_value=180.0, step=5.0)
        
        with col3:
            st.info("💡 典型值: 高度120km, 速度7.8km/s, 路径角-3°到-5°")
        
        st.markdown("---")
        st.subheader("航天器物理参数")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            mass_vehicle = st.number_input("质量 (kg)", value=800.0, min_value=100.0, max_value=10000.0, step=100.0)
            ref_area = st.number_input("参考面积 (m²)", value=2.0, min_value=0.1, max_value=20.0, step=0.1)
            nose_radius = st.number_input("鼻锥半径 (m)", value=0.05, min_value=0.01, max_value=1.0, step=0.05)
        
        with col5:
            ablation_threshold = st.number_input("烧蚀阈值温度 (K)", value=1000.0, min_value=500.0, max_value=3000.0, step=100.0)
            Cd0 = st.number_input("零攻角阻力系数 Cd0", value=0.15, min_value=0.01, max_value=0.5, step=0.01)
            CL_alpha = st.number_input("升力系数导数 (1/°)", value=0.10, min_value=0.01, max_value=0.3, step=0.01)
        
        with col6:
            alpha_max_LD = st.number_input("最大升阻比对应攻角 (°)", value=10.0, min_value=0.0, max_value=30.0, step=1.0)
            alpha_lifting = st.number_input("升力再入攻角 (°)", value=10.0, min_value=0.0, max_value=30.0, step=1.0)
            bank_angle = st.number_input("倾斜角 (°)", value=45.0, min_value=-90.0, max_value=90.0, step=5.0)
        
        st.markdown("---")
        
        if st.button("开始再入仿真", key="run_reentry_sim", type="primary"):
            with st.spinner("正在进行再入轨迹仿真..."):
                try:
                    vehicle = ReentryVehicle(
                        mass=mass_vehicle,
                        reference_area=ref_area,
                        nose_radius=nose_radius,
                        ablation_threshold=ablation_threshold,
                        Cd0=Cd0,
                        CL_alpha=CL_alpha,
                        alpha_max_LD=alpha_max_LD
                    )
                    
                    init_cond = ReentryInitialConditions(
                        altitude=alt_reentry,
                        velocity=vel_reentry,
                        flight_path_angle=gamma_reentry,
                        heading_angle=chi_reentry,
                        latitude=lat_reentry,
                        longitude=lon_reentry
                    )
                    
                    results_ballistic, results_lifting = simulate_both_modes(
                        vehicle, init_cond,
                        alpha_lifting=alpha_lifting,
                        bank_angle=bank_angle
                    )
                    
                    st.session_state['reentry_results_ballistic'] = results_ballistic
                    st.session_state['reentry_results_lifting'] = results_lifting
                    st.session_state['reentry_sim_done'] = True
                    st.session_state['ablation_threshold'] = ablation_threshold
                    
                    st.success("仿真完成! 请查看 '仿真结果' 标签页")
                except Exception as ex:
                    st.error(f"仿真失败: {str(ex)}")
                    st.session_state['reentry_sim_done'] = False
    
    with tab2:
        if not st.session_state.get('reentry_sim_done', False):
            st.info("请先在 '参数设置' 标签页中运行仿真")
        else:
            results_b = st.session_state['reentry_results_ballistic']
            results_l = st.session_state['reentry_results_lifting']
            
            st.subheader("关键参数汇总")
            
            col_sum1, col_sum2 = st.columns(2)
            
            with col_sum1:
                st.markdown("#### 🔴 弹道再入")
                st.metric("最大热流", f"{results_b['max_heat_flux']:.2e} W/m²", 
                         f"t={results_b['max_heat_flux_time']:.1f}s, h={results_b['max_heat_flux_alt']:.1f}km")
                st.metric("最大表面温度", f"{results_b['max_surface_temp']:.1f} K")
                st.metric("最大过载", f"{results_b['max_overload']:.2f} g", 
                         f"t={results_b['max_overload_time']:.1f}s, h={results_b['max_overload_alt']:.1f}km")
                if results_b['ablation_start_time'] is not None:
                    st.metric("烧蚀开始时刻", f"{results_b['ablation_start_time']:.1f} s", 
                             f"h={results_b['ablation_start_alt']:.1f}km")
                else:
                    st.metric("烧蚀开始时刻", "未达到阈值")
                st.metric("落点纬度", f"{results_b['impact_latitude']:.2f}°")
                st.metric("落点经度", f"{results_b['impact_longitude']:.2f}°")
                st.metric("总飞行时间", f"{results_b['total_time']:.1f} s")
                st.metric("质量损失", f"{results_b['mass_loss']:.2f} kg")
            
            with col_sum2:
                st.markdown("#### 🔵 升力再入")
                st.metric("最大热流", f"{results_l['max_heat_flux']:.2e} W/m²", 
                         f"t={results_l['max_heat_flux_time']:.1f}s, h={results_l['max_heat_flux_alt']:.1f}km")
                st.metric("最大表面温度", f"{results_l['max_surface_temp']:.1f} K")
                st.metric("最大过载", f"{results_l['max_overload']:.2f} g", 
                         f"t={results_l['max_overload_time']:.1f}s, h={results_l['max_overload_alt']:.1f}km")
                if results_l['ablation_start_time'] is not None:
                    st.metric("烧蚀开始时刻", f"{results_l['ablation_start_time']:.1f} s", 
                             f"h={results_l['ablation_start_alt']:.1f}km")
                else:
                    st.metric("烧蚀开始时刻", "未达到阈值")
                st.metric("落点纬度", f"{results_l['impact_latitude']:.2f}°")
                st.metric("落点经度", f"{results_l['impact_longitude']:.2f}°")
                st.metric("总飞行时间", f"{results_l['total_time']:.1f} s")
                st.metric("质量损失", f"{results_l['mass_loss']:.2f} kg")
            
            st.markdown("---")
            
            tab_res1, tab_res2, tab_res3 = st.tabs(["轨迹曲线", "热流与过载", "地面轨迹"])
            
            with tab_res1:
                st.subheader("高度-速度曲线")
                fig1 = create_reentry_altitude_velocity_plot(results_b, results_l)
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("""
                **说明:**
                - 弹道再入（红色）：快速下降，高速度下进入稠密大气层，热流和过载峰值较高
                - 升力再入（蓝色）：利用升力拉平轨迹，减速更平缓，热流和过载峰值较低
                """)
                
                st.subheader("高度-时间曲线")
                fig2 = create_reentry_altitude_time_plot(results_b, results_l)
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab_res2:
                st.subheader("热流密度-时间曲线")
                ablation_thresh_display = st.session_state.get('ablation_threshold', 1000.0)
                fig3 = create_reentry_heat_flux_plot(results_b, results_l, 1e5)
                st.plotly_chart(fig3, use_container_width=True)
                
                st.markdown("""
                **说明:**
                - 驻点热流密度与速度三次方和大气密度平方根成正比
                - 绿色虚线为烧蚀阈值参考线
                - 升力再入通过拉平轨迹延长热流作用时间，但降低峰值热流
                """)
                
                st.subheader("过载-时间曲线")
                fig4 = create_reentry_overload_plot(results_b, results_l)
                st.plotly_chart(fig4, use_container_width=True)
                
                st.markdown("""
                **说明:**
                - 弹道再入过载峰值通常较高（可达8-12g）
                - 升力再入可将过载控制在4-6g范围内，更适合载人航天
                """)
            
            with tab_res3:
                st.subheader("地面轨迹投影")
                fig5 = create_reentry_ground_track_plot(results_b, results_l)
                st.plotly_chart(fig5, use_container_width=True)
                
                st.markdown("""
                **说明:**
                - 圆形标记为再入点，X形标记为落点
                - 升力再入通过倾斜角调节可实现横向机动，改变落点位置
                - 弹道再入主要沿初始航向飞行，机动能力有限
                """)
    
    with tab3:
        st.subheader("1976标准大气模型验证")
        
        st.markdown("""
        本模块使用1976标准大气模型，按高度分段计算温度、压强和密度：
        - 120km ~ 86km：热层模型
        - 86km ~ 47km：平流层顶模型
        - 47km 以下：按标准对流层和平流层分段计算
        """)
        
        alt_test = st.slider("选择高度 (km)", 0.0, 120.0, 50.0, step=1.0)
        rho, T, p = standard_atmosphere_1976(alt_test)
        
        col_atm1, col_atm2, col_atm3 = st.columns(3)
        with col_atm1:
            st.metric("温度", f"{T:.2f} K")
        with col_atm2:
            st.metric("压强", f"{p:.4f} kPa")
        with col_atm3:
            st.metric("密度", f"{rho:.4e} kg/m³")
        
        altitudes = np.linspace(0, 120, 200)
        temps = []
        pressures = []
        densities = []
        
        for alt in altitudes:
            rho, T, p = standard_atmosphere_1976(alt)
            temps.append(T)
            pressures.append(p)
            densities.append(rho)
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig_atm = make_subplots(rows=1, cols=3, subplot_titles=('温度 (K)', '压强 (kPa)', '密度 (kg/m³)'))
        
        fig_atm.add_trace(go.Scatter(x=temps, y=altitudes, mode='lines', line=dict(color='red'), showlegend=False), row=1, col=1)
        fig_atm.add_trace(go.Scatter(x=pressures, y=altitudes, mode='lines', line=dict(color='blue'), showlegend=False), row=1, col=2)
        fig_atm.add_trace(go.Scatter(x=densities, y=altitudes, mode='lines', line=dict(color='green'), showlegend=False), row=1, col=3)
        
        fig_atm.update_xaxes(title_text='T (K)', row=1, col=1)
        fig_atm.update_xaxes(title_text='P (kPa)', type='log', row=1, col=2)
        fig_atm.update_xaxes(title_text='ρ (kg/m³)', type='log', row=1, col=3)
        fig_atm.update_yaxes(title_text='高度 (km)', row=1, col=1)
        
        fig_atm.update_layout(height=400, width=900, title='1976标准大气模型')
        st.plotly_chart(fig_atm, use_container_width=True)
    
    with tab4:
        st.subheader("碎片散布场预测")
        
        st.markdown("""
        当再入过程中动压超过结构极限时，航天器会发生解体并产生多个碎片。
        每个碎片具有不同的面质比和弹道系数，从解体点开始独立飞行直到落地。
        """)
        
        col_deb1, col_deb2, col_deb3 = st.columns(3)
        
        with col_deb1:
            breakup_threshold = st.number_input("解体动压阈值 (kPa)", value=50.0, min_value=10.0, max_value=200.0, step=5.0)
            num_debris = st.slider("碎片数量", min_value=5, max_value=20, value=8, step=1)
        
        with col_deb2:
            min_amr_factor = st.slider("最小面质比倍数", min_value=0.2, max_value=1.0, value=0.5, step=0.1)
            max_amr_factor = st.slider("最大面质比倍数", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        
        with col_deb3:
            velocity_pert = st.number_input("速度扰动 (m/s)", value=50.0, min_value=0.0, max_value=200.0, step=10.0)
            seed_debris = st.number_input("随机种子", value=42, min_value=0, max_value=9999, step=1)
        
        st.markdown("---")
        
        if st.button("开始碎片散布场仿真", key="run_debris_sim", type="primary"):
            with st.spinner("正在进行碎片散布场仿真..."):
                try:
                    vehicle = ReentryVehicle(
                        mass=mass_vehicle,
                        reference_area=ref_area,
                        nose_radius=nose_radius,
                        ablation_threshold=ablation_threshold,
                        Cd0=Cd0,
                        CL_alpha=CL_alpha,
                        alpha_max_LD=alpha_max_LD
                    )
                    
                    init_cond = ReentryInitialConditions(
                        altitude=alt_reentry,
                        velocity=vel_reentry,
                        flight_path_angle=gamma_reentry,
                        heading_angle=chi_reentry,
                        latitude=lat_reentry,
                        longitude=lon_reentry
                    )
                    
                    debris_result = simulate_debris_field(
                        vehicle, init_cond,
                        breakup_threshold_pa=breakup_threshold * 1000.0,
                        num_debris=num_debris,
                        min_amr_factor=min_amr_factor,
                        max_amr_factor=max_amr_factor,
                        velocity_perturbation=velocity_pert,
                        seed=seed_debris
                    )
                    
                    if debris_result is None:
                        st.warning("⚠️ 再入过程中动压未达到解体阈值，航天器未发生解体。")
                        st.info("建议降低解体动压阈值或使用更陡的飞行路径角。")
                    else:
                        st.session_state['debris_result'] = debris_result
                        st.session_state['debris_sim_done'] = True
                        st.success("碎片散布场仿真完成!")
                except Exception as ex:
                    st.error(f"仿真失败: {str(ex)}")
                    st.session_state['debris_sim_done'] = False
        
        st.markdown("---")
        
        if not st.session_state.get('debris_sim_done', False):
            st.info("请设置参数后点击 '开始碎片散布场仿真' 按钮")
        else:
            debris_result = st.session_state['debris_result']
            
            col_deb_sum1, col_deb_sum2, col_deb_sum3 = st.columns(3)
            
            with col_deb_sum1:
                st.metric("解体高度", f"{debris_result['breakup_altitude']:.1f} km")
                st.metric("解体时刻", f"{debris_result['breakup_time']:.1f} s")
                st.metric("解体动压", f"{debris_result['breakup_dynamic_pressure']/1000:.1f} kPa")
            
            with col_deb_sum2:
                st.metric("经度跨度", f"{debris_result['lon_span']:.2f} °")
                st.metric("纬度跨度", f"{debris_result['lat_span']:.2f} °")
                st.metric("碎片数量", f"{len(debris_result['debris'])} 个")
            
            with col_deb_sum3:
                st.metric("平均落点经度", f"{debris_result['mean_impact_lon']:.2f} °")
                st.metric("平均落点纬度", f"{debris_result['mean_impact_lat']:.2f} °")
                st.metric("长轴长度", f"{debris_result['major_axis_length']:.2f} °")
            
            st.markdown("---")
            
            st.subheader("地面散布图")
            fig_debris = create_debris_field_plot(debris_result)
            st.plotly_chart(fig_debris, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("碎片参数表")
            debris_data = []
            for d in debris_result['debris']:
                debris_data.append({
                    '碎片编号': d['id'],
                    '面质比 (m²/kg)': f"{d['area_mass_ratio']:.4f}",
                    '弹道系数 (kg/m²)': f"{d['ballistic_coefficient']:.1f}",
                    '阻力系数 Cd': f"{d['Cd']:.2f}",
                    '落点经度 (°)': f"{d['impact_lon']:.3f}",
                    '落点纬度 (°)': f"{d['impact_lat']:.3f}",
                    '飞行时间 (s)': f"{d['result']['total_time']:.1f}"
                })
            
            df_debris = pd.DataFrame(debris_data)
            st.dataframe(df_debris, use_container_width=True)
    
    with tab5:
        st.subheader("再入窗口约束分析")
        
        st.markdown("""
        通过扫描飞行路径角和航向角的组合，找出能让碎片（或航天器）落在指定安全海域内的再入初始条件。
        """)
        
        col_win1, col_win2, col_win3 = st.columns(3)
        
        with col_win1:
            target_lon = st.number_input("目标落点经度 (°)", value=10.0, min_value=-180.0, max_value=180.0, step=1.0)
            target_lat = st.number_input("目标落点纬度 (°)", value=0.0, min_value=-90.0, max_value=90.0, step=1.0)
            allowed_radius = st.number_input("允许偏差半径 (km)", value=100.0, min_value=10.0, max_value=1000.0, step=10.0)
        
        with col_win2:
            gamma_min = st.number_input("最小飞行路径角 (°)", value=-7.0, min_value=-15.0, max_value=0.0, step=0.5)
            gamma_max = st.number_input("最大飞行路径角 (°)", value=-1.0, min_value=-10.0, max_value=0.0, step=0.5)
            gamma_step = st.number_input("飞行路径角步长 (°)", value=0.5, min_value=0.1, max_value=2.0, step=0.1)
        
        with col_win3:
            chi_delta = st.number_input("航向角扫描范围 (±°)", value=30.0, min_value=5.0, max_value=90.0, step=5.0)
            chi_step = st.number_input("航向角步长 (°)", value=5.0, min_value=1.0, max_value=15.0, step=1.0)
        
        st.markdown("---")
        
        if st.button("开始再入窗口分析", key="run_window_analysis", type="primary"):
            with st.spinner("正在进行再入窗口分析（可能需要几分钟）..."):
                try:
                    vehicle = ReentryVehicle(
                        mass=mass_vehicle,
                        reference_area=ref_area,
                        nose_radius=nose_radius,
                        ablation_threshold=ablation_threshold,
                        Cd0=Cd0,
                        CL_alpha=CL_alpha,
                        alpha_max_LD=alpha_max_LD
                    )
                    
                    init_cond = ReentryInitialConditions(
                        altitude=alt_reentry,
                        velocity=vel_reentry,
                        flight_path_angle=gamma_reentry,
                        heading_angle=chi_reentry,
                        latitude=lat_reentry,
                        longitude=lon_reentry
                    )
                    
                    window_result = analyze_reentry_window(
                        vehicle, init_cond,
                        target_lon=target_lon,
                        target_lat=target_lat,
                        allowed_radius_km=allowed_radius,
                        gamma_min=gamma_min,
                        gamma_max=gamma_max,
                        gamma_step=gamma_step,
                        chi_delta=chi_delta,
                        chi_step=chi_step
                    )
                    
                    st.session_state['window_result'] = window_result
                    st.session_state['window_analysis_done'] = True
                    
                    if window_result['valid_parameters']:
                        st.success(f"找到 {len(window_result['valid_parameters'])} 组可行参数!")
                    else:
                        st.warning("未找到满足条件的可行参数，建议扩大扫描范围或放宽允许偏差。")
                except Exception as ex:
                    st.error(f"分析失败: {str(ex)}")
                    st.session_state['window_analysis_done'] = False
        
        st.markdown("---")
        
        if not st.session_state.get('window_analysis_done', False):
            st.info("请设置参数后点击 '开始再入窗口分析' 按钮")
        else:
            window_result = st.session_state['window_result']
            
            col_win_sum1, col_win_sum2 = st.columns(2)
            
            with col_win_sum1:
                st.metric("扫描参数组合数", f"{len(window_result['all_results'])} 组")
                st.metric("可行参数组合数", f"{len(window_result['valid_parameters'])} 组")
            
            with col_win_sum2:
                if window_result['valid_parameters']:
                    best = window_result['valid_parameters'][0]
                    st.metric("最小偏差", f"{best['distance_km']:.1f} km")
                    st.metric("最优飞行路径角", f"{best['flight_path_angle']:.1f} °")
                else:
                    st.metric("最小偏差", "无可行解")
                    st.metric("最优飞行路径角", "无可行解")
            
            st.markdown("---")
            
            st.subheader("可行域热力图")
            fig_window = create_reentry_window_heatmap(window_result)
            st.plotly_chart(fig_window, use_container_width=True)
            
            st.markdown("---")
            
            if window_result['valid_parameters']:
                st.subheader("推荐参数组合表（按偏差从小到大排序）")
                valid_data = []
                for i, p in enumerate(window_result['valid_parameters'][:20]):
                    valid_data.append({
                        '排名': i + 1,
                        '飞行路径角 (°)': f"{p['flight_path_angle']:.1f}",
                        '航向角 (°)': f"{p['heading_angle']:.1f}",
                        '落点经度 (°)': f"{p['impact_longitude']:.3f}",
                        '落点纬度 (°)': f"{p['impact_latitude']:.3f}",
                        '偏差距离 (km)': f"{p['distance_km']:.1f}"
                    })
                
                df_valid = pd.DataFrame(valid_data)
                st.dataframe(df_valid, use_container_width=True)
                
                if len(window_result['valid_parameters']) > 20:
                    st.info(f"仅显示前20组最优参数，共 {len(window_result['valid_parameters'])} 组可行解。")

elif page == "☄️ 碰撞风险评估":
    st.header("☄️ 空间碎片碰撞风险评估")
    st.markdown("""
    本模块实现空间碎片碰撞风险评估功能，支持单对碰撞分析、批量碎片筛查和碰撞统计分析。
    采用B平面短期遭遇模型计算碰撞概率，支持自动规避机动设计。
    """)
    
    tab1, tab2, tab3 = st.tabs(["单对碰撞分析", "批量筛查", "碰撞统计"])
    
    with tab1:
        st.subheader("单对碰撞分析")
        
        col_params, col_results = st.columns([1, 1])
        
        with col_params:
            st.markdown("#### 物体1（主航天器）")
            a1 = st.number_input("半长轴 a₁ (km)", value=7000.0, min_value=6400.0, step=100.0, key="cr_a1")
            e1 = st.number_input("偏心率 e₁", value=0.001, min_value=0.0, max_value=0.5, step=0.001, key="cr_e1")
            i1 = st.number_input("倾角 i₁ (°)", value=97.5, min_value=0.0, max_value=180.0, step=1.0, key="cr_i1")
            raan1 = st.number_input("RAAN Ω₁ (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="cr_raan1")
            argp1 = st.number_input("近地点幅角 ω₁ (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="cr_argp1")
            nu1 = st.number_input("真近点角 ν₁ (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="cr_nu1")
            r1_radius = st.number_input("等效半径 r₁ (m)", value=5.0, min_value=0.1, step=0.5, key="cr_r1")
            sigma1 = st.number_input("位置不确定性 σ₁ (km)", value=0.05, min_value=0.001, step=0.01, key="cr_sigma1")
            
            st.markdown("---")
            st.markdown("#### 物体2（碎片/目标）")
            a2 = st.number_input("半长轴 a₂ (km)", value=7000.05, min_value=6400.0, step=100.0, key="cr_a2")
            e2 = st.number_input("偏心率 e₂", value=0.001, min_value=0.0, max_value=0.5, step=0.001, key="cr_e2")
            i2 = st.number_input("倾角 i₂ (°)", value=97.5, min_value=0.0, max_value=180.0, step=1.0, key="cr_i2")
            raan2 = st.number_input("RAAN Ω₂ (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="cr_raan2")
            argp2 = st.number_input("近地点幅角 ω₂ (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="cr_argp2")
            nu2 = st.number_input("真近点角 ν₂ (°)", value=0.05, min_value=0.0, max_value=360.0, step=1.0, key="cr_nu2")
            r2_radius = st.number_input("等效半径 r₂ (m)", value=3.0, min_value=0.1, step=0.5, key="cr_r2")
            sigma2 = st.number_input("位置不确定性 σ₂ (km)", value=0.05, min_value=0.001, step=0.01, key="cr_sigma2")
            
            st.markdown("---")
            st.markdown("#### 分析参数")
            pred_days = st.slider("预测时长 (天)", 1, 30, 7, key="cr_pred_days")
            prob_threshold = st.number_input("碰撞概率预警阈值", value=1e-4, format="%.1e", key="cr_threshold")
            t_maneuver_offset_hours = st.slider("规避机动提前时间 (小时)", 1, 24, 6, key="cr_maneuver_offset")
            
            if st.button("开始碰撞风险分析", type="primary", key="cr_run_single"):
                with st.spinner("正在进行碰撞风险分析..."):
                    try:
                        el1 = KeplerElements(a1, e1, i1, raan1, argp1, nu1, units='km_deg')
                        el2 = KeplerElements(a2, e2, i2, raan2, argp2, nu2, units='km_deg')
                        
                        cov1 = get_position_covariance(sigma1)
                        cov2 = get_position_covariance(sigma2)
                        
                        result = collision_risk_analysis(
                            el1, el2, cov1, cov2,
                            radius1=r1_radius / 1000.0,
                            radius2=r2_radius / 1000.0,
                            t_start=0,
                            t_end=pred_days * 86400,
                            dt_coarse=30
                        )
                        
                        st.session_state['single_collision_result'] = result
                        st.session_state['single_collision_done'] = True
                        st.session_state['el1_single'] = el1
                        st.session_state['el2_single'] = el2
                        st.session_state['cov1_single'] = cov1
                        st.session_state['cov2_single'] = cov2
                        st.session_state['r1_single'] = r1_radius / 1000.0
                        st.session_state['r2_single'] = r2_radius / 1000.0
                        st.session_state['prob_threshold_single'] = prob_threshold
                        st.session_state['t_maneuver_offset_single'] = t_maneuver_offset_hours * 3600
                        
                        st.success("分析完成!")
                    except Exception as ex:
                        st.error(f"分析失败: {str(ex)}")
                        st.session_state['single_collision_done'] = False
        
        with col_results:
            if not st.session_state.get('single_collision_done', False):
                st.info("请设置参数后点击 '开始碰撞风险分析' 按钮")
            else:
                result = st.session_state['single_collision_result']
                
                st.markdown("#### 分析结果")
                
                t_ca_hours = result['t_closest'] / 3600
                prob = result['collision_probability']
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("最近接近距离", f"{result['min_distance']:.3f} km")
                    st.metric("最近接近时刻", f"{t_ca_hours:.2f} 小时")
                with col_res2:
                    if prob > prob_threshold:
                        st.metric("碰撞概率", f"{prob:.2e}", "⚠️ 超过阈值", delta_color="inverse")
                    else:
                        st.metric("碰撞概率", f"{prob:.2e}", "✓ 安全", delta_color="normal")
                    st.metric("碰撞截面半径", f"{result['collision_radius'] * 1000:.1f} m")
                
                st.markdown("**B平面遭遇位置:**")
                st.write(f"ξ = {result['b_plane_pos'][0]:.4f} km, ζ = {result['b_plane_pos'][1]:.4f} km")
                
                st.markdown("**相对速度:**")
                v_rel_mag = np.linalg.norm(result['v_rel'])
                st.write(f"{v_rel_mag:.3f} km/s")
                
                if prob > prob_threshold:
                    st.warning("⚠️ 碰撞概率超过预警阈值，建议执行规避机动")
                    
                    if st.button("计算最小Δv规避方案", key="cr_calc_maneuver"):
                        with st.spinner("正在搜索最优规避方案..."):
                            try:
                                maneuver = find_min_dv_maneuver(
                                    st.session_state['el1_single'],
                                    st.session_state['el2_single'],
                                    st.session_state['cov1_single'],
                                    st.session_state['cov2_single'],
                                    radius1=st.session_state['r1_single'],
                                    radius2=st.session_state['r2_single'],
                                    t_ca=result['t_closest'],
                                    t_maneuver_offset=st.session_state['t_maneuver_offset_single'],
                                    target_prob=prob_threshold / 100,
                                    max_dv=0.5,
                                    dv_step=0.0005
                                )
                                
                                st.session_state['maneuver_result'] = maneuver
                                st.session_state['maneuver_done'] = True
                                
                                if maneuver['success']:
                                    st.success(f"找到最优规避方案!")
                                else:
                                    st.warning(maneuver['message'])
                            except Exception as ex:
                                st.error(f"规避方案计算失败: {str(ex)}")
                    
                    if st.session_state.get('maneuver_done', False):
                        maneuver = st.session_state['maneuver_result']
                        if maneuver['success']:
                            st.markdown("#### 规避方案")
                            col_man1, col_man2 = st.columns(2)
                            with col_man1:
                                st.metric("所需Δv", f"{maneuver['dv_magnitude'] * 1000:.2f} m/s")
                                st.metric("机动时刻", f"{maneuver['t_maneuver'] / 3600:.2f} 小时")
                            with col_man2:
                                st.metric("径向分量", f"{maneuver['radial_component'] * 1000:.2f} m/s")
                                st.metric("迹向分量", f"{maneuver['along_track_component'] * 1000:.2f} m/s")
                                st.metric("法向分量", f"{maneuver['normal_component'] * 1000:.2f} m/s")
                            
                            st.markdown("**规避后碰撞概率:**")
                            prob_after = maneuver['result_after_maneuver']['collision_probability']
                            st.write(f"{prob_after:.2e} (降至目标值以下)")
                
                st.markdown("---")
                
                fig_bplane = create_b_plane_plot(result)
                st.plotly_chart(fig_bplane, use_container_width=True)
                
                fig_approach = create_approach_distance_plot(result)
                st.plotly_chart(fig_approach, use_container_width=True)
    
    with tab2:
        st.subheader("批量碎片筛查")
        
        col_batch_params, col_batch_results = st.columns([1, 2])
        
        with col_batch_params:
            st.markdown("#### 主航天器参数")
            a_main = st.number_input("半长轴 (km)", value=7000.0, min_value=6400.0, step=100.0, key="batch_a_main")
            e_main = st.number_input("偏心率", value=0.001, min_value=0.0, max_value=0.5, step=0.001, key="batch_e_main")
            i_main = st.number_input("倾角 (°)", value=97.5, min_value=0.0, max_value=180.0, step=1.0, key="batch_i_main")
            raan_main = st.number_input("RAAN (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="batch_raan_main")
            argp_main = st.number_input("近地点幅角 (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="batch_argp_main")
            nu_main = st.number_input("真近点角 (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="batch_nu_main")
            r_main_radius = st.number_input("等效半径 (m)", value=5.0, min_value=0.1, step=0.5, key="batch_r_main")
            sigma_main = st.number_input("位置不确定性 (km)", value=0.05, min_value=0.001, step=0.01, key="batch_sigma_main")
            
            st.markdown("---")
            st.markdown("#### 筛查参数")
            num_debris = st.slider("碎片目标数量 (1-20)", 1, 20, 3, key="batch_num_debris")
            batch_pred_days = st.slider("预测时长 (天)", 1, 30, 7, key="batch_pred_days")
            batch_prob_threshold = st.number_input("预警阈值", value=1e-4, format="%.1e", key="batch_threshold")
            batch_maneuver_offset = st.slider("规避提前时间 (小时)", 1, 24, 6, key="batch_maneuver_offset")
            
            st.markdown("---")
            st.markdown("#### 碎片参数输入")
            
            input_mode = st.radio("碎片输入方式", ["手动输入", "随机生成示例"], key="batch_input_mode", horizontal=True)
            
            if input_mode == "手动输入":
                st.info(f"请在下方输入 {num_debris} 个碎片的参数")
            else:
                if st.button("填充随机示例数据", key="batch_fill_random"):
                    np.random.seed(42)
                    for i in range(num_debris):
                        st.session_state[f'batch_a_d_{i}'] = a_main + np.random.uniform(-5, 5)
                        st.session_state[f'batch_e_d_{i}'] = np.random.uniform(0, 0.005)
                        st.session_state[f'batch_i_d_{i}'] = i_main + np.random.uniform(-0.5, 0.5)
                        st.session_state[f'batch_raan_d_{i}'] = np.random.uniform(0, 360)
                        st.session_state[f'batch_argp_d_{i}'] = np.random.uniform(0, 360)
                        st.session_state[f'batch_nu_d_{i}'] = np.random.uniform(0, 360)
                        st.session_state[f'batch_r_d_{i}'] = np.random.uniform(1, 5)
                        st.session_state[f'batch_sigma_d_{i}'] = np.random.uniform(0.02, 0.1)
                    st.success("已填充随机示例数据")
            
            with st.expander("展开/折叠 碎片参数详情", expanded=True):
                for i in range(num_debris):
                    st.markdown(f"##### 碎片 #{i+1}")
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.number_input(f"半长轴 a_{i+1} (km)", value=st.session_state.get(f'batch_a_d_{i}', 7000.0 + i*0.5), 
                                       min_value=6400.0, step=10.0, key=f'batch_a_d_{i}')
                        st.number_input(f"偏心率 e_{i+1}", value=st.session_state.get(f'batch_e_d_{i}', 0.001 + i*0.001), 
                                       min_value=0.0, max_value=0.5, step=0.001, key=f'batch_e_d_{i}')
                        st.number_input(f"倾角 i_{i+1} (°)", value=st.session_state.get(f'batch_i_d_{i}', 97.5 + i*0.1), 
                                       min_value=0.0, max_value=180.0, step=0.5, key=f'batch_i_d_{i}')
                    with col_d2:
                        st.number_input(f"RAAN Ω_{i+1} (°)", value=st.session_state.get(f'batch_raan_d_{i}', 0.0 + i*5.0), 
                                       min_value=0.0, max_value=360.0, step=1.0, key=f'batch_raan_d_{i}')
                        st.number_input(f"近地点幅角 ω_{i+1} (°)", value=st.session_state.get(f'batch_argp_d_{i}', 0.0 + i*10.0), 
                                       min_value=0.0, max_value=360.0, step=1.0, key=f'batch_argp_d_{i}')
                        st.number_input(f"真近点角 ν_{i+1} (°)", value=st.session_state.get(f'batch_nu_d_{i}', 0.0 + i*15.0), 
                                       min_value=0.0, max_value=360.0, step=1.0, key=f'batch_nu_d_{i}')
                    
                    col_d3, col_d4 = st.columns(2)
                    with col_d3:
                        st.number_input(f"等效半径 r_{i+1} (m)", value=st.session_state.get(f'batch_r_d_{i}', 3.0), 
                                       min_value=0.1, step=0.5, key=f'batch_r_d_{i}')
                    with col_d4:
                        st.number_input(f"位置不确定性 σ_{i+1} (km)", value=st.session_state.get(f'batch_sigma_d_{i}', 0.05), 
                                       min_value=0.001, step=0.01, key=f'batch_sigma_d_{i}')
                    
                    if i < num_debris - 1:
                        st.markdown("---")
            
            if st.button("开始批量筛查", type="primary", key="batch_run"):
                with st.spinner(f"正在对 {num_debris} 个目标进行碰撞风险筛查..."):
                    try:
                        el_main = KeplerElements(a_main, e_main, i_main, raan_main, argp_main, nu_main, units='km_deg')
                        cov_main = get_position_covariance(sigma_main)
                        
                        debris_list = []
                        debris_cov_list = []
                        debris_radii = []
                        
                        for i in range(num_debris):
                            a_d = st.session_state[f'batch_a_d_{i}']
                            e_d = st.session_state[f'batch_e_d_{i}']
                            i_d = st.session_state[f'batch_i_d_{i}']
                            raan_d = st.session_state[f'batch_raan_d_{i}']
                            argp_d = st.session_state[f'batch_argp_d_{i}']
                            nu_d = st.session_state[f'batch_nu_d_{i}']
                            
                            el_d = KeplerElements(a_d, e_d, i_d, raan_d, argp_d, nu_d, units='km_deg')
                            debris_list.append(el_d)
                            
                            sigma_d = st.session_state[f'batch_sigma_d_{i}']
                            debris_cov_list.append(get_position_covariance(sigma_d))
                            
                            r_d = st.session_state[f'batch_r_d_{i}']
                            debris_radii.append(r_d / 1000.0)
                        
                        batch_results = batch_screening(
                            el_main, debris_list, cov_main, debris_cov_list,
                            main_radius=r_main_radius / 1000.0,
                            debris_radii=debris_radii,
                            t_start=0,
                            t_end=batch_pred_days * 86400,
                            dt_coarse=30,
                            prob_threshold=batch_prob_threshold,
                            target_prob=batch_prob_threshold / 100,
                            t_maneuver_offset=batch_maneuver_offset * 3600
                        )
                        
                        st.session_state['batch_results'] = batch_results
                        st.session_state['batch_done'] = True
                        
                        num_high_risk = sum(1 for r in batch_results if r['exceeds_threshold'])
                        if num_high_risk > 0:
                            st.warning(f"筛查完成! 发现 {num_high_risk} 个高风险目标")
                        else:
                            st.success("筛查完成! 所有目标碰撞概率均在安全范围内")
                    except Exception as ex:
                        st.error(f"筛查失败: {str(ex)}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.session_state['batch_done'] = False
        
        with col_batch_results:
            if not st.session_state.get('batch_done', False):
                st.info("请设置参数后点击 '开始批量筛查' 按钮")
            else:
                batch_results = st.session_state['batch_results']
                
                fig_table = create_batch_screening_table(batch_results, batch_prob_threshold)
                st.plotly_chart(fig_table, use_container_width=True)
                
                high_risk = [r for r in batch_results if r['exceeds_threshold']]
                if high_risk:
                    st.markdown("---")
                    st.markdown("#### 高风险目标详情")
                    
                    for idx, risk in enumerate(high_risk):
                        with st.expander(f"碎片 #{risk['debris_id']} - 碰撞概率: {risk['collision_probability']:.2e}", expanded=(idx == 0)):
                            t_ca = risk['t_closest'] / 3600
                            st.write(f"最近接近距离: {risk['min_distance']:.3f} km")
                            st.write(f"最近接近时刻: {t_ca:.2f} 小时")
                            
                            if 'maneuver' in risk and risk['maneuver']['success']:
                                man = risk['maneuver']
                                st.markdown("**推荐规避方案:**")
                                st.write(f"所需Δv: {man['dv_magnitude'] * 1000:.2f} m/s")
                                st.write(f"机动时刻: {man['t_maneuver'] / 3600:.2f} 小时")
                                st.write(f"径向分量: {man['radial_component'] * 1000:.2f} m/s")
                                st.write(f"迹向分量: {man['along_track_component'] * 1000:.2f} m/s")
                                st.write(f"法向分量: {man['normal_component'] * 1000:.2f} m/s")
                            elif 'maneuver' in risk and not risk['maneuver']['success']:
                                st.warning("未找到有效的规避方案")
                            
                            fig_bplane_risk = create_b_plane_plot(risk['raw_result'])
                            st.plotly_chart(fig_bplane_risk, use_container_width=True)
    
    with tab3:
        st.subheader("碰撞统计分析")
        
        col_stats_params, col_stats_results = st.columns([1, 2])
        
        with col_stats_params:
            st.markdown("#### 主航天器参数")
            a_stats = st.number_input("半长轴 (km)", value=7000.0, min_value=6400.0, step=100.0, key="stats_a")
            e_stats = st.number_input("偏心率", value=0.001, min_value=0.0, max_value=0.5, step=0.001, key="stats_e")
            i_stats = st.number_input("倾角 (°)", value=97.5, min_value=0.0, max_value=180.0, step=1.0, key="stats_i")
            raan_stats = st.number_input("RAAN (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="stats_raan")
            argp_stats = st.number_input("近地点幅角 (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="stats_argp")
            nu_stats = st.number_input("真近点角 (°)", value=0.0, min_value=0.0, max_value=360.0, step=1.0, key="stats_nu")
            r_stats_radius = st.number_input("等效半径 (m)", value=5.0, min_value=0.1, step=0.5, key="stats_r_main")
            sigma_stats = st.number_input("位置不确定性 (km)", value=0.05, min_value=0.001, step=0.01, key="stats_sigma")
            
            st.markdown("---")
            st.markdown("#### 统计参数")
            num_debris_stats = st.slider("碎片数量 (1-20)", 1, 20, 5, key="stats_num_debris")
            
            st.markdown("---")
            st.markdown("#### 碎片参数输入")
            
            stats_input_mode = st.radio("碎片输入方式", ["手动输入", "随机生成示例"], 
                                       key="stats_input_mode", horizontal=True)
            
            if stats_input_mode == "随机生成示例":
                if st.button("填充随机示例数据", key="stats_fill_random"):
                    np.random.seed(123)
                    for i in range(num_debris_stats):
                        st.session_state[f'stats_a_d_{i}'] = a_stats + np.random.uniform(-20, 20)
                        st.session_state[f'stats_e_d_{i}'] = np.random.uniform(0, 0.01)
                        st.session_state[f'stats_i_d_{i}'] = i_stats + np.random.uniform(-1, 1)
                        st.session_state[f'stats_raan_d_{i}'] = np.random.uniform(0, 360)
                        st.session_state[f'stats_argp_d_{i}'] = np.random.uniform(0, 360)
                        st.session_state[f'stats_nu_d_{i}'] = np.random.uniform(0, 360)
                        st.session_state[f'stats_r_d_{i}'] = np.random.uniform(1, 5)
                        st.session_state[f'stats_sigma_d_{i}'] = np.random.uniform(0.02, 0.1)
                    st.success("已填充随机示例数据")
            
            with st.expander("展开/折叠 碎片参数详情", expanded=False):
                for i in range(num_debris_stats):
                    st.markdown(f"##### 碎片 #{i+1}")
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.number_input(f"半长轴 a_{i+1} (km)", value=st.session_state.get(f'stats_a_d_{i}', 7000.0 + i*2.0), 
                                       min_value=6400.0, step=10.0, key=f'stats_a_d_{i}')
                        st.number_input(f"偏心率 e_{i+1}", value=st.session_state.get(f'stats_e_d_{i}', 0.001 + i*0.002), 
                                       min_value=0.0, max_value=0.5, step=0.001, key=f'stats_e_d_{i}')
                        st.number_input(f"倾角 i_{i+1} (°)", value=st.session_state.get(f'stats_i_d_{i}', 97.5 + i*0.2), 
                                       min_value=0.0, max_value=180.0, step=0.5, key=f'stats_i_d_{i}')
                    with col_s2:
                        st.number_input(f"RAAN Ω_{i+1} (°)", value=st.session_state.get(f'stats_raan_d_{i}', 0.0 + i*10.0), 
                                       min_value=0.0, max_value=360.0, step=1.0, key=f'stats_raan_d_{i}')
                        st.number_input(f"近地点幅角 ω_{i+1} (°)", value=st.session_state.get(f'stats_argp_d_{i}', 0.0 + i*20.0), 
                                       min_value=0.0, max_value=360.0, step=1.0, key=f'stats_argp_d_{i}')
                        st.number_input(f"真近点角 ν_{i+1} (°)", value=st.session_state.get(f'stats_nu_d_{i}', 0.0 + i*30.0), 
                                       min_value=0.0, max_value=360.0, step=1.0, key=f'stats_nu_d_{i}')
                    
                    col_s3, col_s4 = st.columns(2)
                    with col_s3:
                        st.number_input(f"等效半径 r_{i+1} (m)", value=st.session_state.get(f'stats_r_d_{i}', 3.0), 
                                       min_value=0.1, step=0.5, key=f'stats_r_d_{i}')
                    with col_s4:
                        st.number_input(f"位置不确定性 σ_{i+1} (km)", value=st.session_state.get(f'stats_sigma_d_{i}', 0.05), 
                                       min_value=0.001, step=0.01, key=f'stats_sigma_d_{i}')
                    
                    if i < num_debris_stats - 1:
                        st.markdown("---")
            
            if st.button("生成碰撞统计数据", type="primary", key="stats_run"):
                with st.spinner("正在生成碰撞统计数据..."):
                    try:
                        el_main_stats = KeplerElements(a_stats, e_stats, i_stats, raan_stats, argp_stats, nu_stats, units='km_deg')
                        cov_main_stats = get_position_covariance(sigma_stats)
                        
                        debris_list_stats = []
                        debris_cov_list_stats = []
                        debris_radii_stats = []
                        
                        for i in range(num_debris_stats):
                            a_d = st.session_state[f'stats_a_d_{i}']
                            e_d = st.session_state[f'stats_e_d_{i}']
                            i_d = st.session_state[f'stats_i_d_{i}']
                            raan_d = st.session_state[f'stats_raan_d_{i}']
                            argp_d = st.session_state[f'stats_argp_d_{i}']
                            nu_d = st.session_state[f'stats_nu_d_{i}']
                            
                            el_d = KeplerElements(a_d, e_d, i_d, raan_d, argp_d, nu_d, units='km_deg')
                            debris_list_stats.append(el_d)
                            
                            sigma_d = st.session_state[f'stats_sigma_d_{i}']
                            debris_cov_list_stats.append(get_position_covariance(sigma_d))
                            
                            r_d = st.session_state[f'stats_r_d_{i}']
                            debris_radii_stats.append(r_d / 1000.0)
                        
                        stats_result = generate_collision_statistics(
                            el_main_stats, debris_list_stats, cov_main_stats, debris_cov_list_stats,
                            main_radius=r_stats_radius / 1000.0,
                            debris_radii=debris_radii_stats,
                            time_windows=[1, 2, 3, 5, 7, 14, 30],
                            t_start=0,
                            t_end=30 * 86400
                        )
                        
                        st.session_state['stats_result'] = stats_result
                        st.session_state['stats_done'] = True
                        st.session_state['stats_num_debris'] = num_debris_stats
                        st.success("统计数据生成完成!")
                    except Exception as ex:
                        st.error(f"统计分析失败: {str(ex)}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.session_state['stats_done'] = False
        
        with col_stats_results:
            if not st.session_state.get('stats_done', False):
                st.info("请设置参数后点击 '生成碰撞统计数据' 按钮")
            else:
                stats_result = st.session_state['stats_result']
                display_num = st.session_state.get('stats_num_debris', num_debris_stats)
                
                fig_cumulative = create_collision_cumulative_plot(stats_result)
                st.plotly_chart(fig_cumulative, use_container_width=True)
                
                fig_dist = create_distance_distribution_plot(stats_result)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 统计摘要")
                
                col_sum1, col_sum2 = st.columns(2)
                with col_sum1:
                    st.metric("分析碎片数量", f"{display_num} 个")
                    st.metric("最小接近距离", f"{np.min(stats_result['distances']):.2f} km")
                with col_sum2:
                    st.metric("平均接近距离", f"{np.mean(stats_result['distances']):.2f} km")
                    st.metric("最大接近距离", f"{np.max(stats_result['distances']):.2f} km")
                
                final_prob = stats_result['cumulative_probs'][-1]['prob_any_collision']
                st.metric("30天累积碰撞概率", f"{final_prob:.2e}")

st.markdown("---")
st.caption("🛰️ 航天器轨道力学分析与变轨仿真工具 | 基于开普勒轨道力学与数值方法")
