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
    perigee_event, apogee_event, altitude_event
)
from maneuvers import (
    hohmann_transfer, bielliptic_transfer,
    inclination_change, phasing_maneuver,
    lambert_transfer, multi_turn_lambert
)
from visualization import (
    create_3d_orbit_plot, create_ground_track_plot,
    create_velocity_profile, create_maneuver_plot,
    create_coverage_heatmap, create_coverage_by_latitude
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
            "⚡ 机动优化"
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
            tof_hours = st.slider("转移时间 (小时)", 0.1, 24.0, 2.0, step=0.1)
        
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
                st.info("尝试调整转移时间或轨道参数")

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
        
        a_drag = st.number_input("初始半长轴 (km)", value=6600.0, min_value=6400.0, step=10.0, key="drag_a")
        e_drag = st.number_input("初始偏心率", value=0.05, min_value=0.0, max_value=0.5, step=0.01, key="drag_e")
        area_mass = st.slider("面质比 (m²/kg)", 0.001, 0.1, 0.01, step=0.001)
        sim_days = st.slider("仿真天数", 1, 365, 30)
        
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
    
    fig_heatmap = create_coverage_heatmap(None)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    fig_lat = create_coverage_by_latitude()
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

st.markdown("---")
st.caption("🛰️ 航天器轨道力学分析与变轨仿真工具 | 基于开普勒轨道力学与数值方法")
