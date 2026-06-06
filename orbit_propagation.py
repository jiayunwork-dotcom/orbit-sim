import numpy as np
from orbit_core import (
    MU_EARTH, R_EARTH, J2_EARTH, OMEGA_EARTH,
    KeplerElements, kepler_to_rv, rv_to_kepler
)


class Perturbations:
    def __init__(self, use_j2=True, use_drag=False, use_srp=False,
                 Cd=2.2, area_mass_ratio=0.01,
                 solar_pressure=4.56e-6, reflectivity=1.0):
        self.use_j2 = use_j2
        self.use_drag = use_drag
        self.use_srp = use_srp
        self.Cd = Cd
        self.area_mass_ratio = area_mass_ratio
        self.solar_pressure = solar_pressure
        self.reflectivity = reflectivity
    
    def _exponential_atmosphere(self, r_mag):
        h = r_mag - R_EARTH
        
        if h < 100:
            h0 = 100
            rho0 = 4.8e-7
            H = 26.5
        elif h < 150:
            h0 = 150
            rho0 = 1.8e-8
            H = 38.5
        elif h < 200:
            h0 = 200
            rho0 = 2.5e-9
            H = 40.5
        elif h < 250:
            h0 = 250
            rho0 = 6.0e-10
            H = 44.0
        elif h < 300:
            h0 = 300
            rho0 = 1.9e-10
            H = 50.5
        elif h < 400:
            h0 = 400
            rho0 = 3.7e-11
            H = 58.0
        elif h < 500:
            h0 = 500
            rho0 = 6.9e-12
            H = 60.0
        elif h < 600:
            h0 = 600
            rho0 = 1.4e-12
            H = 73.0
        elif h < 700:
            h0 = 700
            rho0 = 3.5e-13
            H = 88.5
        elif h < 800:
            h0 = 800
            rho0 = 1.0e-13
            H = 124.5
        elif h < 900:
            h0 = 900
            rho0 = 3.6e-14
            H = 181.0
        elif h < 1000:
            h0 = 1000
            rho0 = 1.5e-14
            H = 268.0
        else:
            return 0.0
        
        return rho0 * np.exp(-(h - h0) / H)
    
    def j2_acceleration(self, r):
        r_mag = np.linalg.norm(r)
        x, y, z = r
        
        factor = -1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / r_mag**5
        z_factor = 5 * z**2 / r_mag**2
        
        ax = factor * x * (1 - z_factor)
        ay = factor * y * (1 - z_factor)
        az = factor * z * (3 - z_factor)
        
        return np.array([ax, ay, az])
    
    def drag_acceleration(self, r, v):
        r_mag = np.linalg.norm(r)
        rho = self._exponential_atmosphere(r_mag)
        
        v_atmo = np.array([-OMEGA_EARTH * r[1], OMEGA_EARTH * r[0], 0.0])
        v_rel = v - v_atmo
        v_rel_mag = np.linalg.norm(v_rel)
        
        if v_rel_mag < 1e-10 or rho == 0:
            return np.zeros(3)
        
        return -0.5 * self.Cd * self.area_mass_ratio * rho * v_rel_mag * v_rel
    
    def srp_acceleration(self, r, sun_dir=None):
        if sun_dir is None:
            sun_dir = np.array([1.0, 0.0, 0.0])
        
        sun_dir = sun_dir / np.linalg.norm(sun_dir)
        r_mag = np.linalg.norm(r)
        
        earth_rad_angle = R_EARTH / r_mag
        sun_angle = np.arccos(np.dot(r, sun_dir) / r_mag)
        
        if sun_angle < np.pi / 2 - earth_rad_angle:
            return np.zeros(3)
        
        return -self.solar_pressure * (1 + self.reflectivity) * self.area_mass_ratio * sun_dir
    
    def total_acceleration(self, r, v, sun_dir=None):
        acc = -MU_EARTH * r / np.linalg.norm(r)**3
        
        if self.use_j2:
            acc += self.j2_acceleration(r)
        
        if self.use_drag:
            acc += self.drag_acceleration(r, v)
        
        if self.use_srp:
            acc += self.srp_acceleration(r, sun_dir)
        
        return acc


def rk4_step(state, dt, perturbations, sun_dir=None):
    r, v = state[:3], state[3:]
    
    def derivative(state):
        r, v = state[:3], state[3:]
        a = perturbations.total_acceleration(r, v, sun_dir)
        return np.concatenate([v, a])
    
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk78_step(state, dt, perturbations, sun_dir=None, tol=1e-6):
    r, v = state[:3], state[3:]
    
    def derivative(state):
        r, v = state[:3], state[3:]
        a = perturbations.total_acceleration(r, v, sun_dir)
        return np.concatenate([v, a])
    
    c = np.array([0., 1./12, 1./6, 1./4, 1./2, 5./6, 1./6, 2./3, 1./3, 1., 0., 1.])
    a_coeffs = [
        [],
        [1./12],
        [1./24, 1./24],
        [1./12, 0., 1./4],
        [1./8, 0., 0., 3./8],
        [5./24, 0., 0., -27./24, 25./24],
        [1./12, 0., 0., 0., 1./2, 0., 1./12],
        [-25./48, 0., 0., 243./48, -125./16, 125./24, 0., 5./48],
        [3./20, 0., 0., -27./20, 27./10, -21./10, 3./5, 9./20, 9./20],
        [-17./48, 0., 0., 99./16, -125./12, 125./8, -5./16, 5./48, 0., 0.],
        [10./42, 0., 0., 0., 125./14, -125./14, 50./21, -125./28, 0., 0., 5./42],
        [1./10, 0., 0., 0., -15./4, 15./4, -5./2, 5./2, 1./2, 0., 0., 0.]
    ]
    b7 = np.array([41./840, 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 41./840, 0.])
    b8 = np.array([0., 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 0., 41./840])
    
    current_dt = dt
    max_attempts = 20
    
    for attempt in range(max_attempts):
        k = []
        k.append(derivative(state))
        
        for i in range(1, 12):
            y = state.copy()
            for j in range(i):
                y += current_dt * a_coeffs[i][j] * k[j]
            k.append(derivative(y))
        
        y7 = state.copy()
        y8 = state.copy()
        for i in range(12):
            y7 += current_dt * b7[i] * k[i]
            y8 += current_dt * b8[i] * k[i]
        
        err = np.linalg.norm(y8 - y7)
        
        if err <= tol:
            return y8, current_dt
        
        if err < 1e-15:
            return y8, current_dt
        
        new_dt = 0.9 * current_dt * (tol / max(err, 1e-15))**0.2
        new_dt = max(new_dt, current_dt * 0.1)
        new_dt = min(new_dt, current_dt * 10.0)
        current_dt = new_dt
    
    return y8, current_dt


def propagate_numerical(elements, t_start, t_end, dt=10.0,
                       method='rk4', perturbations=None,
                       events=None, return_all=False):
    if perturbations is None:
        perturbations = Perturbations(use_j2=False, use_drag=False, use_srp=False)
    
    r0, v0 = kepler_to_rv(elements)
    state = np.concatenate([r0, v0])
    
    times = [t_start]
    states = [state.copy()]
    
    t = t_start
    current_dt = dt
    
    while t < t_end:
        step_dt = min(current_dt, t_end - t)
        
        if method == 'rk4':
            state = rk4_step(state, step_dt, perturbations)
            actual_dt = step_dt
        elif method == 'rk78':
            state, actual_dt = rk78_step(state, step_dt, perturbations)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        t += actual_dt
        times.append(t)
        states.append(state.copy())
        
        if method == 'rk78':
            current_dt = actual_dt * 1.2
            current_dt = min(current_dt, dt * 5)
            current_dt = max(current_dt, dt * 0.1)
        
        if events is not None:
            for event in events:
                if event.check(state, t):
                    event.record(state, t)
    
    if return_all:
        return np.array(times), np.array(states)
    else:
        final_state = states[-1]
        r_final, v_final = final_state[:3], final_state[3:]
        return rv_to_kepler(r_final, v_final, units=elements.units)


class EventDetector:
    def __init__(self, event_func, direction=0):
        self.event_func = event_func
        self.direction = direction
        self.events = []
        self._prev_value = None
    
    def check(self, state, t):
        value = self.event_func(state, t)
        
        if self._prev_value is not None:
            if self.direction == 0:
                if self._prev_value * value <= 0:
                    return True
            elif self.direction > 0:
                if self._prev_value < 0 and value >= 0:
                    return True
            else:
                if self._prev_value > 0 and value <= 0:
                    return True
        
        self._prev_value = value
        return False
    
    def record(self, state, t):
        self.events.append((t, state.copy()))


def altitude_event(threshold=0):
    def func(state, t):
        r = state[:3]
        return np.linalg.norm(r) - (R_EARTH + threshold)
    return EventDetector(func, direction=-1)


def perigee_event():
    def func(state, t):
        r, v = state[:3], state[3:]
        return np.dot(r, v)
    return EventDetector(func, direction=-1)


def apogee_event():
    def func(state, t):
        r, v = state[:3], state[3:]
        return np.dot(r, v)
    return EventDetector(func, direction=1)


def j2_precession_rates(elements):
    a = elements.a_km
    e = elements.e
    i = elements.i_rad
    n = elements.get_mean_motion()
    p = a * (1 - e**2)
    
    raan_rate = -1.5 * n * J2_EARTH * (R_EARTH / p)**2 * np.cos(i)
    argp_rate = 0.75 * n * J2_EARTH * (R_EARTH / p)**2 * (5 * np.cos(i)**2 - 1)
    mean_anomaly_rate = 0.75 * n * J2_EARTH * (R_EARTH / p)**2 * np.sqrt(1 - e**2) * (3 * np.cos(i)**2 - 1)
    
    return {
        'raan_deg_per_day': np.rad2deg(raan_rate) * 86400,
        'argp_deg_per_day': np.rad2deg(argp_rate) * 86400,
        'mean_anomaly_deg_per_day': np.rad2deg(mean_anomaly_rate) * 86400
    }


def sun_synchronous_inclination(altitude, e=0):
    a = R_EARTH + altitude
    n = np.sqrt(MU_EARTH / a**3)
    p = a * (1 - e**2)
    
    target_raan_rate = 2 * np.pi / 365.25 / 86400
    
    def equation(i):
        raan_rate = -1.5 * n * J2_EARTH * (R_EARTH / p)**2 * np.cos(i)
        return raan_rate - target_raan_rate
    
    from scipy.optimize import brentq
    inc = brentq(equation, 0.1, np.pi - 0.1)
    
    return np.rad2deg(inc)
