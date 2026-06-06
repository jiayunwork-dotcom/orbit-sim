import numpy as np
from scipy.optimize import fsolve

MU_EARTH = 398600.4418
R_EARTH = 6378.137
J2_EARTH = 1.082635854e-3
OMEGA_EARTH = 7.2921159e-5


class Units:
    @staticmethod
    def km_to_re(km):
        return km / R_EARTH
    
    @staticmethod
    def re_to_km(re):
        return re * R_EARTH
    
    @staticmethod
    def deg_to_rad(deg):
        return np.deg2rad(deg)
    
    @staticmethod
    def rad_to_deg(rad):
        return np.rad2deg(rad)


class KeplerElements:
    def __init__(self, a, e, i, raan, argp, nu, units='km_deg'):
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.argp = argp
        self.nu = nu
        self.units = units
        self._to_si()
    
    def _to_si(self):
        if 'km' in self.units:
            self.a_km = self.a
            self.a = self.a
        elif 're' in self.units:
            self.a_km = Units.re_to_km(self.a)
            self.a = self.a_km
        
        if 'deg' in self.units:
            self.i_rad = Units.deg_to_rad(self.i)
            self.raan_rad = Units.deg_to_rad(self.raan)
            self.argp_rad = Units.deg_to_rad(self.argp)
            self.nu_rad = Units.deg_to_rad(self.nu)
        else:
            self.i_rad = self.i
            self.raan_rad = self.raan
            self.argp_rad = self.argp
            self.nu_rad = self.nu
    
    def get_rp(self):
        return self.a_km * (1 - self.e)
    
    def get_ra(self):
        return self.a_km * (1 + self.e)
    
    def get_hp(self):
        return self.get_rp() - R_EARTH
    
    def get_ha(self):
        return self.get_ra() - R_EARTH
    
    def get_period(self):
        return 2 * np.pi * np.sqrt(self.a_km**3 / MU_EARTH)
    
    def get_semimajor_axis(self):
        return self.a_km
    
    def get_eccentricity(self):
        return self.e
    
    def get_inclination(self):
        return self.i
    
    def get_raan(self):
        return self.raan
    
    def get_argp(self):
        return self.argp
    
    def get_true_anomaly(self):
        return self.nu
    
    def get_angular_momentum(self):
        return np.sqrt(MU_EARTH * self.a_km * (1 - self.e**2))
    
    def get_specific_energy(self):
        return -MU_EARTH / (2 * self.a_km)
    
    def get_mean_motion(self):
        return np.sqrt(MU_EARTH / self.a_km**3)
    
    def get_eccentric_anomaly(self):
        nu = self.nu_rad
        e = self.e
        cos_E = (e + np.cos(nu)) / (1 + e * np.cos(nu))
        sin_E = (np.sqrt(1 - e**2) * np.sin(nu)) / (1 + e * np.cos(nu))
        return np.arctan2(sin_E, cos_E)
    
    def get_mean_anomaly(self):
        E = self.get_eccentric_anomaly()
        return E - self.e * np.sin(E)
    
    def get_velocity_at_nu(self, nu=None):
        if nu is None:
            nu = self.nu_rad
        else:
            nu = Units.deg_to_rad(nu) if 'deg' in self.units else nu
        r = self.a_km * (1 - self.e**2) / (1 + self.e * np.cos(nu))
        return vis_viva(self.a_km, r)
    
    def __repr__(self):
        return (f"KeplerElements(a={self.a_km:.2f} km, e={self.e:.4f}, i={self.i:.2f}°, "
                f"RAAN={self.raan:.2f}°, argp={self.argp:.2f}°, nu={self.nu:.2f}°)")


def vis_viva(a, r):
    return np.sqrt(MU_EARTH * (2 / r - 1 / a))


def kepler_equation_solver(M, e, tol=1e-12, max_iter=100):
    if e < 0.8:
        E = M
    else:
        E = np.pi
    
    for _ in range(max_iter):
        dE = (M - (E - e * np.sin(E))) / (1 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def true_to_eccentric(nu, e):
    nu = np.asarray(nu)
    return 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))


def eccentric_to_true(E, e):
    E = np.asarray(E)
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))


def mean_to_true(M, e):
    E = kepler_equation_solver(M, e)
    return eccentric_to_true(E, e)


def true_to_mean(nu, e):
    E = true_to_eccentric(nu, e)
    return E - e * np.sin(E)


def kepler_to_rv(elements):
    a = elements.a_km
    e = elements.e
    i = elements.i_rad
    raan = elements.raan_rad
    argp = elements.argp_rad
    nu = elements.nu_rad
    
    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))
    
    r_pqw = np.array([
        r_mag * np.cos(nu),
        r_mag * np.sin(nu),
        0.0
    ])
    
    v_pqw = np.array([
        -np.sqrt(MU_EARTH / p) * np.sin(nu),
        np.sqrt(MU_EARTH / p) * (e + np.cos(nu)),
        0.0
    ])
    
    R3_raan = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan), np.cos(raan), 0],
        [0, 0, 1]
    ])
    
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R3_argp = np.array([
        [np.cos(argp), -np.sin(argp), 0],
        [np.sin(argp), np.cos(argp), 0],
        [0, 0, 1]
    ])
    
    Q = R3_raan @ R1_i @ R3_argp
    
    r_eci = Q @ r_pqw
    v_eci = Q @ v_pqw
    
    return r_eci, v_eci


def rv_to_kepler(r, v, units='km_deg'):
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    
    n = np.cross([0, 0, 1], h)
    n_mag = np.linalg.norm(n)
    
    e_vec = ((v_mag**2 - MU_EARTH / r_mag) * r - np.dot(r, v) * v) / MU_EARTH
    e = np.linalg.norm(e_vec)
    
    energy = v_mag**2 / 2 - MU_EARTH / r_mag
    
    if abs(e - 1.0) > 1e-10:
        a = -MU_EARTH / (2 * energy)
    else:
        a = np.inf
    
    i = np.arccos(h[2] / h_mag)
    
    if n_mag != 0:
        raan = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0
    
    if n_mag != 0 and e > 1e-10:
        argp = np.arccos(np.dot(n, e_vec) / (n_mag * e))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0
    
    if e > 1e-10:
        nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        if n_mag != 0:
            nu = np.arccos(np.dot(n, r) / (n_mag * r_mag))
            if r[2] < 0:
                nu = 2 * np.pi - nu
        else:
            nu = np.arctan2(r[1], r[0])
    
    if 'deg' in units:
        i_deg = Units.rad_to_deg(i)
        raan_deg = Units.rad_to_deg(raan)
        argp_deg = Units.rad_to_deg(argp)
        nu_deg = Units.rad_to_deg(nu)
        return KeplerElements(a, e, i_deg, raan_deg, argp_deg, nu_deg, units=units)
    else:
        return KeplerElements(a, e, i, raan, argp, nu, units=units)


def propagate_orbit(elements, t0, t):
    n = elements.get_mean_motion()
    M0 = elements.get_mean_anomaly()
    dt = t - t0
    M = M0 + n * dt
    nu = mean_to_true(M, elements.e)
    
    new_elements = KeplerElements(
        elements.a_km, elements.e, elements.i,
        elements.raan, elements.argp,
        Units.rad_to_deg(nu) if 'deg' in elements.units else nu,
        units=elements.units
    )
    return new_elements


def get_orbit_points(elements, num_points=200):
    nus = np.linspace(0, 2 * np.pi, num_points)
    points = []
    for nu in nus:
        el = KeplerElements(
            elements.a_km, elements.e, elements.i,
            elements.raan, elements.argp,
            Units.rad_to_deg(nu),
            units='km_deg'
        )
        r, _ = kepler_to_rv(el)
        points.append(r)
    return np.array(points)


def get_ground_track(elements, t0, duration, num_points=500):
    times = np.linspace(t0, t0 + duration, num_points)
    lons = []
    lats = []
    
    for t in times:
        el = propagate_orbit(elements, t0, t)
        r, _ = kepler_to_rv(el)
        
        lon = np.arctan2(r[1], r[0]) - OMEGA_EARTH * (t - t0)
        lon = np.mod(lon + np.pi, 2 * np.pi) - np.pi
        
        lat = np.arcsin(r[2] / np.linalg.norm(r))
        
        lons.append(Units.rad_to_deg(lon))
        lats.append(Units.rad_to_deg(lat))
    
    return np.array(lons), np.array(lats)
