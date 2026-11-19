import astropy.constants as const
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from galpy.orbit import Orbit
from galpy.potential import (
    EinastoPotential,
    PlummerPotential,
    SCFPotential,
    TruncatedHernquistPotential,
    TruncatedNFWPotential,
    evaluateDensities,
    evaluatePotentials,
    interpSphericalPotential,
    scf_compute_coeffs_nbody,
    vcirc,
)
from galpy.util import conversion
from scipy.optimize import curve_fit


def spherical_potential_plotter(file, plot=True, output=None):
    """
    Builds a spherical potential from the DM particles in 'file' and plots the potential and density

    Parameters:

    file : .hdf5 file
            Data from the  TNG simulation (one halo)

    plot_potential : bool
                     Boolean that determines if the potential should be plotted

    plot_density : bool
                   Boolean that determines if the density should be plotted

    Returns:

    pot : galpy.potential
          The potential constructed from the data in 'file'

    rplot : astropy Quantity
            Radii (kpc) at which the density was evaluated

    rho : astropy Quantity
          Density (Msun/kpc^3) at those radii
    """
    # galpy's unit scales
    ro, vo = 8.0 * u.kpc, 220.0 * u.km / u.s

    # Open file, get particle positions and masses
    with h5py.File(file, "r") as f:
        hdr = dict(f["Header"].attrs)
        pos = f["PartType1/Coordinates"][:]
        m = np.full(len(pos), hdr["MassTable"][1])   # all DM particles have equal mass in TNG

    # Get hubble param and divide by it, convention in TNG
    h = hdr.get("HubbleParam", 1.0)
    L_unit = hdr.get("UnitLength_in_cm", 3.085678e21) * u.cm / h
    M_unit = hdr.get("UnitMass_in_g", 1.989e43) * u.g / h

    # Convert data to more conventional units
    pos = pos * L_unit.to(u.kpc)
    m = m * M_unit.to(u.Msun)

    # Calculate the center of all particles
    center = np.average(pos.value, axis=0, weights=m.value) * u.kpc
    # Calculate all radii w.r.t. center and sort them in increasing r
    r = np.linalg.norm((pos - center).value, axis=1) * u.kpc
    order = np.argsort(r)
    # The mass is the total enclosed mass
    r, M = r[order], np.cumsum(m[order])

    # Make a grid of radii, interpolate between radii in r and assign a mass to each gridpoint.
    # Only start at the 100th particle, starting from the 0th particle includes too much noise
    rgrid_phys = np.geomspace(r[100], r[-1], 200)
    Mgrid_phys = np.interp(rgrid_phys, r, M)

    # Radial force grid in physical units, with G from astropy: F = -G M / r^2  [(km/s)^2 / kpc]
    Fgrid_phys = (-const.G * Mgrid_phys / rgrid_phys**2).to(u.km**2 / u.s**2 / u.kpc)

    # Express rgrid and Fgrid in galpy's unit scales
    rgrid = (rgrid_phys / ro).decompose().value
    Fgrid = (Fgrid_phys / (vo**2 / ro)).decompose().value

    pot = interpSphericalPotential(
        rforce=lambda x: np.interp(x, rgrid, Fgrid),
        rgrid=rgrid, ro=ro.to_value(u.kpc), vo=vo.to_value(u.km / u.s))

    # Get the potentia at the grid points
    Phi = evaluatePotentials(pot, rgrid_phys, 0. * u.kpc, quantity=True)
    Phi = Phi.to(u.km**2 / u.s**2)
    # Shift so the outer edge matches a point mass, Phi = -G M_tot / r (zero at infinity)
    Phi = Phi - Phi[-1] + (-const.G * Mgrid_phys[-1] / rgrid_phys[-1]).to(u.km**2 / u.s**2)

    # Get the density at the grid points
    rho = evaluateDensities(pot, rgrid_phys, 0. * u.kpc, quantity=True)
    rho = rho.to(u.Msun / u.kpc**3)

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        axs[0].semilogx(rgrid_phys.value, Phi.value)
        axs[0].set_xlabel("r [kpc]")
        axs[0].set_ylabel(r"$\Phi$ [(km/s)$^2$]")
        axs[0].set_title("Potential")

        axs[1].loglog(rgrid_phys.value, rho.value)
        axs[1].set_xlabel("r [kpc]")
        axs[1].set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]")
        axs[1].set_title("Density profile from the galpy potential")

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    return pot, rgrid_phys, rho, Mgrid_phys


def data_fitter(file, plot=True, output=None):
    """
    Takes the data from the simulation and fits 4 models: Einasto, Plummer and truncated NFW and
    Hernquist. Returns the fitted parameters and the rms deviation of the fit.

    Parameters:

    file : .hdf5 file
            Data from the  TNG simulation (one halo)

    plot : bool
           Boolean that determines if the results should be plotted

    Returns:

    results_params : dict
            The names of the fitted density models (keys) and the corresponding fitted parameters
            with their corresponding rms values (values)

    results_density : dict
            The names of the fitted density models (keys) and the corresponding model i.e. rho(r)
            (Msun/kpc^3) over the grid (values)

    r : astropy Quantity
            Radii (kpc) at which the density was evaluated

    """

    # Extract potential and density from the data
    _, r, rho, _ = spherical_potential_plotter(file, plot=False)

    # galpy's unit scales as plain numbers (kpc, km/s), only used to construct the fit models
    ro, vo = 8.0, 220.0

    # units to express results in, dimensionless unit needed for Einasto
    one = u.dimensionless_unscaled
    rho_unit = u.Msun / u.kpc**3

    # Density models: galpy potentials evaluated through their analytic density
    def plummer(r, amp, b):
        p = PlummerPotential(amp=amp, b=b, ro=ro, vo=vo)
        return evaluateDensities(p, r, 0. * u.kpc, quantity=True)

    def einasto(r, amp, rm2, n):
        p = EinastoPotential(amp=amp, rm2=rm2, n=float(n), ro=ro, vo=vo)
        return evaluateDensities(p, r, 0. * u.kpc, quantity=True)

    def nfw_trunc(r, amp, a, rc):
        p = TruncatedNFWPotential(amp=amp, a=a, rc=rc, ro=ro, vo=vo)
        return evaluateDensities(p, r, 0. * u.kpc, quantity=True)

    def hernquist_trunc(r, amp, a, rc):
        p = TruncatedHernquistPotential(amp=amp, a=a, rc=rc, ro=ro, vo=vo)
        return evaluateDensities(p, r, 0. * u.kpc, quantity=True)

    # Drop non-positive densities (the derivative of the interpolated force can be noisy),
    # since we fit in log space
    good = rho > 0 * rho_unit
    r_fit, rho_fit = r[good], rho[good]

    # A function to perform a single fit, note that scipy doesn't take astropy units so the
    # variables of this function shouldn't have units (andd won't have them if scipy uses the
    # function)
    def fit(model, p0, units):
        """
        Fit in log space. This fits log(rho) as a function of r
        Parameters are log10 of their value in 'units', so they stay positive.
        """
        # This is the functin that's fitted, takes a model and takes log() of it.
        def log_model(r_val, *logp):
            # Reattach units to the parameters, what scipy puts into this function won't have units
            params = [10**lp * un for lp, un in zip(logp, units)]
            # Evaluate the log(model), the galpy model *does* need units so they're reattached
            return np.log10(model(r_val * u.kpc, *params).to_value(rho_unit))

        # Apply the scipy curve fit on the function that doesn't need units itself, returns
        # (fitted parameters, covariance). Note that .to_value is used because scipy doesn't take
        # units!
        popt, _ = curve_fit(
            log_model,
            r_fit.to_value(u.kpc),
            np.log10(rho_fit.to_value(rho_unit)),
            p0=np.log10([p.to_value(un) for p, un in zip(p0, units)]),
            maxfev=10000
        )
        # Return everything, undo the logarithm, reattach units
        return [10**p * un for p, un in zip(popt, units)]

    # Starting guesses from the data: scale radius a0 and density rho0 at the middle of the grid
    a0 = r_fit[len(r_fit) // 2]
    rho0 = rho_fit[len(r_fit) // 2]
    rc0 = r_fit[-1] / 2

    models = {
        # Plummer: rho(b) = 3 amp / (4 pi b^3) * 2^(-5/2)
        "Plummer":             (plummer,         [rho0 * 4 * np.pi * a0**3 / 3 * 2**2.5, a0],  [u.Msun, u.kpc]),
        # Einasto: rho(rm2) = amp, so amp ~ rho0 at rm2 ~ a0
        "Einasto":             (einasto,         [rho0, a0, 5 * one],                       [rho_unit, u.kpc, one]),
        # NFW: rho(a) = amp / (4 pi a^3) / 4
        "Truncated NFW":       (nfw_trunc,       [4 * rho0 * 4 * np.pi * a0**3, a0, rc0],   [u.Msun, u.kpc, u.kpc]),
        # Hernquist: rho(a) = amp / (4 pi a^3) / 8
        "Truncated Hernquist": (hernquist_trunc, [8 * rho0 * 4 * np.pi * a0**3, a0, rc0],   [u.Msun, u.kpc, u.kpc]),
    }

    # Fit every model first, keep the results
    results_density = {}
    results_params = {}
    for name, (model, p0, units) in models.items():
        params = fit(model, p0, units)
        rho_model = model(r, *params).to_value(rho_unit)

        # rms residual in log10 space, over the points used in the fit
        resid = (np.log10(model(r_fit, *params).to_value(rho_unit))
                 - np.log10(rho_fit.to_value(rho_unit)))
        rms = np.sqrt(np.mean(resid**2))

        results_params[name] = (params, rms)
        results_density[name] = (rho_model, rms)

    if plot:
        # Layout: five panels in one row (overlay first, then each fit), all sharing the same x and y axes
        fig, ax = plt.subplots(1, figsize=(6, 5))

        # Same axis limits for every panel, set from the data
        ymin = rho_fit.to_value(rho_unit).min() / 3
        ymax = rho_fit.to_value(rho_unit).max() * 3

        # All fits overlaid
        ax.loglog(r.value, rho.value, "k", lw=2, label="galpy density")
        for name, (rho_model, rms) in results_density.items():
            ax.loglog(r.value, rho_model, "--", label=name)
        ax.set(xlabel="r [kpc]", ylabel=r"$\rho$ [M$_\odot$ kpc$^{-3}$]", title="Density fits")
        ax.set_ylim(ymin, ymax)
        ax.legend()

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    return results_params, results_density, r


def build_Einasto(file):
    """
    Takes the data from the simulation and constructs the Einasto model with the fitted parameters
    supplied by "data_fitter"

    Parameters:

    file : .hdf5 file
            Data from the  TNG simulation (one halo)

    Returns:

    pot_einasto : galpy.potential
            The galpy Einasto potential made with the fitted parameters returned by "data_fitter"

    rho_einasto : astropy Quantity
            The Einasto density profile (Msun/kpc^3) corresponding to the constructed Einasto
            potential

    r : astropy Quantity
            Radii (kpc) at which the density was evaluated

    """
    params, _, r = data_fitter(file, plot=False)

    # galpy's unit scales (kpc, km/s)
    ro, vo = 8.0, 220.0

    # Fitted Einasto parameters: amp (Msun/kpc^3), rm2 (kpc), n (dimensionless)
    (amp_e, rm2_e, n_e), _ = params["Einasto"]

    # Build the galpy potential from the fitted parameters
    pot_einasto = EinastoPotential(amp=amp_e, rm2=rm2_e, n=n_e.to_value(u.dimensionless_unscaled),
                                ro=ro, vo=vo)
    # Calculate the density from the potential
    rho_einasto = evaluateDensities(pot_einasto, r, 0. * u.kpc, quantity=True).to(u.Msun / u.kpc**3)

    return pot_einasto, rho_einasto, r


def circular_velocity(pot, r, plot=False):
    """
    Computes the circular velocity v_c(r) = sqrt(r |F_r|) = sqrt(G M(<r) / r) of a spherical galpy
    potential and optionally plots it as a function of r.

    Parameters:

    pot : galpy.potential
          The spherical potential: pot_einasto from build_Einasto

    r : astropy Quantity
        Radii (kpc) at which to evaluate v_c: the grid returned by build_Einasto

    plot : bool
           Whether to plot v_c as a function of r

    Returns:

    vc : astropy Quantity
         The circular velocity (km/s) at the radii in 'r'

    """
    # Get the circular velocity using galpy
    vc = vcirc(pot, r).to(u.km / u.s)

    # Standard plotting, vc as a function of radius
    if plot:
        plt.semilogx(r.to_value(u.kpc), vc.value)
        plt.xlabel("r [kpc]")
        plt.ylabel(r"$v_c$ [km/s]")
        plt.title("Circular velocity")
        plt.show()

    return vc


def integrate_orbits(pot, r0, f_vc, n_orbits=5, n_steps=1000, ro=8.0, vo=220.0, plot=True, output=None):
    """
    Integrates orbits in a spherical galpy potential and optionally plots them. Every orbit starts
    at radius r0 in the z = 0 plane with a purely tangential velocity f * v_c(r0).

    Parameters:

    pot : galpy.potential
          The spherical potential, e.g. pot_einasto from build_Einasto

    r0 : astropy Quantity
         Starting radius (kpc), should be inside the range covered by the particles.

    f_vc : sequence of float
           Tangential velocity of each orbit as a fraction of the circular velocity at r0.
           1.0 is a circular orbit. Smaller values give eccentric orbits for which r0 is the
           apocenter.

    n_orbits : float
               Integration time in units of the circular orbital period at r0

    n_steps : int
              Number of time steps

    ro, vo : float
             galpy's unit scales (kpc, km/s), must be the same as used for the potential

    plot : bool
           Whether to plot the orbits

    Returns:

    orbits : list of galpy.orbit.Orbit
             The integrated orbits, one for each value in f_vc

    ts : list of astropy Quantity
         The times (Gyr) at which each orbit was integrated

    """
    # Get the velocity at the starting point r0
    vc0 = vcirc(pot, r0).to(u.km / u.s)

    # Integrate for n_orbits circular periods at r0, standard formula for period in spherically
    # symmetric potential
    T = n_orbits * (2 * np.pi * r0 / vc0).to(u.Gyr)
    ts = np.linspace(0, T.to_value(), n_steps) * u.Gyr

    orbits = []
    for f in f_vc:
        # [R, vR, vT, z, vz, phi]: purely tangential velocity, in the z = 0 plane. galpy expects
        # the coordinates to be given in this order
        o = Orbit([r0, 0. * u.km / u.s, f * vc0, 0. * u.kpc, 0. * u.km / u.s, 0. * u.rad],
                  ro=ro, vo=vo)
        # Integrate orbit and append to the collection of orbits
        o.integrate(ts, pot)
        orbits.append(o)

    # Standard plotting, note that you need "normal" values for matplotlib to do its thing. We make
    # sure everything is in kpc and Gyr
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        for f, o in zip(f_vc, orbits):
            x = o.x(ts).to_value(u.kpc)
            y = o.y(ts).to_value(u.kpc)
            r = o.r(ts).to_value(u.kpc)

            axs[0].plot(x, y, label=rf"$v_T = {f:.2f}\,v_c$")
            axs[1].plot(ts.to_value(u.Gyr), r, label=rf"$v_T = {f:.2f}\,v_c$")

        axs[0].set(xlabel="x [kpc]", ylabel="y [kpc]", title="Orbits in the orbital plane", aspect="equal")
        axs[0].legend(loc="lower left", frameon=True)
        axs[1].set(xlabel="t [Gyr]", ylabel="r [kpc]", title="Radius as a function of time")
        axs[1].legend(loc="lower left", frameon=True)

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    return orbits, [ts] * len(orbits)


def velocity_dispersions(file, n_bins=20, plot=True, return_counts=False, output=None):
    """
    Measures the radial and tangential velocity dispersion of the DM particles as a function of
    radius, in logarithmic spherical shells around the center of the halo.

    The tangential dispersion is defined per component: sigma_t^2 = (sigma_theta^2 + sigma_phi^2) / 2,
    so that an isotropic halo has sigma_t = sigma_r.

    Parameters:

    file : .hdf5 file
           Data from the TNG simulation (one halo)

    n_bins : int
             Number of logarithmic radial shells

    plot : bool
           Whether to plot the dispersions as a function of r

    Returns:

    r_mid : astropy Quantity
            Geometric center of each shell (kpc)

    sigma_r : astropy Quantity
              Radial velocity dispersion (km/s) in each shell

    sigma_t : astropy Quantity
              Tangential velocity dispersion (km/s) per component, in each shell

    """
    with h5py.File(file, "r") as f:
        hdr = dict(f["Header"].attrs)
        pos = f["PartType1/Coordinates"][:]
        vel = f["PartType1/Velocities"][:]

    # Same unit conventions as before. TNG velocities are stored as v * sqrt(a), with a the scale
    # factor (a = 1 at z = 0)
    h = hdr.get("HubbleParam", 1.0)
    a = hdr.get("Time", 1.0)
    L_unit = hdr.get("UnitLength_in_cm", 3.085678e21) * u.cm / h
    v_unit = hdr.get("UnitVelocity_in_cm_per_s", 1e5) * u.cm / u.s

    pos = pos * L_unit.to(u.kpc)
    vel = vel * np.sqrt(a) * v_unit.to(u.km / u.s)

    # Positions relative to the center, velocities relative to the bulk velocity of the halo
    # (all particles have the same mass)
    x = (pos - pos.mean(axis=0)).value
    v = (vel - vel.mean(axis=0)).value

    r = np.linalg.norm(x, axis=1)
    theta = np.arccos(x[:, 2] / r)
    phi = np.arctan2(x[:, 1], x[:, 0])

    # Spherical unit vectors at the position of each particle
    e_r = x / r[:, None]
    e_th = np.column_stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_ph = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])

    # Velocity components in spherical coordinates
    v_r = np.sum(v * e_r, axis=1)
    v_th = np.sum(v * e_th, axis=1)
    v_ph = np.sum(v * e_ph, axis=1)

    # Logarithmic shells, from the 100th particle (as before, inside that it is noise) to the outermost
    edges = np.geomspace(np.sort(r)[100], r.max(), n_bins + 1)
    shell = np.digitize(r, edges) - 1

    counts = np.zeros(n_bins, dtype=int)
    sigma_r = np.full(n_bins, np.nan)
    sigma_t = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = shell == i
        counts[i] = sel.sum()
        if sel.sum() < 2:
            continue
        # The standard deviation within a shell removes any mean streaming motion of the shell
        sigma_r[i] = v_r[sel].std()
        sigma_t[i] = np.sqrt((v_th[sel].var() + v_ph[sel].var()) / 2)

    r_mid = np.sqrt(edges[1:] * edges[:-1]) * u.kpc
    sigma_r = sigma_r * u.km / u.s
    sigma_t = sigma_t * u.km / u.s

    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.semilogx(r_mid.to_value(u.kpc), sigma_r.to_value(u.km / u.s), "o-", label=r"$\sigma_r$")
        ax.semilogx(r_mid.to_value(u.kpc), sigma_t.to_value(u.km / u.s), "s-", label=r"$\sigma_t$")
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("velocity dispersion [km/s]")
        ax.set_title("Radial and tangential velocity dispersion")
        ax.legend()

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    if return_counts:
        return r_mid, sigma_r, sigma_t, edges * u.kpc, counts
    return r_mid, sigma_r, sigma_t


def tangential_velocity(file, n_bins=15, r_range=None):
    """
    Measures the full tangential velocity v_t of the DM particles as a function of radius, in
    logarithmic spherical shells. v_t is the speed perpendicular to the radial direction,
    v_t = sqrt(v_theta^2 + v_phi^2) = sqrt(|v|^2 - v_r^2).

    Parameters:

    file : .hdf5 file
           Data from the TNG simulation (one halo)

    n_bins : int
             Number of logarithmic radial shells

    Returns:

    r_mid : astropy Quantity
            Geometric center of each shell (kpc)

    vt_mean : astropy Quantity
              Mean tangential speed v_t in each shell (km/s)

    vt_rms : astropy Quantity
             Root-mean-square tangential speed in each shell (km/s), sqrt(<v_t^2>)

    """
    with h5py.File(file, "r") as f:
        hdr = dict(f["Header"].attrs)
        pos = f["PartType1/Coordinates"][:]
        vel = f["PartType1/Velocities"][:]

    # Same unit conventions as before. TNG velocities are stored as v * sqrt(a), with a the scale
    # factor (a = 1 at z = 0)
    h = hdr.get("HubbleParam", 1.0)
    a = hdr.get("Time", 1.0)
    L_unit = hdr.get("UnitLength_in_cm", 3.085678e21) * u.cm / h
    v_unit = hdr.get("UnitVelocity_in_cm_per_s", 1e5) * u.cm / u.s

    pos = pos * L_unit.to(u.kpc)
    vel = vel * np.sqrt(a) * v_unit.to(u.km / u.s)

    # Positions relative to the center, velocities relative to the bulk velocity of the halo
    # (all particles have the same mass)
    x = (pos - pos.mean(axis=0)).value
    v = (vel - vel.mean(axis=0)).value
    r = np.linalg.norm(x, axis=1)

    # Radial velocity: the projection of the velocity onto the radial direction
    e_r = x / r[:, None]
    v_r = np.sum(v * e_r, axis=1)

    # What remains of the speed after removing the radial part is perpendicular to the radius
    # (the maximum guards against tiny negative values from rounding errors)
    v_t = np.sqrt(np.maximum(np.sum(v**2, axis=1) - v_r**2, 0.))

    # Logarithmic shells, from the 100th particle (as before, inside that it is noise) to the outermost
    # if r_range specified: from r_range[0] to r_range[1]
    if r_range is None:
        edges = np.geomspace(np.sort(r)[100], r.max(), n_bins + 1)
    else:
        edges = np.geomspace(r_range[0].to_value(u.kpc), r_range[1].to_value(u.kpc), n_bins + 1)
    shell = np.digitize(r, edges) - 1

    vt_mean = np.full(n_bins, np.nan)
    vt_rms = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = shell == i
        if sel.sum() < 2:
            continue
        vt_mean[i] = v_t[sel].mean()
        vt_rms[i] = np.sqrt(np.mean(v_t[sel]**2))

    r_mid = np.sqrt(edges[1:] * edges[:-1]) * u.kpc
    return r_mid, vt_mean * u.km / u.s, vt_rms * u.km / u.s


def compare_rotation_curves(file, n_bins=15, plot=True, output=None):
    """
    Plots the circular velocity v_c(r) of the halo from the Einasto model and from the enclosed mass,
    together with the tangential velocity v_t of the particles.

    1) directly from the fitted Einasto potential
    2) from the enclosed mass of the particles, v_c = sqrt(G M(<r) / r)
    3) the tangential velocity v_t of the particles (the speed perpendicular to the radial direction)

    Parameters:

    file : .hdf5 file
           Data from the TNG simulation (one halo)

    n_bins : int
             Number of logarithmic radial shells for the particle velocities

    plot : bool
           Whether to plot the curves

    Returns:

    (r, vc_model) : radii and circular velocity from the Einasto model
    (r_M, vc_mass) : radii and circular velocity from the enclosed mass
    (r_t, vt_mean, vt_rms) : shell centers, mean v_t and rms v_t of the particles
    all as astropy Quantities (kpc and km/s)

    """
    # 1) Directly from the model
    pot_einasto, _, r = build_Einasto(file)
    vc_model = circular_velocity(pot_einasto, r, plot=False)

    # 2) From the enclosed mass: v_c = sqrt(G M(<r) / r). The enclosed mass is the fourth output of
    # potential_plotter, on the same radius grid as the model
    _, r_M, _, M = spherical_potential_plotter(file, plot=False)
    vc_mass = np.sqrt(const.G * M / r_M).to(u.km / u.s)

    # 3) The tangential velocities of the particles
    r_t, vt_mean, vt_rms = tangential_velocity(file, n_bins=n_bins)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.semilogx(r.to_value(u.kpc), vc_model.to_value(u.km / u.s), label="fit - Einasto")
        ax.semilogx(r_M.to_value(u.kpc), vc_mass.to_value(u.km / u.s), "--",
                     label=r"data - enclosed mass")
        ax.semilogx(r_t.to_value(u.kpc), vt_mean.to_value(u.km / u.s), "o-",
                     label=r"data - mean $|v_t|$")
        ax.semilogx(r_t.to_value(u.kpc), vt_rms.to_value(u.km / u.s), "s-",
                     label=r"data - rms $v_t$")
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("velocity [km/s]")
        ax.set_title("Rotation curve")
        ax.legend()

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    return (r, vc_model), (r_M, vc_mass), (r_t, vt_mean, vt_rms)


# ------------ ASYMMETRIC POTENTIALS ----------------

def build_scf(file, N=10, L=4, n_r=40, n_dir=200, n_shells=20, plot=True, output=None):
    """
    Builds a galpy SCF (basis-function expansion) potential directly from the DM particles,
    with no spherical symmetry assumption, and optionally plots how spherical it is.

    N : number of radial terms
    L : number of angular terms (L = 0 is spherical)
    """
    ro, vo = 8.0, 220.0

    with h5py.File(file, "r") as f:
        hdr = dict(f["Header"].attrs)
        pos = f["PartType1/Coordinates"][:]
        m = np.full(len(pos), hdr["MassTable"][1])

    h = hdr.get("HubbleParam", 1.0)
    L_unit = hdr.get("UnitLength_in_cm", 3.085678e21) * u.cm / h
    M_unit = hdr.get("UnitMass_in_g", 1.989e43) * u.g / h

    pos = pos * L_unit.to(u.kpc)
    m = m * M_unit.to(u.Msun)

    x = pos - pos.mean(axis=0)
    r_part = np.sort(np.linalg.norm(x.value, axis=1)) # sorted particle radii in kpc
    # Scale radius of the basis: the median radius works as a starting point
    a_phys = np.median(r_part) * u.kpc

    # SCF works in galpy natural units
    x_nat = conversion.parse_length(x, ro=ro, vo=vo)
    m_nat = conversion.parse_mass(m, ro=ro, vo=vo)
    a_nat = conversion.parse_length(a_phys, ro=ro, vo=vo)

    # Expansion coefficients from the particles; pos must have shape (3, N_particles)
    Acos, Asin = scf_compute_coeffs_nbody(x_nat.T, N, L, mass=m_nat, a=a_nat)
    pot_scf = SCFPotential(Acos=Acos, Asin=Asin, a=a_nat, ro=ro, vo=vo)
    # Added this before the loop as well, now you can get it without having to plot everything
    r_ax = np.geomspace(r_part[100], r_part[-1], 60) * u.kpc

    if plot:
        pot_unit = u.km**2 / u.s**2
        rho_unit = u.Msun / u.kpc**3

        # --- Potential along the three coordinate axes ---
        tiny = 1e-6 * r_ax # galpy works in cylindrical coords, so the z axis needs R ~ 0
        zero = 0.0 * u.kpc
        Phi_x = evaluatePotentials(pot_scf, r_ax, zero, phi=0. * u.rad, quantity=True).to(pot_unit)
        Phi_y = evaluatePotentials(pot_scf, r_ax, zero, phi=90. * u.deg, quantity=True).to(pot_unit)
        Phi_z = evaluatePotentials(pot_scf, tiny, r_ax, phi=0. * u.rad, quantity=True).to(pot_unit)
        Phi_axes = {"x axis": Phi_x, "y axis": Phi_y, "z axis": Phi_z}
        Phi_mean = (Phi_x + Phi_y + Phi_z) / 3

        # Spherical averages: many directions at each radius
        r_grid = np.geomspace(r_part[100], r_part[-1], n_r)

        # Directions spread evenly over the sphere (Fibonacci spiral)
        i = np.arange(n_dir) + 0.5
        cos_t = 1 - 2 * i / n_dir
        sin_t = np.sqrt(1 - cos_t**2)
        phi_dir = np.pi * (1 + np.sqrt(5)) * i

        # Cylindrical coordinates of every (radius, direction) pair, shape (n_r, n_dir)
        R = (r_grid[:, None] * sin_t[None, :]).ravel() * u.kpc
        z = (r_grid[:, None] * cos_t[None, :]).ravel() * u.kpc
        phi = np.tile(phi_dir, n_r) * u.rad

        Phi_all = evaluatePotentials(pot_scf, R, z, phi=phi, quantity=True).to_value(pot_unit).reshape(n_r, n_dir)
        rho_all = evaluateDensities(pot_scf, R, z, phi=phi, quantity=True).to_value(rho_unit).reshape(n_r, n_dir)

        Phi_spherical = Phi_all.mean(axis=1) * pot_unit
        rho_spherical = rho_all.mean(axis=1) * rho_unit

        # Density measured directly from the particles, in logarithmic shells
        edges = np.geomspace(r_part[100], r_part[-1], n_shells + 1)
        counts, _ = np.histogram(r_part, bins=edges)
        m_part = (hdr["MassTable"][1] * M_unit.to(u.Msun)).value
        rho_part = counts * m_part / (4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3))
        r_shell = np.sqrt(edges[1:] * edges[:-1])

        # ---------- PLOTTING ------------
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        # 1. POTENTIAL ALONG AXES
        for name, Phi in Phi_axes.items():
            ax[0, 0].semilogx(r_ax.value, Phi.value, label=name)
        ax[0, 0].set(xlabel="r [kpc]", ylabel=r"$\Phi$ [(km/s)$^2$]", title="Potential along the axes")
        ax[0, 0].legend()

        # 2. POTENTIAL DIFFERENCE WITH MEAN ALONG AXES
        for name, Phi in Phi_axes.items():
            ax[0, 1].semilogx(r_ax.value, ((Phi - Phi_mean) / np.abs(Phi_mean)).decompose().value * 100,
                label=name)
        ax[0, 1].axhline(0, color="k", lw=0.5)
        ax[0, 1].set(xlabel="r [kpc]", ylabel="deviation from the mean [%]",
                     title="Deviation between axes")
        ax[0, 1].legend()

        # 3. POTENTIAL, SPHERICAL AVERAGE
        ax[1, 0].semilogx(r_grid, Phi_spherical.value, label="spherical average")
        lo, hi = np.percentile(Phi_all, [16, 84], axis=1)
        ax[1, 0].fill_between(r_grid, lo, hi, alpha=0.3, label="16-84 percentile")
        ax[1, 0].set(xlabel="r [kpc]", ylabel=r"$\Phi$ [(km/s)$^2$]",
                     title="Potential, spherical average")
        ax[1, 0].legend()

        # 4. DENSITY, SPHERICAL AVERAGE
        ax[1, 1].loglog(r_grid, np.where(rho_spherical.value > 0, rho_spherical.value, np.nan),
                        label="SCF - spherical average")
        lo, hi = np.percentile(rho_all, [16, 84], axis=1)
        ax[1, 1].fill_between(r_grid, np.where(lo > 0, lo, np.nan), np.where(hi > 0, hi, np.nan),
            alpha=0.3, label="SCF - 16-84 percentile")
        ax[1, 1].loglog(r_shell, rho_part, "ks", label="data - in log shells")
        ax[1, 1].set(xlabel="r [kpc]",
                     ylabel=r"$\rho$ [M$_\odot$ kpc$^{-3}$]",
                     title="Density, spherical average")
        ax[1, 1].legend()

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    return pot_scf, r_ax


def plot_potential_density_scf_2d(pot_scf, file, n_r=40, n_dir=200, plot=True, output=None):
    """
    Plots equipotential and density maps of an SCF potential in the x-y and x-z planes.

    Parameters:

    pot_scf : galpy.potential
              The SCF potential, e.g. from build_SCF

    file : .hdf5 file
           Data from the TNG simulation (one halo), the same file that was used for pot_scf

    n_r : int
          Number of radii for the spherical averages

    n_dir : int
            Number of directions, spread evenly over the sphere, to average over at each radius

    n_shells : int
               Number of logarithmic shells for the density measured from the particles

    plot : bool
           Whether to plot

    Returns:

    r_grid : astropy Quantity
             Radii (kpc) of the averaged profiles

    Phi_mean : astropy Quantity
               Potential (km^2/s^2) averaged over all directions

    rho_mean : astropy Quantity
               Density (Msun/kpc^3) averaged over all directions
    """
    pot_unit = u.km**2 / u.s**2
    rho_unit = u.Msun / u.kpc**3

    with h5py.File(file, "r") as f:
        hdr = dict(f["Header"].attrs)
        pos = f["PartType1/Coordinates"][:]
    h = hdr.get("HubbleParam", 1.0)
    L_unit = hdr.get("UnitLength_in_cm", 3.085678e21) * u.cm / h

    pos = pos * L_unit.to(u.kpc)
    x = (pos - pos.mean(axis=0)).value
    r_part = np.sort(np.linalg.norm(x, axis=1))
    r_grid = np.geomspace(r_part[100], r_part[-1], n_r)

    i = np.arange(n_dir) + 0.5
    cos_t = 1 - 2 * i / n_dir
    sin_t = np.sqrt(1 - cos_t**2)
    phi_dir = np.pi * (1 + np.sqrt(5)) * i

    R = (r_grid[:, None] * sin_t[None, :]).ravel() * u.kpc
    z = (r_grid[:, None] * cos_t[None, :]).ravel() * u.kpc
    phi = np.tile(phi_dir, n_r) * u.rad

    Phi_all = evaluatePotentials(pot_scf, R, z, phi=phi, quantity=True).to_value(pot_unit).reshape(n_r, n_dir)
    rho_all = evaluateDensities(pot_scf, R, z, phi=phi, quantity=True).to_value(rho_unit).reshape(n_r, n_dir)

    Phi_mean = Phi_all.mean(axis=1) * pot_unit
    rho_mean = rho_all.mean(axis=1) * rho_unit

    if plot:
        lim = np.percentile(r_part, 90)
        g = np.linspace(-lim, lim, 60)
        X, Y = np.meshgrid(g, g)

        R_xy = np.hypot(X, Y).ravel() * u.kpc
        z_xy = np.zeros(X.size) * u.kpc
        phi_xy = np.arctan2(Y, X).ravel() * u.rad

        R_xz = np.abs(X).ravel() * u.kpc
        z_xz = Y.ravel() * u.kpc
        phi_xz = np.where(X.ravel() >= 0, 0., np.pi) * u.rad

        maps = {}
        for name, (Rm, zm, pm) in {"x-y": (R_xy, z_xy, phi_xy), "x-z": (R_xz, z_xz, phi_xz)}.items():
            maps[name] = (
                evaluatePotentials(pot_scf, Rm, zm, phi=pm, quantity=True).to_value(pot_unit).reshape(X.shape),
                evaluateDensities(pot_scf, Rm, zm, phi=pm, quantity=True).to_value(rho_unit).reshape(X.shape))

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        for a, name, ylab in [(ax[0, 0], "x-y", "y [kpc]"), (ax[0, 1], "x-z", "z [kpc]")]:
            cs = a.contour(g, g, maps[name][0], levels=15)
            a.clabel(cs, fontsize=7, fmt="%.0f")
            a.set(xlabel="x [kpc]", ylabel=ylab, title=f"Equipotentials, {name} plane", aspect="equal")

        for a, name, ylab in [(ax[1, 0], "x-y", "y [kpc]"), (ax[1, 1], "x-z", "z [kpc]")]:
            rho_map = np.ma.masked_less_equal(maps[name][1], 0)
            cf = a.contourf(g, g, np.log10(rho_map), levels=20)
            fig.colorbar(cf, ax=a, label=r"log$_{10}$ $\rho$ [M$_\odot$ kpc$^{-3}$]")
            a.set(xlabel="x [kpc]", ylabel=ylab, title=f"Density, {name} plane", aspect="equal")

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    return r_grid * u.kpc, Phi_mean, rho_mean


def integrate_orbits_scf(pot, r0, f_vc=(1.0, 0.7, 0.4), n_orbits=5, n_steps=1000,
                     ro=8.0, vo=220.0, plot=True, output=None):
    """
    Integrates orbits in a galpy potential, spherical or not, and optionally plots them together with
    their energy. Every orbit starts at radius r0 on the x axis in the z = 0 plane (phi = 0), with a
    purely tangential velocity f * v_c(r0).

    Parameters:

    pot : galpy.potential
          The potential, e.g. pot_scf from build_SCF or pot_einasto from build_Einasto

    r0 : astropy Quantity
         Starting radius (kpc), should be inside the range covered by the particles.

    f_vc : sequence of float
           Tangential velocity of each orbit as a fraction of the circular velocity at r0 (at phi = 0).
           1.0 is a circular orbit in the plane. Smaller values give more eccentric orbits.

    n_orbits : float
               Integration time in units of the circular orbital period at r0

    n_steps : int
              Number of time steps

    ro, vo : float
             galpy's unit scales (kpc, km/s), must be the same as used for the potential

    plot : bool
           Whether to plot the orbits and their energy

    Returns:

    orbits : list of galpy.orbit.Orbit
             The integrated orbits, one for each value in f_vc

    ts : list of astropy Quantity
         The times (Gyr) at which each orbit was integrated

    """
    # Circular velocity at the starting point (phi = 0), where all the orbits start
    vc0 = vcirc(pot, r0, phi=0. * u.rad, quantity=True).to(u.km / u.s)

    # Integrate for n_orbits circular periods at r0
    T = n_orbits * (2 * np.pi * r0 / vc0).to(u.Gyr)
    ts = np.linspace(0, T.to_value(u.Gyr), n_steps) * u.Gyr

    orbits = []
    for f in f_vc:
        # [R, vR, vT, z, vz, phi]: purely tangential velocity, starting in the z = 0 plane
        o = Orbit([r0, 0. * u.km / u.s, f * vc0, 0. * u.kpc, 0. * u.km / u.s, 0. * u.rad],
                  ro=ro, vo=vo)
        o.integrate(ts, pot)
        orbits.append(o)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        for f, o in zip(f_vc, orbits):
            x = o.x(ts, quantity=True).to_value(u.kpc)
            y = o.y(ts, quantity=True).to_value(u.kpc)
            z = o.z(ts, quantity=True).to_value(u.kpc)
            label = rf"$v_T = {f:.2f}\,v_c$"

            ax[0].plot(x, y, label=label)
            ax[1].plot(x, z, label=label)

        ax[0].set(xlabel="x [kpc]", ylabel="y [kpc]", title="Orbits, x-y projection", aspect="equal")
        ax[1].set(xlabel="x [kpc]", ylabel="z [kpc]", title="Orbits, x-z projection", aspect="equal")
        for a in ax:
            a.legend()

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    return orbits, [ts] * len(orbits)


def circular_velocity_scf(pot_scf, r, n_phi=8):
    """
    Computes the circular velocity v_c(r) of an SCF potential with galpy's vcirc, in the z = 0 plane.
    The SCF potential is not axisymmetric, so v_c depends on the azimuth. It is computed at 'n_phi'
    azimuths and averaged.

    Parameters:

    pot_scf : galpy.potential
              The SCF potential, e.g. from build_SCF

    r : astropy Quantity
        Radii (kpc) at which to evaluate v_c, within the range covered by the particles

    n_phi : int
            Number of azimuths (evenly spaced between 0 and 2 pi) to average over

    Returns:

    vc : astropy Quantity
         The azimuthally averaged circular velocity (km/s) at the radii in 'r'

    """
    phis = np.linspace(0., 2 * np.pi, n_phi, endpoint=False) * u.rad
    # Circular velocity at every azimuth, shape (n_phi, len(r))
    vc_phi = np.array([vcirc(pot_scf, r, phi=phi, quantity=True).to_value(u.km / u.s) for phi in phis])
    vc = vc_phi.mean(axis=0) * u.km / u.s
    return vc_phi, vc


def compare_rotation_curves_scf(file, n_bins=15, plot=True, output=None):
    """
    Plots the circular velocity v_c(r) of the halo from the SCF potential and from the enclosed mass,
    together with the tangential velocity v_t of the particles.

    1) directly from the SCF potential, with vcirc in the z = 0 plane averaged over azimuth
    2) from the enclosed mass of the particles, v_c = sqrt(G M(<r) / r)
    3) the tangential velocity v_t of the particles (the speed perpendicular to the radial direction)

    Parameters:

    file : .hdf5 file
           Data from the TNG simulation (one halo)

    n_bins : int
             Number of logarithmic radial shells for the particle velocities

    plot : bool
           Whether to plot the curves

    Returns:

    (r_ax, vc_model) : radii and circular velocity from the SCF potential
    (r_ax, vc_mass) : radii and circular velocity from the enclosed mass
    (r_t, vt_mean, vt_rms) : shell centers, mean v_t and rms v_t of the particles
    all as astropy Quantities (kpc and km/s)

    """
    # 1) Directly from the SCF potential, on the radius grid that build_SCF returns
    pot_scf, r_ax = build_scf(file, plot=False)
    vc_phi_model, vc_model = circular_velocity_scf(pot_scf, r_ax)

    # 2) From the enclosed mass: v_c = sqrt(G M(<r) / r). The enclosed mass is the fourth output of
    # potential_plotter, which is interpolated onto the SCF radius grid (both span the same range)
    _, r_M, _, M = spherical_potential_plotter(file, plot=False)
    M_ax = np.interp(r_ax, r_M, M)
    vc_mass = np.sqrt(const.G * M_ax / r_ax).to(u.km / u.s)

    # 3) The tangential velocities of the particles
    r_range = np.array([r_ax[0].to_value(u.kpc), r_ax[-1].to_value(u.kpc)]) * u.kpc
    r_t, vt_mean, vt_rms = tangential_velocity(file, n_bins=n_bins, r_range=r_range)

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        axs[1].semilogx(r_ax.to_value(u.kpc), vc_model.to_value(u.km / u.s), label="fit - SCF")
        axs[1].semilogx(r_ax.to_value(u.kpc), vc_mass.to_value(u.km / u.s), "--",
                     label=r"data - enclosed mass")
        axs[1].semilogx(r_t.to_value(u.kpc), vt_mean.to_value(u.km / u.s), "o-",
                     label=r"data - mean $|v_t|$")
        axs[1].semilogx(r_t.to_value(u.kpc), vt_rms.to_value(u.km / u.s), "s-",
                     label=r"data - rms $v_t$")
        axs[1].set_xlabel("r [kpc]")
        axs[1].set_ylabel("velocity [km/s]")
        axs[1].set_title("Comparison")
        axs[1].legend()

        axs[0].semilogx(r_ax.to_value(u.kpc), vc_model.value, label="azimuthal mean")
        # Spread between azimuths: it shows the asymmetry of the potential in the plane
        lo, hi = np.percentile(vc_phi_model, [16, 84], axis=0)
        axs[0].fill_between(r_ax.to_value(u.kpc), lo, hi, alpha=0.3, label="16-84% over azimuth")
        axs[0].set_xlabel("r [kpc]")
        axs[0].set_ylabel(r"$v_c$ [km/s]")
        axs[0].set_title("Circular velocity, SCF potential")
        axs[0].legend()

        fig.tight_layout()
        if output is not None:
            fig.savefig(output)

    return (r_ax, vc_model), (r_ax, vc_mass), (r_t, vt_mean, vt_rms)
