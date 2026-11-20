"""Large-scale statistics for an IllustrisTNG subhalo catalogue.

Units: the catalogue stores lengths in ckpc/h and masses in 1e10 Msun/h. Astropy is
used once, at load time, to convert to h-free units. Internally everything is a
plain float array in comoving Mpc (cMpc) and Msun.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree

# -------------- DATA ----------------


def get_subhalo_data(tng_path, min_mass=0.0):
    """Return box size [cMpc], positions [cMpc] (N, 3) and masses [Msun].

    Subhalos with non-positive mass or mass below ``min_mass`` are removed.
    """
    tng_path = Path(tng_path)
    if tng_path.suffix.lower() not in {".hdf5", ".h5"}:
        raise ValueError(f"{tng_path} is not a .hdf5/.h5 file.")

    with h5py.File(tng_path, "r") as f:
        h = float(f["Header"].attrs["HubbleParam"])
        box_raw = float(f["Header"].attrs["BoxSize"])          # ckpc/h
        pos_raw = np.asarray(f["Subhalo/SubhaloPos"], dtype=np.float64)  # ckpc/h
        mass_raw = np.asarray(f["Subhalo/SubhaloMass"], dtype=np.float64)  # 1e10 Msun/h

    # ckpc/h -> cMpc, and 1e10 Msun/h -> Msun
    box_size = ((box_raw / h) * units.kpc).to_value(units.Mpc)
    positions = ((pos_raw / h) * units.kpc).to_value(units.Mpc)
    masses = ((mass_raw / h) * 1e10 * units.M_sun).to_value(units.M_sun)

    keep = masses > 0
    if min_mass > 0:
        keep &= masses >= min_mass
    return box_size, positions[keep], masses[keep]


def sample_positions(positions, max_halos=20_000, rng_seed=42):
    """Uniformly subsample positions, or keep all when max_halos is zero."""
    if max_halos < 0:
        raise ValueError("max_halos must be non-negative")
    if max_halos == 0 or len(positions) <= max_halos:
        return positions
    rng = np.random.default_rng(rng_seed)
    indices = np.sort(rng.choice(len(positions), size=max_halos, replace=False))
    return positions[indices]


# -------------- CALCULATIONS ----------------


def _wrap(positions, box_size):
    """Wrap positions into [0, box_size)."""
    wrapped = np.mod(positions, box_size)
    wrapped[wrapped >= box_size] = 0.0
    return wrapped


def _pair_histogram(tree_a, tree_b, edges, same_catalog=False):
    """Count pairs in radial shells using cKDTree's cumulative pair counter."""
    pairs = np.asarray(
        tree_a.count_neighbors(tree_b, edges, cumulative=False), dtype=np.int64
    )[1:] # first element contains r < rmin, we don't need this
    if same_catalog:
        # halve the ordered pairs (ij and ji), double counting
        pairs = pairs // 2
    return pairs


def get_correlation_estimator(positions, box_size, rmin=0.1, rmax=None, bins=16):
    """Natural (analytic RR) xi(r) in periodic spherical shells."""
    positions = np.asarray(positions, dtype=np.float64)
    n_data = len(positions)
    if n_data < 2:
        raise ValueError("At least two subhalos are needed for a correlation function")
    half_box = box_size / 2.0
    if rmax is None:
        rmax = 0.2 * box_size
    if not (0 < rmin < rmax <= half_box):
        raise ValueError("Require 0 < rmin < rmax <= BoxSize/2 for periodic shells")

    edges = np.geomspace(rmin, rmax, bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # larger leaf size for large rmax
    data_tree = cKDTree(_wrap(positions, box_size), boxsize=box_size, balanced_tree=False, leafsize=48)
    dd = _pair_histogram(data_tree, data_tree, edges, same_catalog=True)
    dd_norm = dd / (n_data * (n_data - 1) / 2)
    rr_norm = (4.0 * np.pi / 3.0) * np.diff(edges**3) / box_size**3

    with np.errstate(divide="ignore", invalid="ignore"):
        natural = dd_norm / rr_norm - 1.0
    return centers, natural


def get_correlation_estimator_comparison(
    positions,
    box_size,
    rmin=0.1,
    rmax=None,
    bins=16,
    random_to_sim=1,
    max_random=100_000,
    rng_seed=42,
):
    """Natural (analytic RR) and Landy-Szalay xi(r) in periodic spherical shells."""
    n_data = len(positions)
    if n_data < 2:
        raise ValueError("At least two subhalos are needed for a correlation function")
    half_box = box_size / 2.0
    if rmax is None:
        rmax = 0.2 * box_size
    if not (0 < rmin < rmax <= half_box):
        raise ValueError("Require 0 < rmin < rmax <= BoxSize/2 for periodic shells")

    edges = np.geomspace(rmin, rmax, bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    data_tree = cKDTree(_wrap(positions, box_size), boxsize=box_size, balanced_tree=False, leafsize=32)
    dd = _pair_histogram(data_tree, data_tree, edges, same_catalog=True)

    # random catalogue, between random_to_sim * n_data and max_random points
    n_random = min(int(random_to_sim * n_data), max_random)
    rng = np.random.default_rng(rng_seed)
    random_tree = cKDTree(rng.uniform(0.0, box_size, size=(n_random, 3)), boxsize=box_size, leafsize=32)
    dr = _pair_histogram(data_tree, random_tree, edges)
    rr = _pair_histogram(random_tree, random_tree, edges, same_catalog=True)

    # normalise by the total number of pairs
    dd_norm = dd / (n_data * (n_data - 1) / 2)
    dr_norm = dr / (n_data * n_random)
    rr_norm = rr / (n_random * (n_random - 1) / 2)

    # below half the box the shell's volume fraction is exact
    rr_norm_analytic = (4.0 * np.pi / 3.0) * np.diff(edges**3) / box_size**3

    with np.errstate(divide="ignore", invalid="ignore"):
        natural = dd_norm / rr_norm_analytic - 1.0
        ls = (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm

    return centers, natural, ls


# --------------- PLOTTING -----------------


def plot_mass_function(
    masses, box_size, bins=30, output=None, min_cumulative_density=1e-3
):
    """Plot differential and cumulative mass functions with a fitted slope."""
    if not np.isfinite(min_cumulative_density) or min_cumulative_density < 0:
        raise ValueError("min_cumulative_density must be finite and non-negative")

    logm = np.log10(masses)
    lo, hi = logm.min(), logm.max()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts, _ = np.histogram(logm, bins=edges)
    dn_dlogm = counts / (box_size**3 * widths)

    masses_sorted = np.sort(masses)
    n_cumulative = np.arange(len(masses_sorted), 0, -1) / box_size**3   # n(>=M)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    good = counts > 0
    axs[0].plot(10**centers[good], dn_dlogm[good], "s-")
    axs[0].set_xlabel(r"$M_{\rm sub}$ $[M_\odot]$")
    axs[0].set_ylabel(r"$dn/d\log_{10}(M/M_\odot)$ $[\mathrm{cMpc}^{-3}]$")
    axs[0].set_title("Subhalo mass function")
    axs[1].step(masses_sorted, n_cumulative, where="post", label="cumulative data")

    # fit to cumulative function
    if masses_sorted[0] < masses_sorted[-1]:
        fit_masses = np.geomspace(
            masses_sorted[0], masses_sorted[-1], num=min(1000, len(masses_sorted))
        )
        fit_counts = len(masses_sorted) - np.searchsorted(
            masses_sorted, fit_masses, side="left"
        )
        fit_density = fit_counts / box_size**3
        fit_mask = fit_density > min_cumulative_density
        if fit_mask.sum() >= 2:
            slope, intercept = np.polyfit(
                np.log10(fit_masses[fit_mask]), np.log10(fit_density[fit_mask]), deg=1
            )
            axs[1].plot(
                fit_masses,
                10 ** (slope * np.log10(fit_masses) + intercept),
                "--",
                label=f"power-law fit ($n \\geq$ {min_cumulative_density:.1e}, slope = {slope:.2f})",
            )
            axs[1].legend()
        else:
            print(
                "Note: fewer than two cumulative mass-function points exceed "
                f"{min_cumulative_density:g} cMpc^-3; skipping the fit."
            )

    axs[1].set_xlabel(r"$M_{\rm sub}$ $[M_\odot]$")
    axs[1].set_ylabel(r"$n(>M_{\rm sub})$ $[\mathrm{cMpc}^{-3}]$")
    axs[1].set_title("Cumulative subhalo mass function")

    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    if output is not None:
        fig.savefig(output)


def plot_correlation_function_comparison(
    positions, box_size, output=None, xi_fit_max=8.0, **kwargs
):
    """Plot correlation estimators and fit the natural estimator below xi_fit_max."""
    if not np.isfinite(xi_fit_max) or xi_fit_max <= 0:
        raise ValueError("xi_fit_max must be a finite positive value")

    radii, natural, ls = get_correlation_estimator_comparison(
        positions, box_size, **kwargs
    )

    # xi is on a log y-axis, so only strictly positive, finite values can be drawn
    ok_nat = np.isfinite(natural) & (natural > 0)
    ok_ls = np.isfinite(ls) & (ls > 0)
    if (~ok_nat).any() or (~ok_ls).any():
        print(
            f"Note: omitted {(~ok_nat).sum()} natural and {(~ok_ls).sum()} Landy-Szalay "
            "bins with xi <= 0 (cannot be shown on a log axis)."
        )

    # we also plot the chi2 / len(radii) for different sizes of random catalogues
    # to see that the LS estimator converges to the natural estimator for pbc
    low_naturals, low_ls = np.zeros((2, 5, natural.size), dtype=float)
    low_naturals[0] = natural
    low_ls[0] = ls
    for i, random_to_sim in enumerate([0.8, 0.6, 0.4, 0.2]):
        _, this_natural, this_ls = get_correlation_estimator_comparison(positions, box_size, random_to_sim=random_to_sim, **kwargs)
        low_naturals[i+1] = this_natural
        low_ls[i+1] = this_ls

    reduced_chi2s = np.sum((low_naturals - low_ls)**2, axis=1, keepdims=False) / len(radii)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    axs[0].plot(radii[ok_nat], natural[ok_nat], "s-", label="natural (analytic RR)")
    axs[0].plot(radii[ok_ls], ls[ok_ls], "o-", label="Landy–Szalay")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$r$ [$\mathrm{cMpc}$]")
    axs[0].set_ylabel(r"$\xi(r)$")
    axs[0].set_title("Subhalo autocorrelation function")

    axs[1].plot(np.array([1.0, 0.8, 0.6, 0.4, 0.2]), reduced_chi2s, "s-")
    axs[1].set_yscale("log")
    axs[1].set_xlabel(r"$N_\mathrm{R} / N_\mathrm{D}$")
    axs[1].set_ylabel(r"$\sum_r (\xi_\mathrm{LS}(r) - \xi_\mathrm{nat}(r))^2 / N_\mathrm{bins}$")
    axs[1].set_title("Comparison between natural & Landy-Szalay")

    # fit to natural estimator
    fit_nat = ok_nat & (natural <= xi_fit_max)
    if fit_nat.sum() >= 2:
        slope, intercept = np.polyfit(
            np.log10(radii[fit_nat]), np.log10(natural[fit_nat]), deg=1
        )
        if slope != 0:
            max_xi_radius = 10 ** (
                (np.log10(natural[ok_nat].max()) - intercept) / slope
            )
            fit_radii = np.geomspace(
                min(max_xi_radius, radii[fit_nat].min()),
                max(max_xi_radius, radii[fit_nat].max()),
                100,
            )
            axs[0].plot(
                fit_radii,
                10 ** (slope * np.log10(fit_radii) + intercept),
                "--",
                label=f"natural fit ($\\xi \\leq$ {xi_fit_max:g}, slope = {slope:.2f})",
            )
    else:
        print(
            f"Note: fewer than two finite positive natural-estimator bins have "
            f"xi <= {xi_fit_max:g}; skipping the fit."
        )
    axs[0].legend()

    fig.tight_layout()
    if output is not None:
        fig.savefig(output)


def plot_correlation_function(
    positions,
    masses,
    box_size,
    n_positions=20_000,
    min_masses=(1e8, 1e9, 1e10),
    output=None,
    xi_fit_max=8.0,
    rng_seed=42,
    **kwargs,
):
    """Compare a uniform position sample with three mass-threshold samples."""
    if not np.isfinite(xi_fit_max) or xi_fit_max <= 0:
        raise ValueError("xi_fit_max must be a finite positive value")

    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if masses.ndim != 1 or len(masses) != len(positions):
        raise ValueError("masses must be a one-dimensional array aligned with positions")
    if not np.isfinite(masses).all():
        raise ValueError("masses must contain only finite values")
    if not isinstance(n_positions, (int, np.integer)) or n_positions < 2:
        raise ValueError("n_positions must be an integer of at least two")
    if n_positions > len(positions):
        raise ValueError(
            f"n_positions ({n_positions}) cannot exceed the number of positions "
            f"({len(positions)})"
        )

    sample = sample_positions(positions, max_halos=n_positions, rng_seed=rng_seed)
    radii, xi = get_correlation_estimator(sample, box_size, **kwargs)

    # xi is on a log y-axis, so only strictly positive, finite values can be drawn
    ok = np.isfinite(xi) & (xi > 0)
    if (~ok).any():
        print(
            f"Note: omitted {(~ok).sum()} uniform-sample bins with xi <= 0 "
            " (cannot be shown on a log axis)."
        )

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    axs[0].plot(
        radii[ok],
        xi[ok],
        "s-",
        label=f"uniform sample ($N={len(sample):,}$)",
    )
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$r$ [$\mathrm{cMpc}$]")
    axs[0].set_ylabel(r"$\xi(r)$")
    axs[0].set_title("Uniform position subsample")

    for min_mass in min_masses:
        selected = positions[masses > min_mass]
        if len(selected) < 2:
            raise ValueError(
                f"min_mass={min_mass:g} leaves fewer than two halos; "
                "cannot compute a correlation function"
            )
        _, mass_xi = get_correlation_estimator(selected, box_size, **kwargs)
        mass_ok = np.isfinite(mass_xi) & (mass_xi > 0)
        if (~mass_ok).any():
            print(
                f"Note: omitted {(~mass_ok).sum()} bins for min_mass={min_mass:g} "
                "(xi <= 0 cannot be shown on a log axis)."
            )
        axs[1].plot(
            radii[mass_ok],
            mass_xi[mass_ok],
            "o-",
            label=(
                rf"$M_{{\rm sub}} > {min_mass:.1e}\,M_\odot$ "
                f"($N={len(selected):,}$)"
            ),
        )

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel(r"$r$ [$\mathrm{cMpc}$]")
    axs[1].set_ylabel(r"$\xi(r)$")
    axs[1].set_title("Mass-threshold subsamples")
    axs[1].legend()

    # Fit only the uniform position subsample.
    fit_nat = ok & (xi <= xi_fit_max)
    if fit_nat.sum() >= 2:
        slope, intercept = np.polyfit(
            np.log10(radii[fit_nat]), np.log10(xi[fit_nat]), deg=1
        )
        if slope != 0:
            max_xi_radius = 10 ** (
                (np.log10(xi[ok].max()) - intercept) / slope
            )
            fit_radii = np.geomspace(
                min(max_xi_radius, radii[fit_nat].min()),
                max(max_xi_radius, radii[fit_nat].max()),
                100,
            )
            axs[0].plot(
                fit_radii,
                10 ** (slope * np.log10(fit_radii) + intercept),
                "--",
                label=f"fit ($\\xi \\leq$ {xi_fit_max:g}, slope = {slope:.2f})",
            )
    else:
        print(
            f"Note: fewer than two finite positive estimator bins have "
            f"xi <= {xi_fit_max:g}; skipping the fit."
        )
    axs[0].legend()

    fig.tight_layout()
    if output is not None:
        fig.savefig(output)


def plot_column_density(positions, n_bins=200, output=None):
    """Plot subhalo number column density in the XY, XZ, and YZ planes."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("positions must be a non-empty array with shape (N, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("positions must contain only finite values")

    projections = ((0, 1), (0, 2), (1, 2))
    coordinates = ("x", "y", "z")
    densities = []
    bin_edges = []
    for axis_x, axis_y in projections:
        density, x_edges, y_edges = np.histogram2d(
            positions[:, axis_x], positions[:, axis_y], bins=n_bins
        )
        pixel_areas = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
        densities.append(density / pixel_areas)
        bin_edges.append((x_edges, y_edges))

    positive = np.concatenate([density[density > 0] for density in densities])
    vmin, vmax = positive.min(), positive.max()
    if vmin == vmax:
        vmax = np.nextafter(vmax, np.inf)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True)
    for ax, (axis_x, axis_y), density, (x_edges, y_edges) in zip(
        axs, projections, densities, bin_edges
    ):
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_equal(density.T, 0),
            shading="auto",
            norm=norm,
            cmap="magma",
        )
        ax.set_xlabel(f"${coordinates[axis_x]}$ [cMpc]")
        ax.set_ylabel(f"${coordinates[axis_y]}$ [cMpc]")
        ax.set_title(f"{coordinates[axis_x].upper()}{coordinates[axis_y].upper()} plane")

    fig.colorbar(
        mesh, ax=axs, label=r"column density [$\mathrm{cMpc}^{-2}$]"
    )
    if output is not None:
        fig.savefig(output)
