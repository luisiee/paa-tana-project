"""Large-scale statistics for an IllustrisTNG subhalo catalogue (project Part B).

Run from any directory with ``python statistics.py``. By default the catalogue is
read from ``../data`` (relative to this file), and plots are written to
``figures`` in the current directory. TNG catalogue masses and
positions are kept in their usual Msun/h and comoving Mpc/h units.
The mass function uses the full catalogue; the correlation function uses a
reproducible uniform sample of at most 20,000 subhalos by default. Set
``--max-correlation-halos 0`` to use the full catalogue.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units
from scipy.spatial import cKDTree

# -------------- CALCULATIONS ----------------


def get_subhalo_data(tng_path):
    """Get box size, subhalo positions and masses from a TNG HDF5 file."""
    assert str.split(tng_path, '.')[-1].lower() in {"hdf5", "h5"}, f"{tng_path} is not a .hdf5/.h5 file."

    data_h5 = h5py.File(tng_path, 'r')
    h = data_h5['Header'].attrs['HubbleParam']

    box_size = data_h5['Header'].attrs['BoxSize'] / h
    positions = np.asarray(data_h5['Subhalo']['SubhaloPos'], np.float64) / h
    masses = np.asarray(data_h5['Subhalo']['SubhaloMass'], np.float64) / h

    # length units are comoving kpc
    # mass units are 1e10 M_sun
    box_size = (box_size * units.kpc).to(units.Mpc)
    positions = (positions * units.kpc).to(units.Mpc)
    masses = (masses * 1e10) * units.M_sun

    return box_size, positions, masses


def sample_positions(positions, max_halos=20_000, rng_seed=7):
    """Uniformly subsample positions, or keep all when max_halos is zero."""
    if max_halos < 0:
        raise ValueError("max_halos must be non-negative")
    if max_halos == 0 or len(positions) <= max_halos:
        return positions
    rng = np.random.default_rng(rng_seed)
    indices = np.sort(rng.choice(len(positions), size=max_halos, replace=False))
    return positions[indices]


def _pair_histogram(tree_a, tree_b, edges, same_catalog=False):
    """Count pairs in radial shells using cKDTree's cumulative pair counter."""
    cumulative = np.asarray(tree_a.count_neighbors(tree_b, edges, cumulative=True),
                            dtype=np.int64)
    if same_catalog:
        # self-pairs (N) and both orientations (ij, ji) of every distinct pair are counted
        # and must be divided out
        cumulative = (cumulative - tree_a.n) // 2
    return np.diff(cumulative)


def get_correlation_estimators(
    positions,
    box_size,
    rmin=None,
    rmax=None,
    bins=18,
    random_to_sim=1,
    max_random=100_000,
    rng_seed=7
):
    """Return natural and Landy-Szalay xi estimates in periodic spherical shells."""

    n_data = len(positions)
    if n_data < 2:
        raise ValueError("At least two subhalos are needed for a correlation function")
    half_box = box_size / 2.0
    if rmax is None:
        rmax = 0.45 * half_box
    if rmin is None:
        rmin = max(box_size * 1e-3, rmax / 1000.0)
    if not (0 < rmin < rmax <= half_box):
        raise ValueError("Require 0 < rmin < rmax <= BoxSize/2 for periodic shells")

    # spherical shells
    edges = np.geomspace(rmin, rmax, bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # generate a tree and calculate the counts of pairs of halos
    data_tree = cKDTree(np.mod(positions, box_size), boxsize=box_size)
    dd = _pair_histogram(data_tree, data_tree, edges, same_catalog=True)

    # generate a random catalogue of random_to_sim * n_data points
    # capped between 20,000 and max_random
    n_random = min(max(int(random_to_sim * n_data), 20_000), max_random)
    rng = np.random.default_rng(rng_seed)
    random_tree = cKDTree(rng.uniform(0.0, box_size / units.Mpc, size=(n_random, 3)) * units.Mpc, boxsize=box_size)

    # calculate the counts of pairs of halos between data and random catalogues
    dr = _pair_histogram(data_tree, random_tree, edges)
    rr = _pair_histogram(random_tree, random_tree, edges, same_catalog=True)

    # normalize by the total number of pairs
    dd_norm = dd / (n_data * (n_data - 1) / 2)
    dr_norm = dr / (n_data * n_random)
    rr_norm = rr / (n_random * (n_random - 1) / 2)

    # for radii below half the periodic box, the shell's volume fraction is exact
    rr_norm_analytical = (4.0 * np.pi / 3.0) * np.diff(edges**3) / box_size**3
    with np.errstate(divide="ignore", invalid="ignore"):
        natural = dd_norm / rr_norm_analytical - 1.0
        ls = (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm

    return centers, natural, ls


# --------------- PLOTTING -----------------


def plot_mass_function(masses, box_size, bins=30, output=None):
    """Plot the mass function and cumulative mass function per comoving Mpc^3."""

    # mass function
    edges = np.linspace(np.log10(masses.min() / units.M_sun), np.log10(masses.max() / units.M_sun), bins + 1)
    if edges[0] == edges[-1]:
        edges = np.linspace(edges[0] - 0.5, edges[-1] + 0.5, bins + 1)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts, _ = np.histogram(np.log10(masses / units.M_sun), bins=edges)
    dndlog10M = counts / (box_size**3 * widths)

    # cumulative mass function
    masses_sorted = np.sort(masses)
    n_cumulative = np.arange(len(masses_sorted), 0, -1) / box_size**3

    # plotting
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    axs[0].plot(10**centers[counts > 0], dndlog10M[counts > 0], "D-")
    axs[0].set_xlabel(r"$M_{\rm sub}$ $[M_\odot]$")
    axs[0].set_ylabel(r"$dn/d\log_{10}(M/M_\odot)$ $[\mathrm{cMpc}^{-3}]$")
    axs[0].set_title("Subhalo mass function")
    axs[1].step(masses_sorted, n_cumulative, where="post")
    axs[1].set_xlabel(r"$M_{\rm sub} [M_\odot]$")
    axs[1].set_ylabel(r"# halos $(> M_{\rm sub})$ $[\mathrm{cMpc}^{-3}]$")
    axs[1].set_title("Cumulative subhalo mass function")

    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    if output is not None:
        fig.savefig(output)


def plot_correlation_function(positions, box_size, output=None, **kwargs):
    radii, natural, landy_szalay = get_correlation_estimators(
        positions, box_size, **kwargs
    )
    fig, ax = plt.subplots()
    ax.plot(radii, natural, "D-", label="natural (analytic RR)")
    ax.plot(radii, landy_szalay, "s-", label="Landy–Szalay")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$ [$\mathrm{cMpc}$]")
    ax.set_ylabel(r"$\xi(r)$")
    ax.set_title("Subhalo autocorrelation function")

    ax.legend()
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=180)


if __name__ == "__main__":
    plt.style.use(Path(__file__).parent / 'nicefigures.mplstyle')

    box_size, positions, masses = get_subhalo_data("data/TNG50-1-Dark_subhalosCat.hdf5")
    plot_mass_function(masses, box_size, output="figures/subhalo_mass_function")
    positions_sample = sample_positions(positions)    # sample 20,000 positions
    plot_correlation_function(positions_sample, box_size, output="figures/subhalo_correlation_function")
