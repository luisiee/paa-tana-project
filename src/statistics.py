"""Large-scale statistics for an IllustrisTNG subhalo catalogue (project Part B).

Run from any directory with ``python statistics.py``. By default the catalogue is
read from ``../data`` (relative to this file), and plots are written to
``figures`` in the current directory. TNG catalogue masses and
positions are kept in their usual Msun/h and comoving Mpc/h units.
The mass function uses the full catalogue; the correlation function uses a
reproducible uniform sample of at most 20,000 subhalos by default. Set
``--max-correlation-halos 0`` to use the full catalogue.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def _walk_datasets(group):
	"""Yield (full path, dataset) pairs recursively from an HDF5 group."""
	found = []

	def visit(name, obj):
		if isinstance(obj, h5py.Dataset):
			found.append((name, obj))

	group.visititems(visit)
	return found


def _find_named_dataset(datasets, name):
	candidates = [(path, ds) for path, ds in datasets
				  if path.rsplit("/", 1)[-1].lower() == name.lower()]
	if not candidates:
		return None
	# Prefer the usual TNG /Subhalo/Subhalo... dataset over similarly named data.
	candidates.sort(key=lambda item: ("subhalo/" not in item[0].lower(), -item[1].size))
	return candidates[0][1]


def load_catalog(data_dir):
	"""Load SubhaloMass and SubhaloPos from one or more TNG HDF5 files."""
	data_dir = Path(data_dir)
	files = sorted(p for p in data_dir.rglob("*")
				   if p.is_file() and p.suffix.lower() in {".hdf5", ".h5"})
	if not files:
		raise FileNotFoundError(f"No .hdf5/.h5 files found in {data_dir}")

	masses, positions = [], []
	box_size = None
	for path in files:
		with h5py.File(path, "r") as handle:
			datasets = _walk_datasets(handle)
			mass_ds = _find_named_dataset(datasets, "SubhaloMass")
			pos_ds = _find_named_dataset(datasets, "SubhaloPos")
			if mass_ds is not None and pos_ds is not None:
				mass = np.asarray(mass_ds, dtype=np.float64).reshape(-1)
				pos = np.asarray(pos_ds, dtype=np.float64)
				if pos.ndim == 2 and pos.shape[1] == 3 and len(mass) == len(pos):
					masses.append(mass)
					positions.append(pos)

			if box_size is None:
				for key in ("Header", "/"):
					obj = handle[key]
					attrs = obj.attrs
					for attr in ("BoxSize", "boxsize"):
						if attr in attrs:
							box_size = float(np.asarray(attrs[attr]).reshape(-1)[0])
							break
					if box_size is not None:
						break

	if not masses:
		raise ValueError(
			f"Could not find matching SubhaloMass and SubhaloPos datasets in {data_dir}. "
			"Expected TNG-style dataset names."
		)

	mass = np.concatenate(masses)
	pos = np.concatenate(positions)
	# TNG stores positions in ckpc/h and BoxSize in the same units; work in Mpc/h.
	if box_size is not None and box_size > 1000:
		pos /= 1000.0
		box_size /= 1000.0
	if box_size is None:
		warnings.warn("No Header/BoxSize found; using the coordinate extent as an "
					  "approximate periodic box. Verify this before interpreting xi(r).")
		box_size = float(np.max(pos.max(axis=0) - pos.min(axis=0)))
		pos -= pos.min(axis=0)
	if not np.isfinite(box_size) or box_size <= 0:
		raise ValueError("Invalid simulation BoxSize")
	return mass, pos, box_size


def plot_mass_function(mass, box_size, output, bins=30):
	"""Plot dN/dlog10(M) per comoving (Mpc/h)^3."""
	mass = np.asarray(mass)
	mass = mass[np.isfinite(mass) & (mass > 0)]
	if mass.size == 0:
		raise ValueError("The catalogue contains no positive finite subhalo masses")
	edges = np.linspace(np.log10(mass.min()), np.log10(mass.max()), bins + 1)
	if edges[0] == edges[-1]:
		edges = np.linspace(edges[0] - 0.5, edges[-1] + 0.5, bins + 1)
	counts, _ = np.histogram(np.log10(mass), bins=edges)
	widths = np.diff(edges)
	phi = counts / (box_size ** 3 * widths)
	centers = 0.5 * (edges[:-1] + edges[1:])

	fig, ax = plt.subplots(figsize=(7, 5))
	valid = counts > 0
	ax.plot(centers[valid], phi[valid], "o-")
	ax.set_yscale("log")
	ax.set_xlabel(r"$\log_{10}(M_{\rm sub}/[M_\odot/h])$")
	ax.set_ylabel(r"$dn/d\log_{10}M$ $[h^3\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
	ax.set_title("Subhalo mass function")
	ax.grid(True, which="both", alpha=0.25)
	fig.tight_layout()
	fig.savefig(output, dpi=180)
	plt.close(fig)
	return centers, phi, counts


def _pair_histogram(tree_a, tree_b, edges, same_catalog=False):
	"""Count pairs in radial shells using cKDTree's cumulative pair counter."""
	cumulative = np.asarray(tree_a.count_neighbors(tree_b, edges, cumulative=True),
							dtype=np.int64)
	if same_catalog:
		# Self-pairs (N) and both orientations of every distinct pair are counted.
		cumulative = (cumulative - tree_a.n) // 2
	return np.diff(cumulative)


def correlation_estimators(positions, box_size, rmin=None, rmax=None, bins=18,
						   random_factor=5, max_random=100_000, seed=7):
	"""Return natural and Landy-Szalay xi estimates in periodic spherical shells."""
	positions = np.asarray(positions, dtype=np.float64)
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
	edges = np.geomspace(rmin, rmax, bins + 1)

	data_tree = cKDTree(np.mod(positions, box_size), boxsize=box_size)
	dd = _pair_histogram(data_tree, data_tree, edges, same_catalog=True)
	n_random = min(max(int(random_factor * n_data), 20_000), max_random)
	rng = np.random.default_rng(seed)
	random_tree = cKDTree(rng.uniform(0.0, box_size, size=(n_random, 3)),
						  boxsize=box_size)
	dr = _pair_histogram(data_tree, random_tree, edges)
	rr = _pair_histogram(random_tree, random_tree, edges, same_catalog=True)

	dd_norm = dd / (n_data * (n_data - 1) / 2)
	dr_norm = dr / (n_data * n_random)
	rr_norm = rr / (n_random * (n_random - 1) / 2)
	with np.errstate(divide="ignore", invalid="ignore"):
		ls = (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm

	# For radii below half the periodic box, the shell's volume fraction is exact.
	shell_fraction = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3) / box_size ** 3
	expected = shell_fraction
	with np.errstate(divide="ignore", invalid="ignore"):
		natural = dd_norm / expected - 1.0
	centers = np.sqrt(edges[:-1] * edges[1:])
	return centers, natural, ls, dd, dr, rr


def plot_correlations(positions, box_size, output, **kwargs):
	radii, natural, landy_szalay, *_ = correlation_estimators(
		positions, box_size, **kwargs
	)
	fig, ax = plt.subplots(figsize=(7, 5))
	ax.plot(radii, natural, "o-", label="Natural (analytic RR)")
	ax.plot(radii, landy_szalay, "s-", label="Landy–Szalay")
	ax.axhline(0, color="black", linewidth=0.8)
	ax.set_xscale("log")
	ax.set_xlabel(r"Separation $r$ [$h^{-1}\,\mathrm{Mpc}$ comoving]")
	ax.set_ylabel(r"$\xi(r)$")
	ax.set_title("Subhalo two-point autocorrelation")
	ax.grid(True, which="both", alpha=0.25)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output, dpi=180)
	plt.close(fig)
	return radii, natural, landy_szalay


def sample_positions(positions, max_halos=20_000, seed=7):
	"""Uniformly subsample positions, or keep all when max_halos is zero."""
	positions = np.asarray(positions)
	if max_halos < 0:
		raise ValueError("max_halos must be non-negative")
	if max_halos == 0 or len(positions) <= max_halos:
		return positions
	rng = np.random.default_rng(seed)
	indices = np.sort(rng.choice(len(positions), size=max_halos, replace=False))
	return positions[indices]


def main():
	default_data = Path(__file__).resolve().parent.parent / "data"
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data", type=Path, default=default_data,
						help="directory containing TNG HDF5 catalogue file(s)")
	parser.add_argument("--output", type=Path, default=Path("figures"),
						help="directory for the output plots")
	parser.add_argument("--mass-bins", type=int, default=30)
	parser.add_argument("--radial-bins", type=int, default=18)
	parser.add_argument("--rmin", type=float, default=None,
						help="minimum separation in comoving Mpc/h")
	parser.add_argument("--rmax", type=float, default=None,
						help="maximum separation (must not exceed half the box)")
	parser.add_argument("--random-factor", type=float, default=5,
						help="random-catalogue size relative to the data sample")
	parser.add_argument("--max-random", type=int, default=100_000)
	parser.add_argument("--max-correlation-halos", type=int, default=20_000,
						help="uniformly sample at most this many halos for the correlation "
							 "function; use 0 to use all halos")
	parser.add_argument("--seed", type=int, default=7,
						help="random seed for reproducible sampling and random catalogue")
	args = parser.parse_args()
	if args.max_correlation_halos < 0:
		parser.error("--max-correlation-halos must be non-negative")

	mass, positions, box_size = load_catalog(args.data)
	args.output.mkdir(parents=True, exist_ok=True)
	plot_mass_function(mass, box_size, args.output / "halo_mass_function.png",
					   bins=args.mass_bins)
	correlation_positions = sample_positions(
		positions, max_halos=args.max_correlation_halos, seed=args.seed
	)
	plot_correlations(
		correlation_positions, box_size,
		args.output / "halo_correlation_function.png",
		rmin=args.rmin, rmax=args.rmax, bins=args.radial_bins,
		random_factor=args.random_factor, max_random=args.max_random, seed=args.seed,
	)
	print(f"Loaded {len(mass):,} subhalos; using "
		  f"{len(correlation_positions):,} for the correlation function; "
		  f"BoxSize = {box_size:g} Mpc/h")
	print(f"Saved plots to {args.output.resolve()}")


if __name__ == "__main__":
	main()
