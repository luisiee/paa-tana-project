# paa-tana-project

We, *kobebrande* and *luisiee* are **Group 4** for the project of the course *Theoretical and Numerical Astrophysics* (TANA) for Physics and Astronomy at Ghent University. This project was done during the first semester of the academic year 2026-2027.

## Result Reproduction

Group 4 uses two data files of the *IllustrisTNG* simulation, which can be found on the Ufora page for the course, under the names:

- TNG50-1-Dark_subhalosCat.hdf5
- cutout_188106.hdf5

As these files are untracked by git, they must be manually added to your local clone of the repository. The following file structure must be employed:

```
|-- .git/
|   |-- ...
|
|-- ...
|
|-- data/
|   |-- TNG50-1-Dark_subhalosCat.hdf5
|   |-- cutout_188106.hdf5
```

The discussion for **Part 1** of the project can be found in the pdf file `part1.pdf` or jupyter notebook `part1.ipynb`. Figures can be found in the `figures/` directory, and are reproduced by running the notebooks. Figures use the [nice-figures](https://github.com/luisiee/nice-figures) matplotlib style. The source code can be found in the `src/` directory.
