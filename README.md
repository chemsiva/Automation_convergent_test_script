# Universal DFT Convergence Suite (Quantum ESPRESSO & SIESTA)

A modular, self-contained Python framework for executing systematic density functional theory (DFT) parameter convergence workflows, equation-of-state (EOS) determination, and publication-ready plotting for **any periodic crystal structure** from a `.cif` file. Supports both **Quantum ESPRESSO** (`pw.x`) and **SIESTA** (`siesta`).

Calculations adhere to the first-principles computational guidelines established by Keith Refson (*STFC Rutherford Appleton Laboratory*):
* Parameter sweeps run **single-point SCF** on fixed lattices to isolate numerical convergence.
* Ground-state volume and bulk modulus determination uses a **fixed-cutoff Equation-of-State (EOS) scan** followed by 3rd-order Birch-Murnaghan fitting, eliminating Pulay stress errors inherent to variable-cell algorithms.

---

## Key Features

1. **Generic CIF-Driven Engine**:
   * Supply any `.cif` file (cubic, tetragonal, orthorhombic, monoclinic, triclinic, hexagonal).
   * Automatically parses unit cell vectors, angles, atomic coordinates, stoichiometry, and chemical species.
   * Built-in periodic table data provides standard atomic numbers ($Z$) and atomic masses ($M$).
2. **Interactive Physical Interaction Toggles**:
   * **van der Waals (DFT-D3)**: Explicit optional toggle (yes/no) to include Becke-Johnson damped Grimme D3 corrections.
   * **Spin-Orbit Coupling (SOC)**: Explicit optional toggle (yes/no) for non-collinear relativistic calculations on heavy-atom systems ($\text{Pb, Sn, Bi, I}$).
3. **Smart Checkpointing & Resumability**:
   * Inspects existing output files (`output.out`) before running any calculation.
   * Automatically caches and skips previously completed points, allowing calculations to resume seamlessly after network drops or cluster job walltimes.
4. **Electronic State & Band Gap ($E_g$) Tracking**:
   * Tracks both total energy convergence ($|\Delta E| < 1.0\text{ meV/atom}$) and electronic band gap ($E_g$ in eV) at every cutoff and $k$-point step.
5. **Automated Publication Plotting**:
   * Generates a 300 DPI, 4-panel graphic figure (`convergence_summary.png`) plotting Cutoff, $k$-points, Basis Size, and the Birch-Murnaghan EOS curve with $V_0$ and $B_0$ annotations.
6. **Production Hand-off Generator (`production_inputs/`)**:
   * Exports `converged_parameters.json` (machine-readable parameters).
   * Generates production SCF inputs (`production_scf.in` or `production.fdf`) with the converged cutoff, $k$-mesh, and equilibrium cell.
   * Exports `relaxed_equilibrium.cif` with Birch-Murnaghan minimum lattice parameters.
7. **Dual Execution Modes**:
   * `Direct Execution`: Runs interactively via MPI on the current node.
   * `SLURM Mode`: Automatically drafts `#SBATCH` batch scripts tailored to cluster partitions and core allocations.

---

## Directory Setup

Place your structure `.cif` file in the folder with `convergence_suite.py`:

```
.
├── YourMaterial.cif             # Any crystal structure CIF
├── convergence_suite.py         # Universal runner
└── README.md                    # Documentation
```

---

## Prerequisites

1. **Python 3.8+** (standard library supported; `ase`, `scipy`, and `matplotlib` are used if installed).
2. **MPI implementation** (`mpirun`, `mpiexec`, OpenMPI, or MPICH).
3. **DFT Binaries**:
   * Quantum ESPRESSO: `pw.x`
   * SIESTA: `siesta` (versions 4.x or 5.x)
4. **Pseudopotentials**:
   * For Quantum ESPRESSO: Standard PBE UPF library (e.g., SSSP or Burai default).
   * For SIESTA: PBE `.psml` (e.g., standard `nc-sr-05_pbe`) or `.psf` files.

---

## Interactive Execution Guide

Start the suite in your terminal:

```bash
python3 convergence_suite.py
```

The script guides you through the setup steps:

```
  ── Step 1 : Crystal structure (CIF) ───────────────────
  Found CIF files in current directory:
    1) FAPbI3.cif
    2) FASnI3.cif
  Select number or enter path to CIF file [1]: 1

  ✓ Loaded crystal structure from: FAPbI3.cif
    Chemical formula : CH5I3N2Pb
    Lattice params   : a=6.3613 Å, b=6.3613 Å, c=6.3613 Å
    Cell angles      : α=90.0°, β=90.0°, γ=90.0°
    Unit cell volume : 257.417 Å³
    Total atoms (nat): 12
    Unique elements  : C, H, I, N, Pb  (5 species)

  ── Step 2 : Choose DFT code ──────────────────────────
  Which code?
    1) QE
    2) SIESTA
  Enter choice number [1]: 2

  ── Step 3 : Physical Interactions (vdW & SOC) ─────────
  Enable DFT-D3 dispersion correction (van der Waals)? (yes/no) [yes]: yes
  Enable Spin-Orbit Coupling (SOC)? (yes/no) [no]: no
    vdW correction (DFT-D3) : ENABLED
    Spin-Orbit Coupling     : DISABLED

  ── Step 4 : Executable path ───────────────────────────
  Path to siesta [/usr/local/bin/siesta]: [Press Enter]

  ── Step 5 : Pseudopotential directory ──────────────────
  Path to directory containing .psml OR .psf pseudopotentials: /path/to/pseudos

  Scanning for pseudopotentials for elements in CH5I3N2Pb ...
    C    ✓  C.psml                         [PSML]
    H    ✓  H.psml                         [PSML]
    I    ✓  I.psml                         [PSML]
    N    ✓  N.psml                         [PSML]
    Pb   ✓  Pb.psml                        [PSML]
  All 5 pseudopotentials found [PSML format].

  ── Step 6 : MPI / threading ────────────────────────────
  Detected CPU count: 104
  Number of MPI processes [52]: 28
  OpenMP threads per MPI task (QE only) [1]: 1

  ── Step 7 : Output directory ───────────────────────────
  Output base directory [./FAPbI3_convergence]: ./my_convergence_run

  ── Step 8 : Execution Mode ─────────────────────────────
  Choose mode:
    1) Interactive / Direct Execution (run now with smart checkpointing)
    2) Generate SLURM Batch Job Script (for cluster submission)
  Enter choice number [1]: 1

  ── Step 9 : Select stages to run ───────────────────────
    1) MeshCutoff convergence
    2) k-point mesh convergence
    3) PAO.BasisSize convergence
    4) PAO.EnergyShift convergence
    5) Lattice vs Energy (EOS scan)
    A) All stages
  Enter stage numbers separated by spaces (e.g. 1 2 3) or A for all [A]: A

  Proceed? (yes/no) [yes]: yes
```

---

## Output Structure

The runner produces organized subdirectories containing calculations, tables, figures, and production inputs:

```
my_convergence_run/
├── siesta_01_meshcutoff/
│   ├── meshcut_100Ry/
│   │   ├── input.fdf
│   │   ├── output.out
│   │   └── runner.log
│   └── results.dat             # Tabulated cutoff, energy, and Eg
├── siesta_02_kpoints/
│   └── results.dat
├── siesta_03_basissize/
│   └── results.dat
├── siesta_04_energyshift/
│   └── results.dat
├── siesta_05_eos/
│   ├── results.dat             # Volume vs Total Energy
│   └── eos_fit_curve.dat       # Birch-Murnaghan fit curve
├── convergence_summary.png     # 300 DPI 4-panel publication figure
└── production_inputs/          # Hand-off for electronic structure runs
    ├── converged_parameters.json
    ├── production.fdf (or production_scf.in)
    └── FAPbI3_relaxed_equilibrium.cif
```

---

## Theory Reference

* Refson, K. *Practical calculations using first-principles QM: Convergence, convergence, convergence*. STFC Rutherford Appleton Laboratory.
* Birch, F. (1947). *Finite elastic strain of cubic crystals*. Physical Review, 71(11), 809.
* Grimme, S., Ehrlich, S., & Goerigk, L. (2011). *Effect of the damping function in dispersion corrected density functional theory*. J. Comput. Chem., 32(7), 1456–1465.
* García, A., et al. (2018). *PSML: A file format for norm-conserving pseudopotentials*. Computer Physics Communications, 227, 51-71.
