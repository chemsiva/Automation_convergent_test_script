# Automated DFT Convergence Suite for Perovskite Materials: FAPbI₃

A unified, self-contained Python framework for executing systematic density functional theory (DFT) parameter convergence workflows and equation-of-state (EOS) determination. Supports both **Quantum ESPRESSO** (`pw.x`) and **SIESTA** (`siesta`).

The implementation follows the first-principles computational guidelines established by Keith Refson (*STFC Rutherford Appleton Laboratory*):
* Parameter sweeps employ **single-point SCF** on fixed lattices to isolate numerical convergence.
* Ground-state volume and bulk modulus determination uses a **fixed-cutoff Equation-of-State (EOS) scan** followed by 3rd-order Birch-Murnaghan fitting, eliminating Pulay stress errors inherent to variable-cell algorithms.

---

## Features

- **Dual-Engine Support**: Seamless execution of Quantum ESPRESSO and SIESTA from a single driver script.
- **Automated Pseudopotential Detection**:
  - **SIESTA**: Auto-detects XML-based **PSML** (`.psml`) and legacy ASCII **PSF** (`.psf`) formats. Prioritizes PSML and automatically falls back to PSF per element if needed.
  - **Quantum ESPRESSO**: Scans the pseudo directory and auto-selects the best UPF flavor per element (PAW `kjpaw` > USPP `rrkjus` > standard scalar-relativistic), preventing false prefix matches (e.g., distinguishing `C` from `Ce`, `H` from `Hg`, `N` from `Na`).
- **Cluster & Architecture Agnostic**: Auto-detects CPU cores, available binaries, and OpenMP/MPI settings without hardcoded paths.
- **Built-in Birch-Murnaghan Fitting**: Automatically fits $E(V)$ curves to determine the equilibrium lattice constant ($a_0$) and bulk modulus ($B_0$).

---

## Directory Structure

To run the suite, you only need the CIF structure and the master script:

```
.
├── FAPbI3.cif                   # Reference CIF structure
├── FAPbI3_perfect_cubic_300K.cif # Reference cubic CIF
├── convergence_suite.py         # Unified automation script
└── README.md                    # Documentation
```

---

## Prerequisites

1. **Python 3.8+** (standard library only; `scipy` and `matplotlib` are optional for plotting/fitting fallbacks).
2. **MPI implementation** (`mpirun`, `mpiexec`, or OpenMPI/MPICH).
3. **DFT Binaries**:
   * Quantum ESPRESSO: `pw.x`
   * SIESTA: `siesta` (versions 4.x or 5.x)
4. **Pseudopotentials**:
   * For Quantum ESPRESSO: Standard PBE UPF library (e.g., SSSP or Burai default).
   * For SIESTA: PBE `.psml` (e.g., standard `nc-sr-05_pbe`) or `.psf` files.

---

## Convergence Workflow & Stages

### 1. Quantum ESPRESSO Stages

| Stage | Swept Parameter | Range / Values | Fixed Baseline | Convergence Criterion |
|---|---|---|---|---|
| **Stage 1** | Plane-wave cutoff (`ecutwfc`) | $40 \to 100\text{ Ry}$ (step: $10\text{ Ry}$) | Dual factor $\rho = 8\times$, $3\times 3\times 3$ $k$-mesh | $|\Delta E| < 1.0\text{ meV/atom}$ |
| **Stage 2** | Monkhorst-Pack $k$-grid | $2^3 \to 7^3$ | Converged $E_{\text{cut}}$ ($70\text{ Ry}$) | $|\Delta E| < 0.5\text{ meV/atom}$ |
| **Stage 3** | EOS Volume Scan | $V/V_0 \in [0.940, 1.045]$ (22 points) | Converged $E_{\text{cut}} + k$-grid, `calc = 'relax'` | 3rd-order Birch-Murnaghan fit |

### 2. SIESTA Stages

| Stage | Swept Parameter | Range / Values | Fixed Baseline | Convergence Criterion |
|---|---|---|---|---|
| **Stage 1** | Real-space `MeshCutoff` | $100 \to 500\text{ Ry}$ | $3\times 3\times 3$ $k$-mesh, DZP, $0.02\text{ Ry}$ shift | $|\Delta E| < 1.0\text{ meV/atom}$ |
| **Stage 2** | Monkhorst-Pack $k$-grid | $2^3 \to 7^3$ | $300\text{ Ry}$ cutoff, DZP | $|\Delta E| < 0.5\text{ meV/atom}$ |
| **Stage 3** | Basis Set (`PAO.BasisSize`)| $\text{SZ} \to \text{DZ} \to \text{DZP} \to \text{TZP}$ | $300\text{ Ry}$ cutoff, $4\times 4\times 4$ $k$-mesh | $|\Delta E| < 1.0\text{ meV/atom}$ vs larger basis |
| **Stage 4** | Orbital Radius (`PAO.EnergyShift`)| $0.005 \to 0.100\text{ Ry}$ | $300\text{ Ry}$ cutoff, DZP, $4\times 4\times 4$ $k$-mesh | Energy stability plateau |
| **Stage 5** | EOS Volume Scan | $V/V_0 \in [0.940, 1.045]$ (22 points) | Converged parameters, `MD.TypeOfRun CG` | 3rd-order Birch-Murnaghan fit |

---

## Interactive Execution Guide

Start the suite in your terminal:

```bash
python3 convergence_suite.py
```

The script guides you through an interactive setup:

```
  ── Step 1 : Choose DFT code ──────────────────────────
  Options: [1] QE, [2] SIESTA
  Which code? [QE]: 2

  ── Step 2 : Executable path ───────────────────────────
  Path to siesta [/usr/local/bin/siesta]: [Press Enter]

  ── Step 3 : Pseudopotential directory ──────────────────
  Path to directory containing .psml OR .psf pseudopotentials: /path/to/pseudos

  Scanning for pseudopotentials (PSML preferred, PSF fallback) ...
    C    ✓  C.psml                         [PSML]
    H    ✓  H.psml                         [PSML]
    N    ✓  N.psml                         [PSML]
    Pb   ✓  Pb.psml                        [PSML]
    I    ✓  I.psml                         [PSML]
  All 5 pseudopotentials found [PSML format].

  ── Step 4 : MPI / threading ────────────────────────────
  Detected CPU count: 104
  Number of MPI processes [52]: 28
  OpenMP threads per MPI task (QE only) [1]: 1

  ── Step 5 : Output directory ───────────────────────────
  Output base directory [./convergence_results]: ./siesta_results

  ── Step 6 : Select stages to run ───────────────────────
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

## Output and Results

The runner creates structured subdirectories in your chosen output folder:

```
convergence_results/
├── siesta_01_meshcutoff/
│   ├── meshcut_100Ry/
│   │   ├── input.fdf
│   │   ├── output.out
│   │   └── runner.log
│   ├── ...
│   └── results.dat             # Tabulated cutoff vs energy
├── siesta_02_kpoints/
│   └── results.dat             # Tabulated k-points vs energy
├── siesta_03_basissize/
│   └── results.dat             # Tabulated basis size vs energy
├── siesta_04_energyshift/
│   └── results.dat             # Tabulated energy shift vs energy
└── siesta_05_eos/
    ├── results.dat             # Volume vs Total Energy
    └── eos_fit_curve.dat       # Birch-Murnaghan fit parameters and curve
```

Sample output table from Stage 1:
```
  Cutoff      E_tot (eV)     ΔE (meV/atom)
  ──────────────────────────────────────────
     100   -7241.12034512           0.0000
     150   -7245.89201402        -397.6391
     200   -7247.01248911         -93.3729
     250   -7247.34019283         -27.3086
     300   -7247.41029104          -5.8415
     350   -7247.42010294          -0.8177  ←
     400   -7247.42410291          -0.3333  ←
     500   -7247.42601928          -0.1597  ←
  ──────────────────────────────────────────
  Criterion: |ΔE| < 1 meV/atom from previous step (← converged)
```

---

## Pseudopotential Utilities: Converting PSML to PSF

If using an older SIESTA version (v4.0 or v4.1) that does not support `.psml`, use the official `psml2psf` utility:

```bash
# Convert a single element
psml2psf -o Pb.psf /path/to/Pb.psml

# Batch convert all 5 elements for FAPbI3
for sp in C H N Pb I; do
  psml2psf -o "${sp}.psf" "/path/to/psml_dir/${sp}.psml"
done
```

*Note: In `psml2psf`, the `-o` argument must receive a relative filename (e.g., `-o Pb.psf`) without directory slashes.*

---

## SLURM Cluster Batch Submission Example

To submit `convergence_suite.py` via SLURM:

```bash
#!/bin/bash
#SBATCH -J FAPbI3_conv
#SBATCH -o conv_%j.out
#SBATCH -e conv_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=52
#SBATCH --time=48:00:00

# Load environment
export OMP_NUM_THREADS=1

# Execute non-interactively using predefined inputs via standard input redirection
python3 convergence_suite.py << 'EOF'
2
/usr/local/bin/siesta
/scratch/SIVAKUMAR/nc-sr-05_pbe_standard_psml/psf_outputs
52
./siesta_convergence_run
A
yes
EOF
```

---

## Theory Reference

* Refson, K. *Practical calculations using first-principles QM: Convergence, convergence, convergence*. STFC Rutherford Appleton Laboratory.
* Birch, F. (1947). *Finite elastic strain of cubic crystals*. Physical Review, 71(11), 809.
* García, A., et al. (2018). *PSML: A file format for norm-conserving pseudopotentials*. Computer Physics Communications, 227, 51-71.
