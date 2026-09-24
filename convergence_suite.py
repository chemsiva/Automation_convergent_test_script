#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         DFT CONVERGENCE SUITE  –  Universal Crystal Edition      ║
║         Supports: Quantum ESPRESSO  &  SIESTA                   ║
║         Version : 3.0   (Generic CIF-driven)                     ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python3 convergence_suite.py

The script interactively:
  1. Detects or prompts for any .cif structure file
  2. Parses cell, species, coordinates, and atomic properties automatically
  3. Prompts for DFT code (Quantum ESPRESSO or SIESTA)
  4. Auto-detects binaries and pseudopotentials for all elements in the CIF
  5. Configures MPI / OpenMP processes
  6. Executes parameter sweeps and Birch-Murnaghan Equation of State (EOS)
"""

import os
import sys
import math
import shutil
import subprocess
import time
import glob
import re

# ══════════════════════════════════════════════════════════════════
#  PERIODIC TABLE DATA  (Z and atomic mass in amu)
# ══════════════════════════════════════════════════════════════════
PERIODIC_TABLE = {
    "H":  (1,   1.008),   "He": (2,   4.0026),
    "Li": (3,   6.94),    "Be": (4,   9.0122),  "B":  (5,  10.81),   "C":  (6,  12.011),
    "N":  (7,  14.007),   "O":  (8,  15.999),   "F":  (9,  18.998),  "Ne": (10,  20.180),
    "Na": (11, 22.990),   "Mg": (12, 24.305),   "Al": (13, 26.982),  "Si": (14,  28.085),
    "P":  (15, 30.974),   "S":  (16, 32.06),    "Cl": (17, 35.45),   "Ar": (18,  39.948),
    "K":  (19, 39.098),   "Ca": (20, 40.078),   "Sc": (21, 44.956),  "Ti": (22,  47.867),
    "V":  (23, 50.942),   "Cr": (24, 51.996),   "Mn": (25, 54.938),  "Fe": (26,  55.845),
    "Co": (27, 58.933),   "Ni": (28, 58.693),   "Cu": (29, 63.546),  "Zn": (30,  65.38),
    "Ga": (31, 69.723),   "Ge": (32, 72.630),   "As": (33, 74.922),  "Se": (34,  78.971),
    "Br": (35, 79.904),   "Kr": (36, 83.798),   "Rb": (37, 85.468),  "Sr": (38,  87.62),
    "Y":  (39, 88.906),   "Zr": (40, 91.224),   "Nb": (41, 92.906),  "Mo": (42,  95.95),
    "Tc": (43, 98.0),     "Ru": (44, 101.07),   "Rh": (45, 102.91),  "Pd": (46, 106.42),
    "Ag": (47, 107.87),   "Cd": (48, 112.41),   "In": (49, 114.82),  "Sn": (50, 118.71),
    "Sb": (51, 121.76),   "Te": (52, 127.60),   "I":  (53, 126.904), "Xe": (54, 131.29),
    "Cs": (55, 132.91),   "Ba": (56, 137.33),   "La": (57, 138.91),  "Ce": (58, 140.12),
    "Pr": (59, 140.91),   "Nd": (60, 144.24),   "Pm": (61, 145.0),   "Sm": (62, 150.36),
    "Eu": (63, 151.96),   "Gd": (64, 157.25),   "Tb": (65, 158.93),  "Dy": (66, 162.50),
    "Ho": (67, 164.93),   "Er": (68, 167.26),   "Tm": (69, 168.93),  "Yb": (70, 173.05),
    "Lu": (71, 174.97),   "Hf": (72, 178.49),   "Ta": (73, 180.95),  "W":  (74, 183.84),
    "Re": (75, 186.21),   "Os": (76, 190.23),   "Ir": (77, 192.22),  "Pt": (78, 195.08),
    "Au": (79, 196.97),   "Hg": (80, 200.59),   "Tl": (81, 204.38),  "Pb": (82, 207.2),
    "Bi": (83, 208.98),   "Po": (84, 209.0),    "At": (85, 210.0),   "Rn": (86, 222.0),
    "Fr": (87, 223.0),    "Ra": (88, 226.0),    "Ac": (89, 227.0),   "Th": (90, 232.04),
    "Pa": (91, 231.04),   "U":  (92, 238.03)
}

# ══════════════════════════════════════════════════════════════════
#  STRUCTURE CLASS  (Generic CIF Parser & Exporter)
# ══════════════════════════════════════════════════════════════════
class Structure:
    """Represents a periodic crystal structure loaded from a CIF file."""

    def __init__(self, cif_path):
        self.cif_path = os.path.abspath(cif_path)
        self.filename = os.path.basename(cif_path)
        self.prefix = os.path.splitext(self.filename)[0].replace(" ", "_")
        self.load(self.cif_path)

    def load(self, path):
        # Try loading via ASE first if available
        try:
            import ase.io
            atoms = ase.io.read(path)
            self.cell = [list(row) for row in atoms.cell[:]]
            self.cellpar = list(atoms.cell.cellpar())
            self.volume = float(atoms.get_volume())
            self.symbols = atoms.get_chemical_symbols()
            self.positions_frac = [list(p) for p in atoms.get_scaled_positions()]
            self.formula = atoms.get_chemical_formula()
        except Exception:
            # Standalone fallback parser
            self._load_fallback(path)

        # Post-process species & indices
        self.species = sorted(list(set(self.symbols)))
        self.nat = len(self.symbols)
        self.ntyp = len(self.species)
        self.sp_idx = {sp: i + 1 for i, sp in enumerate(self.species)}

        # Fallback formula string if needed
        if not hasattr(self, 'formula') or not self.formula:
            counts = {sp: self.symbols.count(sp) for sp in self.species}
            self.formula = "".join(f"{sp}{counts[sp] if counts[sp] > 1 else ''}" for sp in self.species)

    def _load_fallback(self, path):
        """Pure-Python CIF reader without external dependencies."""
        with open(path) as f:
            content = f.read()

        def get_tag(tag, default=0.0):
            m = re.search(rf'^{tag}\s+([0-9\.\-\+eE]+)', content, re.MULTILINE)
            if m:
                val = re.sub(r'\(.*?\)', '', m.group(1))
                return float(val)
            return default

        a = get_tag('_cell_length_a', 1.0)
        b = get_tag('_cell_length_b', 1.0)
        c = get_tag('_cell_length_c', 1.0)
        alpha = get_tag('_cell_angle_alpha', 90.0)
        beta  = get_tag('_cell_angle_beta',  90.0)
        gamma = get_tag('_cell_angle_gamma', 90.0)
        self.cellpar = [a, b, c, alpha, beta, gamma]

        # Convert parameters to 3x3 lattice vectors
        ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
        ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
        sg = math.sin(gr)
        val = 1.0 + 2.0 * ca * cb * cg - ca*ca - cb*cb - cg*cg
        V_factor = math.sqrt(max(0.0, val))

        v1 = [a, 0.0, 0.0]
        v2 = [b * cg, b * sg, 0.0]
        v3 = [c * cb, c * (ca - cb*cg) / (sg if abs(sg) > 1e-8 else 1.0),
              c * V_factor / (sg if abs(sg) > 1e-8 else 1.0)]
        self.cell = [v1, v2, v3]
        self.volume = a * b * c * V_factor

        # Parse atom site loop
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith('#')]
        headers = []
        data_rows = []
        i = 0
        while i < len(lines):
            if lines[i] == 'loop_':
                j = i + 1
                curr_headers = []
                while j < len(lines) and lines[j].startswith('_atom_site_'):
                    curr_headers.append(lines[j])
                    j += 1
                if curr_headers:
                    while j < len(lines) and not lines[j].startswith('_') and lines[j] != 'loop_':
                        data_rows.append(lines[j].split())
                        j += 1
                    headers = curr_headers
                    break
            i += 1

        x_idx = headers.index('_atom_site_fract_x')
        y_idx = headers.index('_atom_site_fract_y')
        z_idx = headers.index('_atom_site_fract_z')
        sym_idx = headers.index('_atom_site_type_symbol') if '_atom_site_type_symbol' in headers else headers.index('_atom_site_label')

        symbols = []
        positions = []
        for row in data_rows:
            raw_sym = re.sub(r'[^A-Za-z]', '', row[sym_idx]).capitalize()
            symbols.append(raw_sym)
            x = float(re.sub(r'\(.*?\)', '', row[x_idx]))
            y = float(re.sub(r'\(.*?\)', '', row[y_idx]))
            z = float(re.sub(r'\(.*?\)', '', row[z_idx]))
            positions.append([x, y, z])

        self.symbols = symbols
        self.positions_frac = positions
        counts = {sp: symbols.count(sp) for sp in sorted(list(set(symbols)))}
        self.formula = "".join(f"{sp}{counts[sp] if counts[sp] > 1 else ''}" for sp in sorted(counts.keys()))

    def get_scaled_cell(self, scale):
        """Returns 3x3 cell multiplied by isotropic linear factor 'scale'."""
        return [[val * scale for val in row] for row in self.cell]

    # ── Quantum ESPRESSO Exporters ──
    def qe_cell_parameters(self, scale=1.0):
        c = self.get_scaled_cell(scale)
        lines = ["CELL_PARAMETERS {angstrom}"]
        for row in c:
            lines.append(f"  {row[0]:15.9f}  {row[1]:15.9f}  {row[2]:15.9f}")
        return "\n".join(lines)

    def qe_atomic_species(self, pseudo_dir):
        lines = ["ATOMIC_SPECIES"]
        for sp in self.species:
            z, mass = PERIODIC_TABLE.get(sp, (0, 1.0))
            upf = find_qe_upf(pseudo_dir, sp)
            lines.append(f"  {sp:<4} {mass:>9.3f}  {upf}")
        return "\n".join(lines)

    def qe_atomic_positions(self):
        lines = ["ATOMIC_POSITIONS {crystal}"]
        for sym, (x, y, z) in zip(self.symbols, self.positions_frac):
            lines.append(f"  {sym:<4}  {x:12.8f}  {y:12.8f}  {z:12.8f}")
        return "\n".join(lines)

    # ── SIESTA Exporters ──
    def sia_lattice_block(self, scale=1.0):
        c = self.get_scaled_cell(scale)
        lines = [
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors"
        ]
        for row in c:
            lines.append(f"  {row[0]:15.9f}  {row[1]:15.9f}  {row[2]:15.9f}")
        lines.append("%endblock LatticeVectors")
        return "\n".join(lines)

    def sia_species_block(self, pseudo_dir):
        lines = ["%block ChemicalSpeciesLabel"]
        for sp in self.species:
            z, _ = PERIODIC_TABLE.get(sp, (0, 1.0))
            idx = self.sp_idx[sp]
            lines.append(f"  {idx}  {z:3d}  {sp}")
        lines.append("%endblock ChemicalSpeciesLabel")
        return "\n".join(lines)

    def sia_coords_block(self):
        lines = [
            "AtomicCoordinatesFormat Fractional",
            "%block AtomicCoordinatesAndAtomicSpecies"
        ]
        for sym, (x, y, z) in zip(self.symbols, self.positions_frac):
            idx = self.sp_idx[sym]
            lines.append(f"  {x:12.8f}  {y:12.8f}  {z:12.8f}  {idx}  # {sym}")
        lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
        return "\n".join(lines)

    def export_cif(self, out_path, a, b, c, alpha=None, beta=None, gamma=None):
        """Export equilibrium structure to a standardized CIF file."""
        if alpha is None: alpha = self.cellpar[3]
        if beta  is None: beta  = self.cellpar[4]
        if gamma is None: gamma = self.cellpar[5]

        with open(out_path, "w") as f:
            f.write(f"# CIF generated by DFT Convergence Suite for {self.formula}\n")
            f.write(f"data_{self.prefix}_equilibrium\n\n")
            f.write(f"_cell_length_a  {a:.6f}\n")
            f.write(f"_cell_length_b  {b:.6f}\n")
            f.write(f"_cell_length_c  {c:.6f}\n")
            f.write(f"_cell_angle_alpha  {alpha:.4f}\n")
            f.write(f"_cell_angle_beta   {beta:.4f}\n")
            f.write(f"_cell_angle_gamma  {gamma:.4f}\n")
            f.write("_symmetry_space_group_name_H-M  'P 1'\n")
            f.write("_symmetry_Int_Tables_number  1\n\n")
            f.write("loop_\n")
            f.write("  _symmetry_equiv_pos_as_xyz\n")
            f.write("  'x, y, z'\n\n")
            f.write("loop_\n")
            f.write("  _atom_site_label\n")
            f.write("  _atom_site_type_symbol\n")
            f.write("  _atom_site_fract_x\n")
            f.write("  _atom_site_fract_y\n")
            f.write("  _atom_site_fract_z\n")
            for idx, (sym, (x, y, z)) in enumerate(zip(self.symbols, self.positions_frac), 1):
                f.write(f"  {sym}{idx:<3} {sym:<2}  {x:10.6f}  {y:10.6f}  {z:10.6f}\n")


# ══════════════════════════════════════════════════════════════════
#  DEFAULT CONVERGENCE RANGES
# ══════════════════════════════════════════════════════════════════
QE_ECUT_LIST    = [40, 50, 60, 70, 80, 90, 100]       # Ry
QE_ECUT_DUAL    = 8                                   # ecutrho = dual * ecutwfc
QE_KGRIDS       = [(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6),(7,7,7)]
QE_ECUT_4KTEST  = 50                                  # Ry
QE_KGRID_4ETEST = (3,3,3)
QE_ECUT_4EOS    = 70                                  # Ry
QE_KGRID_4EOS   = (4,4,4)

SIA_MESHCUT     = [100, 150, 200, 250, 300, 350, 400, 500]  # Ry
SIA_KGRIDS      = [(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6),(7,7,7)]
SIA_BASIS       = ["SZ", "DZ", "DZP", "TZP"]
SIA_ESHIFT      = ["0.005", "0.010", "0.020", "0.050", "0.100"]  # Ry
SIA_MC_4KTEST   = 300
SIA_KG_4MTEST   = (3,3,3)
SIA_KG_4BASIS   = (4,4,4)
SIA_KG_4ESHIFT  = (4,4,4)
SIA_MC_4EOS     = 300
SIA_KG_4EOS     = (4,4,4)
SIA_BASIS_4EOS  = "DZP"
SIA_ES_4EOS     = "0.010"
SIA_ESHIFT_FIXED = "0.020"
SIA_BASIS_FIXED  = "DZP"

# EOS volume fractions: V/V0 in [0.940 ... 1.045]
EOS_FRACS = [round(0.940 + i * 0.005, 3) for i in range(22)]
EOS_CG_STEPS = 2000

# ══════════════════════════════════════════════════════════════════
#  UTILITIES & UI
# ══════════════════════════════════════════════════════════════════
RY2EV = 13.605693122994
EV2MEV = 1000.0

COLORS = {
    "header": "\033[1;36m", "ok": "\033[1;32m",
    "warn":   "\033[1;33m", "err":"\033[1;31m",
    "reset":  "\033[0m",    "bold":"\033[1m",
}

def c(tag, text):
    if sys.stdout.isatty():
        return f"{COLORS[tag]}{text}{COLORS['reset']}"
    return text

def ask(prompt, default=""):
    if default:
        ans = input(f"  {prompt} [{default}]: ").strip()
        return ans if ans else default
    return input(f"  {prompt}: ").strip()

def ask_choice(prompt, options, default=0):
    print(f"  {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"    {i}) {opt}")
    while True:
        raw = ask("Enter choice number", str(default + 1))
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(c("warn", "  Invalid selection; try again."))

def ask_int(prompt, default):
    while True:
        raw = ask(prompt, str(default))
        try:
            v = int(raw)
            if v > 0: return v
        except ValueError:
            pass
        print(c("warn", "  Please enter a positive integer."))

def detect_exe(names):
    for name in names:
        path = shutil.which(name)
        if path: return path
    return ""

def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

def run_cmd(cmd, logfile):
    t0 = time.time()
    with open(logfile, "w") as lf:
        subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)
    return round(time.time() - t0)

def print_header(title):
    w = 64
    print()
    print(c("header", "═" * w))
    print(c("header", f"  {title}"))
    print(c("header", "═" * w))

# ══════════════════════════════════════════════════════════════════
#  PSEUDOPOTENTIAL DISCOVERY & VALIDATION
# ══════════════════════════════════════════════════════════════════
def find_qe_upf(pseudo_dir, species):
    """Find best matching UPF for species, preventing false prefix matches."""
    raw = (
        glob.glob(os.path.join(pseudo_dir, f"{species}*.UPF")) +
        glob.glob(os.path.join(pseudo_dir, f"{species}*.upf"))
    )
    pat = re.compile(rf"^{re.escape(species)}[^A-Za-z]", re.IGNORECASE)
    candidates = [p for p in raw if pat.match(os.path.basename(p))]
    if not candidates:
        raise FileNotFoundError(
            f"No UPF found for element '{species}' in {pseudo_dir}")
    def rank(p):
        b = os.path.basename(p).lower()
        return 0 if "kjpaw" in b else (1 if "rrkjus" in b else 2)
    return os.path.basename(sorted(candidates, key=rank)[0])

def detect_sia_pseudo_format(pseudo_dir, species_list):
    """Scan pseudo_dir for .psml and .psf for the given species."""
    found = {}
    for sp in species_list:
        pat = re.compile(rf"^{re.escape(sp)}[^A-Za-z]", re.IGNORECASE)
        # Search PSML first
        psml_matches = [
            f for f in glob.glob(os.path.join(pseudo_dir, "*.[pP][sS][mM][lL]"))
            if pat.match(os.path.basename(f))
        ]
        if psml_matches:
            exact = [f for f in psml_matches if os.path.basename(f).lower() == f"{sp.lower()}.psml"]
            found[sp] = (exact[0] if exact else psml_matches[0], ".psml")
            continue
        # Fallback to PSF
        psf_matches = [
            f for f in glob.glob(os.path.join(pseudo_dir, "*.[pP][sS][fF]"))
            if pat.match(os.path.basename(f))
        ]
        if psf_matches:
            exact = [f for f in psf_matches if os.path.basename(f).lower() == f"{sp.lower()}.psf"]
            found[sp] = (exact[0] if exact else psf_matches[0], ".psf")

    missing = [sp for sp in species_list if sp not in found]
    exts = set(ext for _, ext in found.values())
    if not exts:
        fmt = "NONE"
    elif exts == {".psml"}:
        fmt = "PSML"
    elif exts == {".psf"}:
        fmt = "PSF"
    else:
        fmt = "MIXED"
    return fmt, found, missing

def sia_copy_pseudos(dest, pseudo_dir, species_list):
    fmt, found, missing = detect_sia_pseudo_format(pseudo_dir, species_list)
    os.makedirs(dest, exist_ok=True)
    if missing:
        print(c("err", f"\n  [SIESTA] Missing pseudopotentials for: {missing}"))
    for sp, (src, ext) in found.items():
        dst = os.path.join(dest, sp + ext)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

# ══════════════════════════════════════════════════════════════════
#  INPUT BUILDERS
# ══════════════════════════════════════════════════════════════════
def qe_kpoints(kg):
    return f"K_POINTS {{automatic}}\n  {kg[0]} {kg[1]} {kg[2]}  0 0 0"

def qe_input(pseudo_dir, struct, ecutwfc, kg, scale=1.0, calc="scf", vdw=True, soc=False):
    ecutrho = QE_ECUT_DUAL * ecutwfc
    ions = "\n&IONS\n    ion_dynamics = 'bfgs'\n/" if calc == "relax" else ""
    vdw_block = """\
    vdw_corr          = 'dft-d3'
    dftd3_version     = 4
    dftd3_threebody   = .true.
""" if vdw else ""
    soc_block = """\
    noncolin          = .true.
    lspinorb          = .true.
""" if soc else ""
    return f"""\
&CONTROL
    calculation   = '{calc}'
    prefix        = '{struct.prefix}'
    pseudo_dir    = '{pseudo_dir}'
    outdir        = './out'
    tstress       = .true.
    tprnfor       = .true.
    verbosity     = 'medium'
    max_seconds   = 86400
/
&SYSTEM
    ibrav         = 0
    nat           = {struct.nat}
    ntyp          = {struct.ntyp}
    ecutwfc       = {ecutwfc}
    ecutrho       = {ecutrho}
    degauss       = 0.01
    occupations   = 'smearing'
    smearing      = 'gaussian'
{vdw_block}{soc_block}/
&ELECTRONS
    conv_thr         = 1.0d-8
    mixing_beta      = 0.40
    electron_maxstep = 200
    diagonalization  = 'david'
    diago_david_ndim = 4
/{ions}
{struct.qe_cell_parameters(scale)}
{struct.qe_atomic_species(pseudo_dir)}
{struct.qe_atomic_positions()}
{qe_kpoints(kg)}
"""

def qe_npool(kgrid):
    n_kpts = kgrid[0] * kgrid[1] * kgrid[2]
    for p in (8, 4, 2, 1):
        if n_kpts % p == 0:
            return p
    return 1

def qe_mpi_cmd(exe, procs, omp, npool, inp, out):
    pool_flag = f"-nk {npool}" if npool > 1 else ""
    return f"OMP_NUM_THREADS={omp} mpirun -np {procs} {exe} {pool_flag} -in {inp} > {out} 2>&1"

def sia_input(pseudo_dir, struct, meshcut, kg, basis, eshift, scale=1.0, cg_steps=0, vdw=True, soc=False):
    geo_relax = f"""\
MD.TypeOfRun            CG
MD.NumCGsteps           {cg_steps}
MD.MaxForceTol          0.01 eV/Ang
MD.MaxStressTol         0.50 kBar
MD.VariableCell         F
""" if cg_steps > 0 else "MD.NumCGsteps           0"

    vdw_block = """\
DFT.D3                  true
DFT.D3.version          4
DFT.D3.damping          BJ
DFT.D3.threebody        true
""" if vdw else "DFT.D3                  false"

    soc_block = """\
Spin.Type               spin-orbit
""" if soc else """\
SpinPolarized           F
"""

    return f"""\
SystemName              {struct.prefix}
SystemLabel             {struct.prefix}
NumberOfAtoms           {struct.nat}
NumberOfSpecies         {struct.ntyp}

{struct.sia_species_block(pseudo_dir)}

{struct.sia_lattice_block(scale)}

{struct.sia_coords_block()}

XC.functional           GGA
XC.authors              PBE

{vdw_block}
{soc_block}
MeshCutoff              {meshcut} Ry
PAO.BasisType           split
PAO.BasisSize           {basis}
PAO.EnergyShift         {eshift} Ry

%block kgrid_Monkhorst_Pack
  {kg[0]}  0  0  0.0
  0  {kg[1]}  0  0.0
  0  0  {kg[2]}  0.0
%endblock kgrid_Monkhorst_Pack

SolutionMethod          diagon
MaxSCFIterations        300
ElectronicTemperature   300 K
SCF.MustConverge        true
DM.MixingWeight         0.10
DM.NumberPulay          5
DM.Tolerance            1.0d-5
DM.Require.Energy.Convergence T
DM.Energy.Tolerance     1.0d-5 eV
DM.UseSaveDM            F

# ── Geometry Relaxation ──────────────────────────────────────────
{geo_relax}

# ── Output ───────────────────────────────────────────────────────
SpinPolarized           F
WriteForces             T
WriteMullikenPop        1
XML.Write               T
SaveHS                  F
SaveRHO                 F
"""

def sia_mpi_cmd(exe, procs, fdf, out, err):
    return f"mpirun -np {procs} {exe} < {fdf} > {out} 2> {err}"

# ══════════════════════════════════════════════════════════════════
#  PARSERS
# ══════════════════════════════════════════════════════════════════
def qe_parse_energy(outfile):
    energy = None
    try:
        with open(outfile) as f:
            for line in f:
                if line.strip().startswith("!") and "total energy" in line:
                    energy = float(line.split()[-2])
    except FileNotFoundError:
        pass
    return energy

def qe_job_done(outfile):
    try:
        return any("JOB DONE" in l for l in open(outfile))
    except FileNotFoundError:
        return False

def sia_parse_energy(outfile):
    energy = None
    try:
        with open(outfile) as f:
            for line in f:
                if "siesta:         Total =" in line:
                    try:
                        energy = float(line.split("=")[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
    except FileNotFoundError:
        pass
    return energy

def sia_job_done(outfile):
    try:
        return any("Job completed" in l or "siesta: Final energy" in l
                   for l in open(outfile))
    except FileNotFoundError:
        return False

# ══════════════════════════════════════════════════════════════════
#  ANALYSIS & TABLES
# ══════════════════════════════════════════════════════════════════
def print_table(rows, headers, col_widths, criterion_note=""):
    sep = "  ".join(f"{'─'*w}" for w in col_widths)
    hdr = "  ".join(f"{h:>{w}}" for h, w in zip(headers, col_widths))
    print(f"\n  {c('bold', hdr)}")
    print(f"  {sep}")
    for row in rows:
        line = "  ".join(f"{v:>{w}}" for v, w in zip(row, col_widths))
        if "←" in row[-1]:
            print(f"  {c('ok', line)}")
        else:
            print(f"  {line}")
    print(f"  {sep}")
    if criterion_note:
        print(f"\n  {c('warn', criterion_note)}\n")

def summarise_ecut(data, unit_factor, nat):
    rows = []
    prev = None
    for param, e in data:
        de = (e - prev) * unit_factor / nat if prev is not None else 0.0
        flag = "  ←" if abs(de) < 1.0 and prev is not None else ""
        rows.append((str(param), f"{e:.8f}", f"{de:.4f}{flag}"))
        prev = e
    print_table(rows, ["Cutoff", "E_tot", "ΔE (meV/atom)"], [12, 18, 16],
                "Criterion: |ΔE| < 1 meV/atom from previous step  (← converged)")

def summarise_kpoints(data, nat):
    rows = []
    prev = None
    for kg_str, e in data:
        de = (e - prev) * EV2MEV / nat if prev is not None else 0.0
        flag = "  ←" if abs(de) < 0.5 and prev is not None else ""
        rows.append((kg_str, f"{e:.8f}", f"{de:.4f}{flag}"))
        prev = e
    print_table(rows, ["k-mesh", "E_tot (eV)", "ΔE (meV/atom)"], [12, 18, 16],
                "Criterion: |ΔE| < 0.5 meV/atom  (← converged)")

def summarise_basissize(data, nat):
    ORDER = {"SZ": 0, "DZ": 1, "DZP": 2, "TZP": 3}
    data_s = sorted(data, key=lambda r: ORDER.get(r[0], 99))
    comp = {"DZ": "vs SZ", "DZP": "vs DZ", "TZP": "vs DZP"}
    rows = []
    prev = None
    for bs, e in data_s:
        de = (e - prev) * EV2MEV / nat if prev is not None else 0.0
        flag = "  ←" if abs(de) < 1.0 and prev is not None else ""
        rows.append((bs, comp.get(bs, ""), f"{e:.8f}", f"{de:.4f}{flag}"))
        prev = e
    print_table(rows, ["BasisSize", "Comparison", "E_tot (eV)", "ΔE (meV/atom)"],
                [10, 12, 18, 16],
                "Criterion: |ΔE| < 1 meV/atom vs next-larger basis  (← converged)")

def summarise_energyshift(data, nat):
    rows = []
    prev = None
    for es_str, e in data:
        de = (e - prev) * EV2MEV / nat if prev is not None else 0.0
        flag = "  ←" if abs(de) < 1.0 and prev is not None else ""
        rows.append((es_str, f"{e:.8f}", f"{de:.4f}{flag}"))
        prev = e
    print_table(rows, ["EnergyShift", "E_tot (eV)", "ΔE (meV/atom)"], [14, 18, 16],
                "EnergyShift determines orbital cutoff radius (lower = larger basis)")

def summarise_eos(data):
    # data: (frac, scale, vol, energy_eV)
    e_vals = [r[3] for r in data]
    e_min  = min(e_vals)
    rows = []
    for frac, scale, vol, e in data:
        de_rel = (e - e_min) * EV2MEV
        star = "  ★ min" if abs(e - e_min) < 1e-6 else ""
        rows.append((f"{frac:.3f}", f"{scale:.5f}", f"{vol:.3f}", f"{e:.6f}", f"{de_rel:.2f}{star}"))
    print_table(rows,
                ["V/V0", "Scale", "Vol (Å³)", "E_tot (eV)", "ΔE_rel (meV)"],
                [8, 10, 12, 16, 14],
                "Fixed-cutoff EOS scan: eliminates Pulay stress artifacts")

def bm_eos(V, E0, B0, B0_prime, V0):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0) ** 3 * B0_prime +
        (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )

def bm_fit(data, outdir, struct):
    # data: (frac, scale, vol, E_eV)
    vols = [r[2] for r in data]
    e_ev = [r[3] for r in data]
    if len(vols) < 5:
        print("  Not enough points for Birch-Murnaghan fit.")
        return

    EV_PER_A3_TO_GPA = 160.21766208
    min_idx = e_ev.index(min(e_ev))
    V0_guess = vols[min_idx]
    E0_guess = e_ev[min_idx]
    B0_guess_gpa = 30.0
    B0_guess = B0_guess_gpa / EV_PER_A3_TO_GPA

    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(
            bm_eos, vols, e_ev,
            p0=[E0_guess, B0_guess, 4.0, V0_guess],
            bounds=([-math.inf, 0.0, 1.0, min(vols)],
                    [math.inf,  5.0, 10.0, max(vols)]),
            maxfev=10000
        )
        E0_fit, B0_fit, B0p_fit, V0_fit = popt
        B0_gpa = B0_fit * EV_PER_A3_TO_GPA
        fit_ok = True
    except Exception:
        # Parabolic fallback
        v_min = vols[min_idx]
        e_min = e_ev[min_idx]
        if 0 < min_idx < len(vols) - 1:
            dv = vols[min_idx+1] - vols[min_idx-1]
            d2e = (e_ev[min_idx+1] - 2*e_min + e_ev[min_idx-1]) / ((dv/2)**2)
            B0_gpa = v_min * d2e * EV_PER_A3_TO_GPA
        else:
            B0_gpa = 0.0
        E0_fit, B0_fit, B0p_fit, V0_fit = e_min, B0_gpa/EV_PER_A3_TO_GPA, 4.0, v_min
        fit_ok = False

    scale_fit = (V0_fit / struct.volume) ** (1.0 / 3.0)
    a_fit = struct.cellpar[0] * scale_fit
    b_fit = struct.cellpar[1] * scale_fit
    c_fit = struct.cellpar[2] * scale_fit

    method = "Birch-Murnaghan 3rd-order fit" if fit_ok else "Parabolic estimate (fallback)"
    print()
    print(c("bold", f"  ─── Equation of State Fit Results ({method}) ───"))
    print(f"  Equilibrium volume (V0)  : {V0_fit:.4f} Å³  (ref: {struct.volume:.4f} Å³)")
    print(f"  Equilibrium cell lengths : a = {a_fit:.4f} Å, b = {b_fit:.4f} Å, c = {c_fit:.4f} Å")
    print(f"  Ground-state energy (E0) : {E0_fit:.6f} eV")
    print(f"  Bulk Modulus (B0)        : {B0_gpa:.2f} GPa")
    print(f"  Pressure derivative (B0'): {B0p_fit:.2f}")

    # Write fit curve
    fit_path = os.path.join(outdir, "eos_fit_curve.dat")
    v_dense = [min(vols) + i*(max(vols)-min(vols))/200 for i in range(201)]
    with open(fit_path, "w") as f:
        f.write(f"# Birch-Murnaghan 3rd-order EOS fit for {struct.formula}\n")
        f.write(f"# V0 = {V0_fit:.6f} A3 | B0 = {B0_gpa:.4f} GPa | B0_prime = {B0p_fit:.4f}\n")
        f.write(f"# Equilibrium cell: a={a_fit:.5f} b={b_fit:.5f} c={c_fit:.5f} Ang\n")
        f.write("# V_A3      E_fit_eV\n")
        for v in v_dense:
            f.write(f"  {v:.6f}  {bm_eos(v, E0_fit, B0_fit, B0p_fit, V0_fit):.8f}\n")
    print(c("ok", f"\n  ✓ Fit parameters and curve saved to: {fit_path}"))

# ══════════════════════════════════════════════════════════════════
#  STAGE RUNNERS  –  QUANTUM ESPRESSO
# ══════════════════════════════════════════════════════════════════
def qe_run_stage1(cfg):
    exe, procs, omp, pseudo_dir, base, struct = (
        cfg["exe"], cfg["procs"], cfg["omp"],
        cfg["pseudo_dir"], cfg["base"], cfg["struct"])
    stage_dir = os.path.join(base, "qe_01_ecut")
    print_header(f"QE Stage 1 – Plane-wave cutoff convergence ({struct.formula})")
    print(f"  k-mesh fixed : {QE_KGRID_4ETEST}  |  Dual (ecutrho/ecutwfc): {QE_ECUT_DUAL}x")

    results = []
    for ecut in QE_ECUT_LIST:
        subdir = os.path.join(stage_dir, f"ecut_{ecut}Ry")
        inp = os.path.join(subdir, "input.in")
        out = os.path.join(subdir, "output.out")
        os.makedirs(os.path.join(subdir, "out"), exist_ok=True)
        write_file(inp, qe_input(pseudo_dir, struct, ecut, QE_KGRID_4ETEST))
        cmd = qe_mpi_cmd(exe, procs, omp, qe_npool(QE_KGRID_4ETEST), "input.in", "output.out")
        print(f"  → ecutwfc = {ecut:3d} Ry ... ", end="", flush=True)
        t = run_cmd(f"cd {subdir} && {cmd}", os.path.join(subdir, "runner.log"))
        if qe_job_done(out):
            e_ry = qe_parse_energy(out)
            results.append((ecut, e_ry))
            print(c("ok", f"✓  E = {e_ry:.8f} Ry  ({t}s)"))
        else:
            print(c("err", "✗ FAILED"))

    if results:
        print_header("QE Stage 1 – Results")
        summarise_ecut(results, RY2EV * EV2MEV, struct.nat)
        _save_dat(os.path.join(stage_dir, "results.dat"), ["ecut_Ry", "E_Ry"], results)
    return results

def qe_run_stage2(cfg):
    exe, procs, omp, pseudo_dir, base, struct = (
        cfg["exe"], cfg["procs"], cfg["omp"],
        cfg["pseudo_dir"], cfg["base"], cfg["struct"])
    stage_dir = os.path.join(base, "qe_02_kpoints")
    print_header(f"QE Stage 2 – k-point mesh convergence ({struct.formula})")
    print(f"  ecutwfc fixed: {QE_ECUT_4KTEST} Ry")

    results = []
    for kg in QE_KGRIDS:
        kg_str = "x".join(str(k) for k in kg)
        subdir = os.path.join(stage_dir, f"kgrid_{kg_str}")
        inp = os.path.join(subdir, "input.in")
        out = os.path.join(subdir, "output.out")
        os.makedirs(os.path.join(subdir, "out"), exist_ok=True)
        write_file(inp, qe_input(pseudo_dir, struct, QE_ECUT_4KTEST, kg))
        cmd = qe_mpi_cmd(exe, procs, omp, qe_npool(kg), "input.in", "output.out")
        print(f"  → k-mesh {kg_str} ... ", end="", flush=True)
        t = run_cmd(f"cd {subdir} && {cmd}", os.path.join(subdir, "runner.log"))
        if qe_job_done(out):
            e_ev = qe_parse_energy(out) * RY2EV
            results.append((kg_str, e_ev))
            print(c("ok", f"✓  E = {e_ev:.8f} eV  ({t}s)"))
        else:
            print(c("err", "✗ FAILED"))

    if results:
        print_header("QE Stage 2 – Results")
        summarise_kpoints(results, struct.nat)
        _save_dat(os.path.join(stage_dir, "results.dat"), ["k_mesh", "E_eV"], results)
    return results

def qe_run_stage3(cfg):
    exe, procs, omp, pseudo_dir, base, struct = (
        cfg["exe"], cfg["procs"], cfg["omp"],
        cfg["pseudo_dir"], cfg["base"], cfg["struct"])
    stage_dir = os.path.join(base, "qe_03_eos")
    print_header(f"QE Stage 3 – Equation of State Scan ({struct.formula})")
    print(f"  ecutwfc = {QE_ECUT_4EOS} Ry  |  k-mesh = {QE_KGRID_4EOS}  |  calc = relax")

    results = []
    for frac in EOS_FRACS:
        scale = frac ** (1.0 / 3.0)
        vol = struct.volume * frac
        label = f"vol_{frac:.3f}"
        subdir = os.path.join(stage_dir, label)
        inp = os.path.join(subdir, "input.in")
        out = os.path.join(subdir, "output.out")
        os.makedirs(os.path.join(subdir, "out"), exist_ok=True)
        write_file(inp, qe_input(pseudo_dir, struct, QE_ECUT_4EOS, QE_KGRID_4EOS,
                                scale=scale, calc="relax"))
        cmd = qe_mpi_cmd(exe, procs, omp, qe_npool(QE_KGRID_4EOS), "input.in", "output.out")
        print(f"  → frac={frac:.3f}  V={vol:.2f} Å³ ... ", end="", flush=True)
        t = run_cmd(f"cd {subdir} && {cmd}", os.path.join(subdir, "runner.log"))
        if qe_job_done(out):
            e_ev = qe_parse_energy(out) * RY2EV
            results.append((frac, scale, vol, e_ev))
            print(c("ok", f"✓  E = {e_ev:.8f} eV  ({t}s)"))
        else:
            print(c("err", "✗ FAILED"))

    if results:
        print_header("QE Stage 3 – Results")
        summarise_eos(results)
        _save_dat(os.path.join(stage_dir, "results.dat"), ["frac", "scale", "vol_A3", "E_eV"], results)
        bm_fit(results, stage_dir, struct)
    return results

# ══════════════════════════════════════════════════════════════════
#  STAGE RUNNERS  –  SIESTA
# ══════════════════════════════════════════════════════════════════
def sia_run_stage(cfg, stage_id, label, param_list, fdf_gen, parse_fn, done_fn,
                  results_key, display_label, summarise_fn):
    exe, procs, pseudo_dir, base, struct = (
        cfg["exe"], cfg["procs"], cfg["pseudo_dir"], cfg["base"], cfg["struct"])
    stage_dir = os.path.join(base, f"siesta_0{stage_id}_{label}")
    results = []

    for params in param_list:
        sub_label = params["label"]
        subdir    = os.path.join(stage_dir, sub_label)
        fdf_path  = os.path.join(subdir, "input.fdf")
        out_path  = os.path.join(subdir, "output.out")
        err_path  = os.path.join(subdir, "error.err")
        os.makedirs(subdir, exist_ok=True)
        write_file(fdf_path, fdf_gen(cfg, params))
        sia_copy_pseudos(subdir, pseudo_dir, struct.species)
        print(f"  → {display_label} = {params['display']} ... ", end="", flush=True)
        cmd = sia_mpi_cmd(exe, procs, "input.fdf", "output.out", "error.err")
        t = run_cmd(f"cd {subdir} && {cmd}", os.path.join(subdir, "runner.log"))
        if done_fn(out_path):
            e = parse_fn(out_path)
            if e is not None:
                results.append(params["key"] + [e])
                print(c("ok", f"✓  E = {e:.8f} eV  ({t}s)"))
            else:
                print(c("warn", "✓ but no energy parsed"))
        else:
            print(c("err", "✗ FAILED"))

    if results:
        summarise_fn(results)
        _save_dat(os.path.join(stage_dir, "results.dat"), results_key, results)
    return results

def sia_run_stage1(cfg):
    struct = cfg["struct"]
    print_header(f"SIESTA Stage 1 – MeshCutoff convergence ({struct.formula})")
    print(f"  k-mesh: {SIA_KG_4MTEST}  |  BasisSize: {SIA_BASIS_FIXED}  |  EnergyShift: {SIA_ESHIFT_FIXED} Ry")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], struct, p["mc"], SIA_KG_4MTEST,
                         SIA_BASIS_FIXED, SIA_ESHIFT_FIXED)

    params = [{"label": f"meshcut_{mc}Ry", "display": str(mc),
               "mc": mc, "key": [mc]} for mc in SIA_MESHCUT]

    def summ(results):
        summarise_ecut([(r[0], r[1]) for r in results], EV2MEV, struct.nat)

    return sia_run_stage(cfg, 1, "meshcutoff", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["meshcut_Ry", "E_eV"], "MeshCutoff (Ry)", summ)

def sia_run_stage2(cfg):
    struct = cfg["struct"]
    print_header(f"SIESTA Stage 2 – k-point mesh convergence ({struct.formula})")
    print(f"  MeshCutoff: {SIA_MC_4KTEST} Ry  |  BasisSize: {SIA_BASIS_FIXED}  |  EnergyShift: {SIA_ESHIFT_FIXED} Ry")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], struct, SIA_MC_4KTEST, p["kg"],
                         SIA_BASIS_FIXED, SIA_ESHIFT_FIXED)

    params = [{"label": f"kgrid_{'x'.join(str(k) for k in kg)}",
               "display": "x".join(str(k) for k in kg),
               "kg": kg, "key": ["x".join(str(k) for k in kg)]}
              for kg in SIA_KGRIDS]

    def summ(results):
        summarise_kpoints([(r[0], r[1]) for r in results], struct.nat)

    return sia_run_stage(cfg, 2, "kpoints", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["kgrid", "E_eV"], "k-grid", summ)

def sia_run_stage3(cfg):
    struct = cfg["struct"]
    print_header(f"SIESTA Stage 3 – Basis set convergence ({struct.formula})")
    print(f"  MeshCutoff: {SIA_MC_4KTEST} Ry  |  k-mesh: {SIA_KG_4BASIS}  |  EnergyShift: {SIA_ESHIFT_FIXED} Ry")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], struct, SIA_MC_4KTEST, SIA_KG_4BASIS,
                         p["bs"], SIA_ESHIFT_FIXED)

    params = [{"label": f"basis_{bs}", "display": bs,
               "bs": bs, "key": [bs]} for bs in SIA_BASIS]

    def summ(results):
        summarise_basissize([(r[0], r[1]) for r in results], struct.nat)

    return sia_run_stage(cfg, 3, "basissize", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["basis", "E_eV"], "PAO.BasisSize", summ)

def sia_run_stage4(cfg):
    struct = cfg["struct"]
    print_header(f"SIESTA Stage 4 – PAO.EnergyShift convergence ({struct.formula})")
    print(f"  MeshCutoff: {SIA_MC_4KTEST} Ry  |  k-mesh: {SIA_KG_4ESHIFT}  |  BasisSize: {SIA_BASIS_FIXED}")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], struct, SIA_MC_4KTEST, SIA_KG_4ESHIFT,
                         SIA_BASIS_FIXED, p["es"])

    params = [{"label": f"eshift_{es}Ry", "display": f"{es} Ry",
               "es": es, "key": [es]} for es in SIA_ESHIFT]

    def summ(results):
        summarise_energyshift([(r[0], r[1]) for r in results], struct.nat)

    return sia_run_stage(cfg, 4, "energyshift", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["eshift_Ry", "E_eV"], "PAO.EnergyShift", summ)

def sia_run_stage5(cfg):
    struct = cfg["struct"]
    print_header(f"SIESTA Stage 5 – Equation of State Scan ({struct.formula})")
    print(f"  MeshCutoff: {SIA_MC_4EOS} Ry  |  k-mesh: {SIA_KG_4EOS}  |  Basis: {SIA_BASIS_4EOS}")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], struct, SIA_MC_4EOS, SIA_KG_4EOS,
                         SIA_BASIS_4EOS, SIA_ES_4EOS,
                         scale=p["scale"], cg_steps=EOS_CG_STEPS)

    params = []
    for frac in EOS_FRACS:
        scale = frac ** (1.0 / 3.0)
        vol = struct.volume * frac
        params.append({"label": f"vol_{frac:.3f}",
                       "display": f"{frac:.3f} (V={vol:.2f} Å³)",
                       "scale": scale,
                       "key": [frac, scale, vol]})

    def summ(results):
        summarise_eos([(r[0], r[1], r[2], r[3]) for r in results])
        stage_dir = os.path.join(cfg["base"], "siesta_05_eos")
        bm_fit([(r[0], r[1], r[2], r[3]) for r in results], stage_dir, struct)

    return sia_run_stage(cfg, 5, "eos", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["frac", "scale", "vol_A3", "E_eV"],
                         "Vol fraction", summ)

def _save_dat(path, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# " + "  ".join(str(h) for h in headers) + "\n")
        for row in rows:
            f.write("  ".join(str(v) for v in row) + "\n")

def banner():
    print(c("header", """
╔══════════════════════════════════════════════════════════════════╗
║         DFT CONVERGENCE SUITE  ─  Universal Crystal Edition      ║
║         Supports:  Quantum ESPRESSO  &  SIESTA                   ║
╚══════════════════════════════════════════════════════════════════╝
"""))

# ══════════════════════════════════════════════════════════════════
#  INTERACTIVE SETUP
# ══════════════════════════════════════════════════════════════════
def interactive_setup():
    banner()
    cpu_count = os.cpu_count() or 1

    # ── Step 1: Crystal Structure CIF ──
    print(c("bold", "  ── Step 1 : Crystal structure (CIF) ───────────────────"))
    cif_files = [f for f in os.listdir(".") if f.lower().endswith(".cif")]
    cif_files.sort()
    default_cif = cif_files[0] if cif_files else "structure.cif"
    if cif_files:
        print("  Found CIF files in current directory:")
        for idx, f in enumerate(cif_files, 1):
            print(f"    {idx}) {f}")
        raw = ask(f"Select number or enter path to CIF file", "1")
        if raw.isdigit() and 1 <= int(raw) <= len(cif_files):
            cif_path = cif_files[int(raw) - 1]
        else:
            cif_path = raw
    else:
        cif_path = ask("Path to CIF file", default_cif)

    if not os.path.isfile(cif_path):
        print(c("err", f"  ERROR: CIF file '{cif_path}' not found."))
        sys.exit(1)

    struct = Structure(cif_path)
    print()
    print(c("ok", f"  ✓ Loaded crystal structure from: {cif_path}"))
    print(f"    Chemical formula : {struct.formula}")
    print(f"    Lattice params   : a={struct.cellpar[0]:.4f} Å, b={struct.cellpar[1]:.4f} Å, c={struct.cellpar[2]:.4f} Å")
    print(f"    Cell angles      : α={struct.cellpar[3]:.1f}°, β={struct.cellpar[4]:.1f}°, γ={struct.cellpar[5]:.1f}°")
    print(f"    Unit cell volume : {struct.volume:.3f} Å³")
    print(f"    Total atoms (nat): {struct.nat}")
    print(f"    Unique elements  : {', '.join(struct.species)}  ({struct.ntyp} species)")

    # ── Step 2: DFT Code ──
    print()
    print(c("bold", "  ── Step 2 : Choose DFT code ──────────────────────────"))
    code = ask_choice("Which code?", ["QE", "SIESTA"])

    # ── Step 3: Executable path ──
    print()
    print(c("bold", "  ── Step 3 : Executable path ───────────────────────────"))
    if code == "QE":
        auto = detect_exe(["pw.x"])
        exe  = ask("Path to pw.x", auto or "/path/to/pw.x")
    else:
        auto = detect_exe(["siesta"])
        exe  = ask("Path to siesta", auto or "/path/to/siesta")

    if not os.path.isfile(exe):
        print(c("warn", f"  WARNING: '{exe}' not found – inputs will be generated but runs will fail."))

    # ── Step 4: Pseudopotential directory ──
    print()
    print(c("bold", "  ── Step 4 : Pseudopotential directory ──────────────────"))
    if code == "QE":
        qe_home  = os.environ.get("QE_HOME", "")
        auto_ps  = os.path.join(qe_home, "pseudo") if qe_home else ""
        burai_ps = os.path.expanduser("~/.burai/.pseudopot")
        if os.path.isdir(burai_ps):
            auto_ps = burai_ps
        pseudo_dir = ask("Path to directory containing .UPF pseudopotentials",
                         auto_ps or "/path/to/upf_pseudos")
        print()
        print(f"  Scanning for UPF pseudopotentials for elements in {struct.formula} ...")
        found_all = True
        for sp in struct.species:
            try:
                f = find_qe_upf(pseudo_dir, sp)
                print(c("ok", f"    {sp:3s}  ✓  {f}"))
            except FileNotFoundError:
                print(c("err", f"    {sp:3s}  ✗  NOT FOUND in {pseudo_dir}"))
                found_all = False
        if not found_all:
            print(c("warn", "\n  Some UPFs are missing – calculations may fail."))
    else:
        sia_dir  = os.path.dirname(exe) if exe else ""
        nc_guess = os.path.join(os.path.dirname(sia_dir), "nc-sr-05_pbe_standard_psml")
        if not os.path.isdir(nc_guess):
            nc_guess = ""
        pseudo_dir = ask("Path to directory containing .psml OR .psf pseudopotentials",
                         nc_guess or "/path/to/pseudo_dir")
        print()
        print(f"  Scanning for pseudopotentials for elements in {struct.formula} ...")
        fmt, found, missing = detect_sia_pseudo_format(pseudo_dir, struct.species)
        for sp in struct.species:
            if sp in found:
                src, ext = found[sp]
                tag = c("ok", f"✓  {os.path.basename(src):<30} [{ext[1:].upper()}]")
            else:
                tag = c("err", f"✗  NOT FOUND  (.psml and .psf both absent)")
            print(f"    {sp:3s}  {tag}")
        if fmt == "NONE":
            print(c("err", "\n  No pseudopotentials found – check the directory."))
        elif missing:
            print(c("err", f"\n  Missing species: {missing}"))
        elif fmt == "MIXED":
            print(c("warn", f"\n  Mixed formats (PSML+PSF). Will use best available per species."))
        else:
            print(c("ok", f"\n  All {len(struct.species)} pseudopotentials found [{fmt} format]."))

    # ── Step 5: MPI & Threading ──
    print()
    print(c("bold", "  ── Step 5 : MPI / threading ────────────────────────────"))
    print(f"  Detected CPU count: {cpu_count}")
    procs = ask_int("Number of MPI processes", cpu_count // 2 or 1)
    omp = 1
    if code == "QE":
        omp = ask_int("OpenMP threads per MPI task (1 = pure MPI)", 1)

    # ── Step 6: Output Directory ──
    print()
    print(c("bold", "  ── Step 6 : Output directory ───────────────────────────"))
    default_out = os.path.join(os.getcwd(), f"{struct.prefix}_convergence")
    base = ask("Output base directory", default_out)
    os.makedirs(base, exist_ok=True)

    # ── Step 7: Stages ──
    print()
    print(c("bold", "  ── Step 7 : Select stages to run ───────────────────────"))
    if code == "QE":
        print("    1) ecutwfc convergence")
        print("    2) k-point mesh convergence")
        print("    3) Lattice vs Energy (EOS scan)")
        print("    A) All stages")
    else:
        print("    1) MeshCutoff convergence")
        print("    2) k-point mesh convergence")
        print("    3) PAO.BasisSize convergence")
        print("    4) PAO.EnergyShift convergence")
        print("    5) Lattice vs Energy (EOS scan)")
        print("    A) All stages")

    max_stage = 3 if code == "QE" else 5
    raw = ask("Enter stage numbers separated by spaces (e.g. 1 2 3) or A for all", "A")
    if raw.upper() == "A":
        stages = list(range(1, max_stage + 1))
    else:
        stages = sorted(set(int(x) for x in raw.split() if x.isdigit()))
        stages = [s for s in stages if 1 <= s <= max_stage]

    # ── Summary ──
    print()
    print(c("bold", "  ─── Configuration Summary ─────────────────────────────"))
    print(f"  Structure      : {struct.filename}  ({struct.formula}, {struct.nat} atoms)")
    print(f"  Lattice Volume : {struct.volume:.3f} Å³")
    print(f"  Code           : {code}")
    print(f"  Executable     : {exe}")
    print(f"  Pseudo dir     : {pseudo_dir}")
    print(f"  MPI processes  : {procs}")
    if code == "QE":
        print(f"  OMP threads    : {omp}  (total cores used: {procs * omp})")
    print(f"  Output dir     : {base}")
    print(f"  Stages to run  : {stages}")
    print()

    ok = ask("Proceed? (yes/no)", "yes")
    if ok.lower() not in ("yes", "y"):
        print("  Aborted.")
        sys.exit(0)

    return {
        "struct":     struct,
        "code":       code,
        "exe":        exe,
        "pseudo_dir": pseudo_dir,
        "procs":      procs,
        "omp":        omp,
        "base":       base,
        "stages":     stages,
    }

# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def main():
    cfg = interactive_setup()
    code   = cfg["code"]
    stages = cfg["stages"]
    struct = cfg["struct"]
    t_start = time.time()

    if code == "QE":
        runners = {
            1: qe_run_stage1,
            2: qe_run_stage2,
            3: qe_run_stage3,
        }
    else:
        runners = {
            1: sia_run_stage1,
            2: sia_run_stage2,
            3: sia_run_stage3,
            4: sia_run_stage4,
            5: sia_run_stage5,
        }

    for s in stages:
        runners[s](cfg)

    total_time = round(time.time() - t_start)
    m, sec = divmod(total_time, 60)
    h, m   = divmod(m, 60)

    print()
    print(c("ok", "═" * 64))
    print(c("ok", f"  CONVERGENCE SUITE COMPLETE for {struct.formula}"))
    print(c("ok", f"  Total runtime : {h:02d}h {m:02d}m {sec:02d}s"))
    print(c("ok", f"  Results saved : {cfg['base']}"))
    print(c("ok", "═" * 64))

if __name__ == "__main__":
    main()
