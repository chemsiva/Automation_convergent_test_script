#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          DFT CONVERGENCE SUITE  –  FAPbI3 Cubic Perovskite       ║
║          Supports: Quantum ESPRESSO  &  SIESTA                   ║
║          Version : 1.0   (2026-09-24)                  d          ║
╠══════════════════════════════════════════════════════════════════╣
║  Usage : python3 convergence_suite.py                            ║
║  The script will ask interactively:                              ║
║    • Which DFT code  (QE or SIESTA)                              ║
║    • Path to executable                                          ║
║    • Path to pseudopotential directory                           ║
║    • Number of MPI processes                                     ║
║    • Number of OpenMP threads per MPI task  (QE only)            ║
║    • Which stages to run                                         ║
║                                                                  ║
║  Stages – Quantum ESPRESSO                                       ║
║    1. ecutwfc convergence     (plane-wave cutoff)                ║
║    2. k-point mesh convergence                                   ║
║    3. Lattice vs Energy       (EOS scan, Birch-Murnaghan fit)    ║
║                                                                  ║
║  Stages – SIESTA                                                 ║
║    1. MeshCutoff convergence  (real-space grid)                  ║
║    2. k-point mesh convergence                                   ║
║    3. PAO.BasisSize convergence  (SZ / DZ / DZP / TZP)          ║
║    4. PAO.EnergyShift convergence                                ║
║    5. Lattice vs Energy       (EOS scan, Birch-Murnaghan fit)    ║
║                                                                  ║
║  Portability                                                     ║
║    • No hardcoded paths or processor counts                      ║
║    • Auto-detects executables via PATH                           ║
║    • Auto-detects CPU count as default suggestion                ║
║    • Works on any Linux/macOS cluster or workstation             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import shutil
import subprocess
import time

# ══════════════════════════════════════════════════════════════════
#  CRYSTAL STRUCTURE  –  FAPbI3 cubic  (DO NOT CHANGE)
#  Source: WMD group PBEsol-optimised structure, 300 K
#  Space group P1,  a = b = c = 6.36130 Å,  12 atoms
# ══════════════════════════════════════════════════════════════════
A_ANG   = 6.36130          # reference lattice parameter (Å)
NAT     = 12               # number of atoms
NTYP    = 5                # number of species: C, H, N, Pb, I

# fractional atomic coordinates (x, y, z, species-label)
ATOMS = [
    (0.500001, 0.569011, 0.500000, "C"),
    (0.500000, 0.741416, 0.500000, "H"),
    (0.814009, 0.567630, 0.500000, "H"),
    (0.704462, 0.315280, 0.500000, "H"),
    (0.295538, 0.315280, 0.500000, "H"),
    (0.185991, 0.567630, 0.500000, "H"),
    (0.682927, 0.475280, 0.500000, "N"),
    (0.317071, 0.475280, 0.500000, "N"),
    (0.000000, 0.000000, 0.000000, "Pb"),
    (0.500000, 0.000000, 0.000000, "I"),
    (0.000000, 0.500000, 0.000000, "I"),
    (0.000000, 0.000000, 0.500000, "I"),
]

# ══════════════════════════════════════════════════════════════════
#  CONVERGENCE TEST RANGES  –  edit these ranges if needed
# ══════════════════════════════════════════════════════════════════

# QE
QE_ECUT_LIST    = [40, 50, 60, 70, 80, 90, 100]   # Ry
QE_ECUT_DUAL    = 8     # ecutrho = dual * ecutwfc (PAW)
QE_KGRIDS       = [(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6),(7,7,7)]
QE_ECUT_4KTEST  = 50    # Ry  (fixed during k-point test)
QE_KGRID_4ETEST = (3,3,3)  # (fixed during ecut test)
QE_ECUT_4EOS    = 70    # Ry  (converged value for EOS)
QE_KGRID_4EOS   = (4,4,4)  # (converged grid for EOS)

# SIESTA
SIA_MESHCUT     = [100,150,200,250,300,350,400,500]  # Ry
SIA_KGRIDS      = [(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6),(7,7,7)]
SIA_BASIS       = ["SZ","DZ","DZP","TZP"]
SIA_ESHIFT      = ["0.005","0.010","0.020","0.050","0.100"]  # Ry
SIA_MC_4KTEST   = 300   # Ry
SIA_KG_4MTEST   = (3,3,3)
SIA_KG_4BASIS   = (4,4,4)
SIA_KG_4ESHIFT  = (4,4,4)
SIA_MC_4EOS     = 300
SIA_KG_4EOS     = (4,4,4)
SIA_BASIS_4EOS  = "DZP"
SIA_ES_4EOS     = "0.010"
SIA_ESHIFT_FIXED = "0.020"
SIA_BASIS_FIXED  = "DZP"

# EOS (both codes)
EOS_FRACS = [round(0.940 + i*0.005, 3) for i in range(22)]  # 0.940 … 1.045
EOS_CG_STEPS = 100   # ionic relaxation steps at each fixed a

# ══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

BOHR = 1.889726             # Å → Bohr
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
    """Prompt user; return default on empty input."""
    if default:
        ans = input(f"  {prompt} [{default}]: ").strip()
        return ans if ans else default
    ans = input(f"  {prompt}: ").strip()
    return ans


def ask_int(prompt, default):
    while True:
        raw = ask(prompt, str(default))
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
        print(f"    {c('warn','Please enter a positive integer.')}")


def ask_choice(prompt, choices, default=None):
    """Return one of choices."""
    choices_str = " / ".join(choices)
    defval = default or choices[0]
    while True:
        ans = ask(f"{prompt}  ({choices_str})", defval).strip().upper()
        for ch in choices:
            if ans.upper() == ch.upper():
                return ch
        print(f"    {c('warn', 'Please choose one of: ' + choices_str)}")


def detect_exe(names):
    """Return first found executable path or empty string."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return ""


def good_fft(n):
    """Smallest integer ≥ n whose prime factors are only {2,3,5}."""
    def ok(x):
        for p in (2,3,5):
            while x % p == 0:
                x //= p
        return x == 1
    while not ok(n):
        n += 1
    return n


def fft_grid_qe(ecutwfc, ecutrho, celldm1_bohr):
    """Estimate FFT grid for cubic QE cell."""
    g = math.sqrt(ecutrho)
    a = celldm1_bohr
    nr_raw = int(math.ceil(2.0 * g * a / (2.0 * math.pi))) + 2
    nr = good_fft(max(nr_raw, 24))
    return nr, nr, nr


def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def run_cmd(cmd, logfile):
    """Run shell command, capture to logfile; return wall-time seconds."""
    t0 = time.time()
    with open(logfile, "w") as lf:
        subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)
    return round(time.time() - t0)


def print_header(title):
    w = 60
    print()
    print(c("header", "═" * w))
    print(c("header", f"  {title}"))
    print(c("header", "═" * w))


# ══════════════════════════════════════════════════════════════════
#  QE INPUT BUILDERS
# ══════════════════════════════════════════════════════════════════

def find_qe_upf(pseudo_dir, species):
    """
    Scan pseudo_dir for a UPF file for 'species'.
    Priority: PAW (kjpaw) > USPP (rrkjus) > any .UPF
    Returns filename (basename), raises FileNotFoundError if none found.
    """
    import glob as _glob
    # Glob broadly then filter to exact element prefix (e.g. "C." or "C-")
    # This prevents C→Ce, H→Hg, N→Na false matches.
    raw = (
        _glob.glob(os.path.join(pseudo_dir, f"{species}*.UPF")) +
        _glob.glob(os.path.join(pseudo_dir, f"{species}*.upf"))
    )
    # keep only filenames whose stem starts exactly with species followed by
    # a non-letter character (dot, underscore, dash, digit)
    import re as _re
    pat = _re.compile(rf"^{_re.escape(species)}[^A-Za-z]", _re.IGNORECASE)
    candidates = [p for p in raw if pat.match(os.path.basename(p))]
    if not candidates:
        raise FileNotFoundError(
            f"No UPF found for {species} in {pseudo_dir}  "
            f"(expected e.g. {species}.pbe-n-kjpaw_psl.1.0.0.UPF)")
    def rank(p):
        b = os.path.basename(p).lower()
        return 0 if "kjpaw" in b else (1 if "rrkjus" in b else 2)
    return os.path.basename(sorted(candidates, key=rank)[0])


def qe_atomic_species(pseudo_dir):
    """Build ATOMIC_SPECIES block; auto-selects best UPF per element."""
    masses  = {"C":12.011,"H":1.008,"N":14.007,"Pb":207.2,"I":126.904}
    lines   = ["ATOMIC_SPECIES"]
    missing = []
    for sp in ["C","H","N","Pb","I"]:
        try:
            fname = find_qe_upf(pseudo_dir, sp)
        except FileNotFoundError as e:
            missing.append(str(e))
            fname = f"{sp}.pbe-MISSING.UPF"
        lines.append(f"  {sp:<4} {masses[sp]:>9.3f}  {fname}")
    if missing:
        print(c("err", "\n  [QE] Missing pseudopotentials:"))
        for m in missing:
            print(c("err", f"    {m}"))
    return "\n".join(lines)


def qe_atomic_positions():
    lines = ["ATOMIC_POSITIONS {crystal}"]
    for x, y, z, sp in ATOMS:
        lines.append(f"  {sp:<3}  {x:.6f}  {y:.6f}  {z:.6f}")
    return "\n".join(lines)


def qe_kpoints(kg):
    return f"K_POINTS {{automatic}}\n  {kg[0]} {kg[1]} {kg[2]}  0 0 0"


def qe_input(pseudo_dir, ecutwfc, kg, a_ang=A_ANG, calc="scf"):
    ecutrho  = QE_ECUT_DUAL * ecutwfc
    cdm1     = a_ang * BOHR
    nr1,nr2,nr3 = fft_grid_qe(ecutwfc, ecutrho, cdm1)
    ions = "\n&IONS\n    ion_dynamics = 'bfgs'\n/" if calc == "relax" else ""
    return f"""\
&CONTROL
    calculation   = '{calc}'
    prefix        = 'FAPbI3'
    pseudo_dir    = '{pseudo_dir}'
    outdir        = './out'
    tstress       = .true.
    tprnfor       = .true.
    verbosity     = 'medium'
    max_seconds   = 86400
/
&SYSTEM
    ibrav         = 1
    celldm(1)     = {cdm1:.6f}
    nat           = {NAT}
    ntyp          = {NTYP}
    ecutwfc       = {ecutwfc}
    ecutrho       = {ecutrho}
    degauss       = 0.01
    occupations   = 'smearing'
    smearing      = 'gaussian'
    nr1           = {nr1}
    nr2           = {nr2}
    nr3           = {nr3}
/
&ELECTRONS
    conv_thr         = 1.0d-8
    mixing_beta      = 0.40
    electron_maxstep = 200
    diagonalization  = 'david'
    diago_david_ndim = 4
/{ions}
{qe_atomic_species(pseudo_dir)}
{qe_atomic_positions()}
{qe_kpoints(kg)}
"""


# ══════════════════════════════════════════════════════════════════
#  SIESTA INPUT BUILDERS
# ══════════════════════════════════════════════════════════════════

def detect_sia_pseudo_format(pseudo_dir):
    """
    Scan pseudo_dir for .psml and .psf for the 5 required species.
    Returns (fmt, found, missing):
      fmt    : 'PSML' | 'PSF' | 'MIXED' | 'NONE'
      found  : {sp: (abs_path, ext)}
      missing: [sp, ...]
    PSML is preferred over PSF when both exist for the same species.
    """
    species = ["C","H","N","Pb","I"]
    found   = {}
    import glob as _glob
    import re as _re
    for sp in species:
        pat = _re.compile(rf"^{_re.escape(sp)}[^A-Za-z]", _re.IGNORECASE)
        # search for psml first (preferred)
        psml_matches = [
            f for f in _glob.glob(os.path.join(pseudo_dir, "*.[pP][sS][mM][lL]"))
            if pat.match(os.path.basename(f))
        ]
        if psml_matches:
            # prefer exact match like Pb.psml if available
            exact = [f for f in psml_matches if os.path.basename(f).lower() == f"{sp.lower()}.psml"]
            found[sp] = (exact[0] if exact else psml_matches[0], ".psml")
            continue
        # fallback: search for psf
        psf_matches = [
            f for f in _glob.glob(os.path.join(pseudo_dir, "*.[pP][sS][fF]"))
            if pat.match(os.path.basename(f))
        ]
        if psf_matches:
            exact = [f for f in psf_matches if os.path.basename(f).lower() == f"{sp.lower()}.psf"]
            found[sp] = (exact[0] if exact else psf_matches[0], ".psf")
    missing = [sp for sp in species if sp not in found]
    exts    = set(ext for _, ext in found.values())
    if not exts:
        fmt = "NONE"
    elif exts == {".psml"}:
        fmt = "PSML"
    elif exts == {".psf"}:
        fmt = "PSF"
    else:
        fmt = "MIXED"
    return fmt, found, missing


def sia_species_block(pseudo_dir):
    """
    Build %block ChemicalSpeciesLabel with correct pseudo filename per species.
    Auto-detects PSML vs PSF in pseudo_dir.
    """
    Z   = {"C":6,"H":1,"N":7,"Pb":82,"I":53}
    idx = {"C":1,"H":2,"N":3,"Pb":4,"I":5}
    _, found, _ = detect_sia_pseudo_format(pseudo_dir)
    lines = ["%block ChemicalSpeciesLabel"]
    for sp in ["C","H","N","Pb","I"]:
        fname = (os.path.basename(found[sp][0])
                 if sp in found else f"{sp}.MISSING")
        lines.append(f"  {idx[sp]}  {Z[sp]:3d}  {sp}")
    lines.append("%endblock ChemicalSpeciesLabel")
    return "\n".join(lines)


def sia_coords_block():
    sp_idx = {"C":1,"H":2,"N":3,"Pb":4,"I":5}
    lines = ["%block AtomicCoordinatesAndAtomicSpecies"]
    for x,y,z,sp in ATOMS:
        lines.append(f"  {x:.6f}   {y:.6f}   {z:.6f}   {sp_idx[sp]}   # {sp}")
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    return "\n".join(lines)


def sia_kblock(kg):
    return (f"%block kgrid_Monkhorst_Pack\n"
            f"  {kg[0]:2d}   0   0   0.0\n"
            f"   0  {kg[1]:2d}   0   0.0\n"
            f"   0   0  {kg[2]:2d}   0.0\n"
            f"%endblock kgrid_Monkhorst_Pack")


def sia_input(pseudo_dir, meshcut, kg, basis, eshift,
              a_ang=A_ANG, cg_steps=0):
    return f"""\
SystemName              FAPbI3_cubic
SystemLabel             FAPbI3
NumberOfAtoms           {NAT}
NumberOfSpecies         {NTYP}

{sia_species_block(pseudo_dir)}

LatticeConstant         {a_ang:.6f} Ang

%block LatticeVectors
  1.000000  0.000000  0.000000
  0.000000  1.000000  0.000000
  0.000000  0.000000  1.000000
%endblock LatticeVectors

AtomicCoordinatesFormat  Fractional

{sia_coords_block()}

XC.functional           GGA
XC.authors              PBE

DFT.D3                  true
DFT.D3.version          4
DFT.D3.damping          BJ
DFT.D3.threebody        true

MeshCutoff              {meshcut} Ry

PAO.BasisType           split
PAO.BasisSize           {basis}
PAO.EnergyShift         {eshift} Ry

{sia_kblock(kg)}

SolutionMethod          diagon
MaxSCFIterations        300
ElectronicTemperature   300 K
SCF.MustConverge        true
DM.MixingWeight         0.10
DM.NumberPulay          5
DM.Tolerance            1.0d-5
DM.Require.Energy.Convergence   T
DM.Energy.Tolerance             1.0d-5 eV
DM.UseSaveDM            F

MD.TypeOfRun            CG
MD.NumCGsteps           {cg_steps}
MD.MaxForceTol          0.01 eV/Ang
MD.MaxStressTol         0.50 kBar
MD.VariableCell         F

SpinPolarized           F
WriteForces             T
WriteMullikenPop        1
XML.Write               T
SaveHS                  F
SaveRHO                 F
"""


def sia_copy_pseudos(dest, pseudo_dir):
    """
    Detect PSML or PSF (or both) in pseudo_dir for C H N Pb I.
    PSML preferred; falls back to PSF per-species if PSML absent.
    Symlinks found files into dest; copies if cross-device.
    Prints a clear warning for any species with no pseudo found.
    """
    fmt, found, missing = detect_sia_pseudo_format(pseudo_dir)
    os.makedirs(dest, exist_ok=True)

    if missing:
        print(c("err", f"\n  [SIESTA] No pseudo found for: {missing}"))
        print(c("err",  "  Add .psml or .psf files for these elements to pseudo_dir."))

    for sp, (src, ext) in found.items():
        dst = os.path.join(dest, sp + ext)
        if not os.path.exists(dst):
            try:    os.symlink(src, dst)
            except OSError: shutil.copy2(src, dst)


# ══════════════════════════════════════════════════════════════════
#  OUTPUT PARSERS
# ══════════════════════════════════════════════════════════════════

def qe_parse_energy(outfile):
    """Return total energy in Ry, or None."""
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
    """Return total energy in eV, or None."""
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
#  CONVERGENCE TABLES
# ══════════════════════════════════════════════════════════════════

def print_table(rows, headers, col_widths, criterion_note=""):
    """Generic table printer. rows: list of strings per column."""
    sep = "  ".join(f"{'─'*w}" for w in col_widths)
    hdr = "  ".join(f"{h:>{w}}" for h, w in zip(headers, col_widths))
    print(f"\n  {c('bold', hdr)}")
    print(f"  {sep}")
    for row in rows:
        line = "  ".join(f"{v:>{w}}" for v, w in zip(row, col_widths))
        # highlight converged rows (last col contains "←")
        if "←" in row[-1]:
            print(f"  {c('ok', line)}")
        else:
            print(f"  {line}")
    print(f"  {sep}")
    if criterion_note:
        print(f"\n  {c('warn', criterion_note)}\n")


def summarise_ecut(data, unit_factor, unit_label):
    """data: list of (param, energy_in_ry_or_ev)"""
    rows = []
    prev = None
    for param, e in data:
        de = (e - prev) * unit_factor / NAT if prev is not None else 0.0
        flag = "  ←" if abs(de) < 1.0 and prev is not None else ""
        rows.append((str(param), f"{e:.8f}", f"{de:.4f}{flag}"))
        prev = e
    print_table(rows,
                ["Cutoff", "E_tot", "ΔE (meV/atom)"],
                [12, 18, 16],
                "Criterion: |ΔE| < 1 meV/atom from previous step  (← converged)")


def summarise_kpoints(data):
    rows = []
    prev = None
    for kg_str, e in data:
        de = (e - prev) * EV2MEV / NAT if prev is not None else 0.0
        flag = "  ←" if abs(de) < 0.5 and prev is not None else ""
        rows.append((kg_str, f"{e:.8f}", f"{de:.4f}{flag}"))
        prev = e
    print_table(rows,
                ["k-mesh", "E_tot (eV)", "ΔE (meV/atom)"],
                [12, 18, 16],
                "Criterion: |ΔE| < 0.5 meV/atom  (← converged)")


def summarise_basissize(data):
    ORDER = {"SZ":0,"DZ":1,"DZP":2,"TZP":3}
    data_s = sorted(data, key=lambda r: ORDER.get(r[0], 99))
    rows = []
    prev = None
    comp = {"SZ":"minimal","DZ":"double-ζ","DZP":"double-ζ+pol","TZP":"triple-ζ+pol"}
    for bs, e in data_s:
        de = (e - prev) * EV2MEV / NAT if prev is not None else 0.0
        flag = "  ←" if abs(de) < 1.0 and prev is not None else ""
        rows.append((bs, comp.get(bs,""), f"{e:.8f}", f"{de:.4f}{flag}"))
        prev = e
    print_table(rows,
                ["BasisSize","Description","E_tot (eV)","ΔE (meV/atom)"],
                [10,16,18,16],
                "Criterion: |ΔE| < 1 meV/atom vs next-larger basis")


def summarise_eos(data):
    """data: list of (frac, a_ang, vol, energy)"""
    rows = []
    for frac, a, v, e in sorted(data, key=lambda r: r[2]):
        rows.append((f"{frac:.3f}", f"{a:.5f}", f"{v:.4f}", f"{e:.8f}"))
    print_table(rows,
                ["Vol frac","a (Å)","V (Å³)","E_tot"],
                [9, 10, 10, 18])


# ══════════════════════════════════════════════════════════════════
#  BIRCH-MURNAGHAN EOS FIT
# ══════════════════════════════════════════════════════════════════

def bm_fit(data_ev, base_dir):
    """
    data_ev: list of (frac, a_ang, vol_A3, energy_eV)
    Fit BM 3rd-order EOS; write fit curve; print results.
    """
    vols = [r[2] for r in data_ev]
    engs = [r[3] for r in data_ev]
    try:
        from scipy.optimize import curve_fit
        import numpy as np
    except ImportError:
        i0 = engs.index(min(engs))
        print(f"\n  {c('warn','scipy not found – skipping BM fit.')}")
        print(f"  Raw minimum: a = {data_ev[i0][1]:.5f} Å  "
              f"V = {vols[i0]:.4f} Å³  E = {engs[i0]:.8f} eV")
        return

    V = __import__("numpy").array(vols, dtype=float)
    E = __import__("numpy").array(engs, dtype=float)
    import numpy as np

    def bm3(V, E0, V0, B0, Bp):
        B0_eV = B0 / 160.21766208   # GPa → eV/Å³
        eta   = (V0 / V) ** (2.0/3.0)
        return (E0 + 9.0*V0*B0_eV/16.0 *
                (((eta-1.0)**3)*Bp + ((eta-1.0)**2)*(6.0-4.0*eta)))

    i0 = int(np.argmin(E))
    try:
        popt, pcov = curve_fit(bm3, V, E, p0=[E[i0],V[i0],30.0,4.0],
                               maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        E0, V0, B0, Bp = popt
        a0 = V0**(1.0/3.0)
        print_header("Birch–Murnaghan 3rd-order EOS Fit")
        print(f"    E₀  = {E0:.6f} ± {perr[0]:.6f}  eV")
        print(f"    V₀  = {V0:.4f}  ± {perr[1]:.4f}   Å³")
        print(c("ok",
                f"    a₀  = {a0:.5f}  Å    ← equilibrium lattice parameter"))
        print(c("ok",
                f"    B₀  = {B0:.2f}   ± {perr[2]:.2f}    GPa  ← bulk modulus"))
        print(f"    B₀' = {Bp:.3f}   ± {perr[3]:.3f}")
        rmse = float(np.sqrt(np.mean((E - bm3(V, *popt))**2))) * 1000.0
        print(f"    RMSE = {rmse:.3f}  meV")

        # write fitted curve
        Vd = np.linspace(min(V)*0.98, max(V)*1.02, 200)
        Ed = bm3(Vd, *popt)
        fit_path = os.path.join(base_dir, "eos_fit_curve.dat")
        with open(fit_path, "w") as f:
            f.write("# V(Å³)   E_fit(eV)\n")
            for v, e in zip(Vd, Ed):
                f.write(f"{v:.4f}  {e:.8f}\n")
        print(f"\n    Fitted curve → {fit_path}")
    except RuntimeError as err:
        print(f"  {c('err','BM fit did not converge')}: {err}")


# ══════════════════════════════════════════════════════════════════
#  STAGE RUNNERS  –  Quantum ESPRESSO
# ══════════════════════════════════════════════════════════════════

def qe_mpi_cmd(exe, mpi_procs, omp, npool, input_f, output_f):
    pool_arg = f"-npool {npool}" if npool > 1 else ""
    return (f"OMP_NUM_THREADS={omp} OMP_PROC_BIND=close OMP_PLACES=cores "
            f"mpirun -np {mpi_procs} {exe} -ntg 2 {pool_arg} "
            f"-i {input_f} > {output_f} 2>&1")


def qe_npool(kg):
    nk = kg[0]*kg[1]*kg[2]
    for p in (4, 3, 2, 1):
        if nk % p == 0:
            return p
    return 1


def qe_run_stage1(cfg):
    exe, procs, omp, pseudo_dir, base = (
        cfg["exe"], cfg["procs"], cfg["omp"],
        cfg["pseudo_dir"], cfg["base"])
    stage_dir = os.path.join(base, "qe_01_ecut")
    print_header("QE Stage 1 – ecutwfc convergence")
    print(f"  k-mesh fixed: {QE_KGRID_4ETEST}  |  ecutrho = {QE_ECUT_DUAL}×ecutwfc")

    results = []
    for ecut in QE_ECUT_LIST:
        label  = f"ecut_{ecut}Ry"
        subdir = os.path.join(stage_dir, label)
        inp    = os.path.join(subdir, "input.in")
        out    = os.path.join(subdir, "output.out")
        os.makedirs(os.path.join(subdir, "out"), exist_ok=True)
        write_file(inp, qe_input(pseudo_dir, ecut, QE_KGRID_4ETEST))
        cmd = qe_mpi_cmd(exe, procs, omp, qe_npool(QE_KGRID_4ETEST), "input.in", "output.out")
        print(f"  → ecutwfc = {ecut} Ry ... ", end="", flush=True)
        t = run_cmd(f"cd {subdir} && {cmd}", os.path.join(subdir, "runner.log"))
        if qe_job_done(out):
            e = qe_parse_energy(out)
            results.append((ecut, e))
            print(c("ok", f"✓  E = {e:.8f} Ry  ({t}s)"))
        else:
            print(c("err", "✗ FAILED"))

    if results:
        print_header("QE Stage 1 – Results")
        summarise_ecut([(r[0], r[1]*RY2EV) for r in results],
                       EV2MEV, "meV")
        _save_dat(os.path.join(stage_dir, "results.dat"),
                  ["ecut_Ry", "E_Ry", "E_eV"],
                  [(r[0], r[1], r[1]*RY2EV) for r in results])
    return results


def qe_run_stage2(cfg):
    exe, procs, omp, pseudo_dir, base = (
        cfg["exe"], cfg["procs"], cfg["omp"],
        cfg["pseudo_dir"], cfg["base"])
    stage_dir = os.path.join(base, "qe_02_kpoints")
    print_header("QE Stage 2 – k-point mesh convergence")
    print(f"  ecutwfc fixed: {QE_ECUT_4KTEST} Ry")

    results = []
    for kg in QE_KGRIDS:
        kg_str = "×".join(str(k) for k in kg)
        label  = f"kgrid_{'x'.join(str(k) for k in kg)}"
        subdir = os.path.join(stage_dir, label)
        inp    = os.path.join(subdir, "input.in")
        out    = os.path.join(subdir, "output.out")
        os.makedirs(os.path.join(subdir, "out"), exist_ok=True)
        write_file(inp, qe_input(pseudo_dir, QE_ECUT_4KTEST, kg))
        cmd = qe_mpi_cmd(exe, procs, omp, qe_npool(kg), "input.in", "output.out")
        print(f"  → k-mesh {kg_str} ... ", end="", flush=True)
        t = run_cmd(f"cd {subdir} && {cmd}", os.path.join(subdir, "runner.log"))
        if qe_job_done(out):
            e = qe_parse_energy(out) * RY2EV
            results.append((kg_str, e))
            print(c("ok", f"✓  E = {e:.8f} eV  ({t}s)"))
        else:
            print(c("err", "✗ FAILED"))

    if results:
        print_header("QE Stage 2 – Results")
        summarise_kpoints(results)
        _save_dat(os.path.join(stage_dir, "results.dat"),
                  ["k_mesh", "E_eV"], results)
    return results


def qe_run_stage3(cfg):
    exe, procs, omp, pseudo_dir, base = (
        cfg["exe"], cfg["procs"], cfg["omp"],
        cfg["pseudo_dir"], cfg["base"])
    stage_dir = os.path.join(base, "qe_03_eos")
    print_header("QE Stage 3 – Lattice parameter vs Energy  (EOS scan)")
    print(f"  ecutwfc = {QE_ECUT_4EOS} Ry  |  k-mesh = {QE_KGRID_4EOS}  |  calc = relax")

    results = []
    for frac in EOS_FRACS:
        a_new  = A_ANG * (frac ** (1.0/3.0))
        vol    = a_new ** 3
        label  = f"vol_{frac:.3f}"
        subdir = os.path.join(stage_dir, label)
        inp    = os.path.join(subdir, "input.in")
        out    = os.path.join(subdir, "output.out")
        os.makedirs(os.path.join(subdir, "out"), exist_ok=True)
        write_file(inp, qe_input(pseudo_dir, QE_ECUT_4EOS, QE_KGRID_4EOS,
                                  a_ang=a_new, calc="relax"))
        cmd = qe_mpi_cmd(exe, procs, omp, qe_npool(QE_KGRID_4EOS), "input.in", "output.out")
        print(f"  → frac={frac:.3f}  a={a_new:.5f} Å ... ", end="", flush=True)
        t = run_cmd(f"cd {subdir} && {cmd}", os.path.join(subdir, "runner.log"))
        if qe_job_done(out):
            e_ry = qe_parse_energy(out)
            e_ev = e_ry * RY2EV
            results.append((frac, a_new, vol, e_ev))
            print(c("ok", f"✓  E = {e_ev:.8f} eV  ({t}s)"))
        else:
            print(c("err", "✗ FAILED"))

    if results:
        print_header("QE Stage 3 – Results")
        summarise_eos(results)
        _save_dat(os.path.join(stage_dir, "results.dat"),
                  ["frac","a_ang","vol_A3","E_eV"], results)
        bm_fit(results, stage_dir)
    return results


# ══════════════════════════════════════════════════════════════════
#  STAGE RUNNERS  –  SIESTA
# ══════════════════════════════════════════════════════════════════

def sia_mpi_cmd(exe, procs, fdf, out, err):
    return f"mpirun -np {procs} {exe} < {fdf} > {out} 2> {err}"


def sia_run_stage(cfg, stage_id, label, param_list, fdf_gen, parse_fn, done_fn,
                  results_key, display_label, summarise_fn):
    exe, procs, pseudo_dir, base = (
        cfg["exe"], cfg["procs"], cfg["pseudo_dir"], cfg["base"])
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
        sia_copy_pseudos(subdir, pseudo_dir)
        print(f"  → {display_label} = {params['display']} ... ",
              end="", flush=True)
        cmd = sia_mpi_cmd(exe, procs, "input.fdf", "output.out", "error.err")
        t   = run_cmd(f"cd {subdir} && {cmd}", os.path.join(subdir,"runner.log"))
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
        _save_dat(os.path.join(stage_dir, "results.dat"),
                  results_key, results)
    return results


def sia_run_stage1(cfg):
    print_header("SIESTA Stage 1 – MeshCutoff convergence")
    print(f"  k-mesh: {SIA_KG_4MTEST}  |  BasisSize: {SIA_BASIS_FIXED}  "
          f"|  EnergyShift: {SIA_ESHIFT_FIXED} Ry")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], p["mc"], SIA_KG_4MTEST,
                         SIA_BASIS_FIXED, SIA_ESHIFT_FIXED)

    params = [{"label":f"meshcut_{mc}Ry","display":str(mc),
               "mc":mc,"key":[mc]} for mc in SIA_MESHCUT]

    def summ(results):
        summarise_ecut([(r[0], r[1]) for r in results], EV2MEV, "meV")

    return sia_run_stage(cfg, 1, "meshcutoff", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["meshcut_Ry","E_eV"], "MeshCutoff (Ry)", summ)


def sia_run_stage2(cfg):
    print_header("SIESTA Stage 2 – k-point mesh convergence")
    print(f"  MeshCutoff: {SIA_MC_4KTEST} Ry  |  BasisSize: {SIA_BASIS_FIXED}")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], SIA_MC_4KTEST, p["kg"],
                         SIA_BASIS_FIXED, SIA_ESHIFT_FIXED)

    params = [{"label":f"kgrid_{'x'.join(str(k) for k in kg)}",
               "display":"×".join(str(k) for k in kg),
               "kg":kg,"key":["×".join(str(k) for k in kg)]}
              for kg in SIA_KGRIDS]

    def summ(results):
        summarise_kpoints([(r[0], r[1]) for r in results])

    return sia_run_stage(cfg, 2, "kpoints", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["k_mesh","E_eV"], "k-mesh", summ)


def sia_run_stage3(cfg):
    print_header("SIESTA Stage 3 – PAO.BasisSize convergence")
    print(f"  MeshCutoff: {SIA_MC_4KTEST} Ry  |  k-mesh: {SIA_KG_4BASIS}")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], SIA_MC_4KTEST, SIA_KG_4BASIS,
                         p["bs"], SIA_ESHIFT_FIXED)

    params = [{"label":f"basis_{bs}","display":bs,
               "bs":bs,"key":[bs]} for bs in SIA_BASIS]

    def summ(results):
        summarise_basissize([(r[0], r[1]) for r in results])

    return sia_run_stage(cfg, 3, "basissize", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["basis","E_eV"], "PAO.BasisSize", summ)


def sia_run_stage4(cfg):
    print_header("SIESTA Stage 4 – PAO.EnergyShift convergence")
    print(f"  MeshCutoff: {SIA_MC_4KTEST} Ry  |  k-mesh: {SIA_KG_4ESHIFT}  "
          f"|  BasisSize: {SIA_BASIS_FIXED}")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], SIA_MC_4KTEST, SIA_KG_4ESHIFT,
                         SIA_BASIS_FIXED, p["es"])

    params = [{"label":f"eshift_{es}Ry","display":f"{es} Ry",
               "es":es,"key":[es]} for es in SIA_ESHIFT]

    def summ(results):
        summarise_ecut([(r[0], r[1]) for r in results], EV2MEV, "meV")

    return sia_run_stage(cfg, 4, "energyshift", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["eshift_Ry","E_eV"], "EnergyShift (Ry)", summ)


def sia_run_stage5(cfg):
    print_header("SIESTA Stage 5 – Lattice vs Energy  (EOS scan)")
    print(f"  MeshCutoff: {SIA_MC_4EOS} Ry  |  k-mesh: {SIA_KG_4EOS}  "
          f"|  BasisSize: {SIA_BASIS_4EOS}  |  CG steps: {EOS_CG_STEPS}")

    def fdf_gen(cfg, p):
        return sia_input(cfg["pseudo_dir"], SIA_MC_4EOS, SIA_KG_4EOS,
                         SIA_BASIS_4EOS, SIA_ES_4EOS,
                         a_ang=p["a"], cg_steps=EOS_CG_STEPS)

    params = []
    for frac in EOS_FRACS:
        a_new = A_ANG * (frac ** (1.0/3.0))
        vol   = a_new ** 3
        params.append({"label":f"vol_{frac:.3f}",
                        "display":f"{frac:.3f}  (a={a_new:.5f} Å)",
                        "a":a_new,"key":[frac, a_new, vol]})

    def summ(results):
        summarise_eos([(r[0],r[1],r[2],r[3]) for r in results])
        stage_dir = os.path.join(cfg["base"], "siesta_05_eos")
        bm_fit([(r[0],r[1],r[2],r[3]) for r in results], stage_dir)

    return sia_run_stage(cfg, 5, "eos", params, fdf_gen,
                         sia_parse_energy, sia_job_done,
                         ["frac","a_ang","vol_A3","E_eV"],
                         "Vol fraction", summ)


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _save_dat(path, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# " + "  ".join(str(h) for h in headers) + "\n")
        for row in rows:
            f.write("  ".join(str(v) for v in row) + "\n")


def banner():
    print(c("header", """
╔══════════════════════════════════════════════════════════════════╗
║         DFT CONVERGENCE SUITE  ─  FAPbI3 Cubic Perovskite        ║
║         Supports:  Quantum ESPRESSO  &  SIESTA 5.x               ║
╚══════════════════════════════════════════════════════════════════╝
"""))


# ══════════════════════════════════════════════════════════════════
#  INTERACTIVE SETUP
# ══════════════════════════════════════════════════════════════════

def interactive_setup():
    banner()
    cpu_count = os.cpu_count() or 1

    print(c("bold", "  ── Step 1 : Choose DFT code ──────────────────────────"))
    code = ask_choice("Which code?", ["QE", "SIESTA"])

    print()
    print(c("bold", "  ── Step 2 : Executable path ───────────────────────────"))
    if code == "QE":
        auto = detect_exe(["pw.x"])
        exe  = ask("Path to pw.x", auto or "/path/to/pw.x")
    else:
        auto = detect_exe(["siesta"])
        exe  = ask("Path to siesta", auto or "/path/to/siesta")

    if not os.path.isfile(exe):
        print(c("warn", f"  WARNING: '{exe}' not found – inputs will be generated "
                "but calculations will fail."))

    print()
    print(c("bold", "  ── Step 3 : Pseudopotential directory ──────────────────"))
    if code == "QE":
        # auto-guess: prefer burai, then $QE_HOME/pseudo
        qe_home  = os.environ.get("QE_HOME", "")
        auto_ps  = os.path.join(qe_home, "pseudo") if qe_home else ""
        burai_ps = os.path.expanduser("~/.burai/.pseudopot")
        if os.path.isdir(burai_ps):
            auto_ps = burai_ps
        pseudo_dir = ask("Path to directory containing .UPF pseudopotentials",
                         auto_ps or "/path/to/upf_pseudos")
        # report what was found
        print()
        print("  Checking UPF pseudopotentials ...")
        found_all = True
        for sp in ["C","H","N","Pb","I"]:
            try:
                f = find_qe_upf(pseudo_dir, sp)
                print(c("ok", f"    {sp:3s}  ✓  {f}"))
            except FileNotFoundError:
                print(c("err", f"    {sp:3s}  ✗  NOT FOUND in {pseudo_dir}"))
                found_all = False
        if not found_all:
            print(c("warn", "\n  Some pseudopotentials missing – fix before running."))
    else:
        # SIESTA: auto-guess nc-sr-05 library next to siesta binary
        sia_dir  = os.path.dirname(exe) if exe else ""
        nc_guess = os.path.join(os.path.dirname(sia_dir),
                                "nc-sr-05_pbe_standard_psml")
        if not os.path.isdir(nc_guess):
            nc_guess = ""
        pseudo_dir = ask(
            "Path to directory containing .psml OR .psf pseudopotentials",
            nc_guess or "/path/to/pseudo_dir")
        # detect format and report
        print()
        print("  Scanning for pseudopotentials (PSML preferred, PSF fallback) ...")
        fmt, found, missing = detect_sia_pseudo_format(pseudo_dir)
        for sp in ["C","H","N","Pb","I"]:
            if sp in found:
                src, ext = found[sp]
                tag = c("ok",   f"✓  {os.path.basename(src):<30} [{ext[1:].upper()}]")
            else:
                tag = c("err", f"✗  NOT FOUND  (.psml and .psf both absent)")
            print(f"    {sp:3s}  {tag}")
        if fmt == "NONE":
            print(c("err", "\n  No pseudopotentials found – check the directory."))
        elif missing:
            print(c("err",  f"\n  Missing species: {missing}"))
        elif fmt == "MIXED":
            print(c("warn", f"\n  Mixed formats (PSML+PSF). Will use best available per species."))
        else:
            print(c("ok",   f"\n  All 5 pseudopotentials found  [{fmt} format]."))

    print()
    print(c("bold", "  ── Step 4 : MPI / threading ────────────────────────────"))
    print(f"  Detected CPU count: {cpu_count}")
    procs = ask_int("Number of MPI processes", cpu_count // 2 or 1)

    omp = 1
    if code == "QE":
        omp = ask_int("OpenMP threads per MPI task (1 = pure MPI)", 1)

    print()
    print(c("bold", "  ── Step 5 : Output directory ───────────────────────────"))
    base = ask("Output base directory", os.path.join(os.getcwd(),
               "convergence_results"))
    os.makedirs(base, exist_ok=True)

    print()
    print(c("bold", "  ── Step 6 : Select stages to run ───────────────────────"))
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
        stages = list(range(1, max_stage+1))
    else:
        stages = sorted(set(int(x) for x in raw.split() if x.isdigit()))
        stages = [s for s in stages if 1 <= s <= max_stage]

    print()
    print(c("bold", "  ─── Configuration Summary ─────────────────────────────"))
    print(f"  Code           : {code}")
    print(f"  Executable     : {exe}")
    if code == "SIESTA":
        fmt, _, _ = detect_sia_pseudo_format(pseudo_dir)
        print(f"  Pseudo dir     : {pseudo_dir}")
        print(f"  Pseudo format  : {fmt}  (.psml preferred, .psf fallback per species)")
    else:
        print(f"  Pseudo dir     : {pseudo_dir}  (.UPF, auto-selected per element)")
    print(f"  MPI processes  : {procs}")
    if code == "QE":
        print(f"  OMP threads    : {omp}  (total cores used: {procs*omp})")
    print(f"  Output dir     : {base}")
    print(f"  Stages to run  : {stages}")
    print()

    ok = ask("Proceed? (yes/no)", "yes")
    if ok.lower() not in ("yes", "y"):
        print("  Aborted.")
        sys.exit(0)

    return {
        "code":       code,
        "exe":        exe,
        "pseudo_dir": pseudo_dir,
        "procs":      procs,
        "omp":        omp,
        "base":       base,
        "stages":     stages,
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    cfg = interactive_setup()
    code   = cfg["code"]
    stages = cfg["stages"]
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
        if s in runners:
            runners[s](cfg)
        else:
            print(c("warn", f"  Unknown stage {s} – skipped."))

    total = round(time.time() - t_start)
    print_header("ALL DONE")
    print(f"  Total wall time : {total} s  ({total//60} min {total%60} s)")
    print(f"  Results in      : {cfg['base']}/")
    print()
    print("  File structure:")
    if code == "QE":
        print("  ├── qe_01_ecut/       results.dat")
        print("  ├── qe_02_kpoints/    results.dat")
        print("  └── qe_03_eos/        results.dat  eos_fit_curve.dat")
    else:
        print("  ├── siesta_01_meshcutoff/   results.dat")
        print("  ├── siesta_02_kpoints/      results.dat")
        print("  ├── siesta_03_basissize/    results.dat")
        print("  ├── siesta_04_energyshift/  results.dat")
        print("  └── siesta_05_eos/          results.dat  eos_fit_curve.dat")
    print()


if __name__ == "__main__":
    main()
