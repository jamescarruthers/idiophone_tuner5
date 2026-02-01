"""
Optimizer for idiophone bar tuning.

Optimizes undercut depths to achieve target frequencies and ratios.
"""

import sys
import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
from typing import Optional, Callable
import warnings

from .materials import Material
from .geometry import BarGeometry, Undercut, create_bar, profile_weight
from .timoshenko import modal_analysis, compute_frequencies, fast_compute_frequencies


# =============================================================================
# Musical note utilities
# =============================================================================

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_to_frequency(note: str) -> float:
    """
    Convert note name to frequency.
    
    Examples: 'A4' = 440 Hz, 'C4' = 261.63 Hz, 'F#3' = 185 Hz
    """
    # Parse note name
    note = note.strip().upper()
    
    if '#' in note:
        note_name = note[:2]
        octave = int(note[2:])
    elif 'B' in note[1:]:  # Handle flats
        # Convert flat to sharp
        idx = NOTE_NAMES.index(note[0])
        note_name = NOTE_NAMES[(idx - 1) % 12]
        octave = int(note[2:])
    else:
        note_name = note[0]
        octave = int(note[1:])
    
    # Semitones from A4
    note_idx = NOTE_NAMES.index(note_name)
    a4_idx = NOTE_NAMES.index('A')
    semitones = note_idx - a4_idx + (octave - 4) * 12
    
    # A4 = 440 Hz
    return 440.0 * (2 ** (semitones / 12))


def frequency_to_note(freq: float) -> tuple[str, float]:
    """
    Convert frequency to nearest note name and cents deviation.
    
    Returns:
        (note_name, cents_off) - e.g. ('A4', -5.2)
    """
    # Semitones from A4
    semitones = 12 * np.log2(freq / 440.0)
    semitones_rounded = round(semitones)
    cents = (semitones - semitones_rounded) * 100
    
    # Convert to note name
    a4_idx = NOTE_NAMES.index('A')
    total_idx = a4_idx + semitones_rounded
    octave = 4 + total_idx // 12
    note_idx = total_idx % 12
    
    note_name = f"{NOTE_NAMES[note_idx]}{octave}"
    return note_name, cents


# =============================================================================
# Standard tuning ratios
# =============================================================================

TUNING_RATIOS = {
    'xylophone': (1.0, 3.0, 6.0),      # Western concert xylophone (3 modes)
    'xylophone_4': (1.0, 3.0, 6.0, 10.0),  # 4-mode xylophone
    'marimba': (1.0, 4.0, 10.0),       # Concert marimba
    'vibraphone': (1.0, 4.0, None),    # Mode 3 not tuned
    'glockenspiel': (1.0, 2.76, 5.4),  # Metal bars (close to 1:2.76:5.4)
    'gamelan': (1.0, 2.0, None),       # Indonesian gamelan (octave-ish)
}


@dataclass
class TuningTarget:
    """Defines the target frequencies for bar tuning."""
    f1_target: float                    # Target fundamental (Hz)
    ratios: tuple[float, ...]           # Target frequency ratios (f1:f2:f3:...)
    weights: tuple[float, ...] = None   # Relative importance of each mode
    
    def __post_init__(self):
        if self.weights is None:
            # Default: weight fundamental most, then decreasing
            self.weights = tuple(1.0 / (i + 1) for i in range(len(self.ratios)))
    
    @classmethod
    def from_note(cls, note: str, ratios: tuple[float, ...] = (1.0, 3.0, 6.0),
                  weights: tuple[float, ...] = None) -> 'TuningTarget':
        """Create target from note name."""
        return cls(
            f1_target=note_to_frequency(note),
            ratios=ratios,
            weights=weights
        )
    
    @property
    def target_frequencies(self) -> np.ndarray:
        """Get target frequencies for all modes."""
        return np.array([self.f1_target * r for r in self.ratios])


# =============================================================================
# Free-free beam curvature peaks
# =============================================================================

# Eigenvalues βL for a free-free Euler-Bernoulli beam: cosh(βL)·cos(βL) = 1
_FREE_FREE_BETA_L = [
    4.73004074486,   # mode 1
    7.85320462410,   # mode 2
    10.99560783800,  # mode 3
    14.13716549130,  # mode 4
    17.27875965740,  # mode 5
    20.42035224560,  # mode 6
    23.56194490190,  # mode 7
    26.70353755550,  # mode 8
]


def _free_free_curvature_peaks(n_modes=8, merge_tol=0.03):
    """Optimal cut positions derived from free-free beam curvature.

    For each of the first 8 vibration modes, finds the interior
    positions where |φ''(x)| peaks (the bending-curvature maxima).
    These are physically meaningful positions where material removal has
    the greatest effect on that mode's frequency.

    Positions are then ranked by a *composite effectiveness score*: the
    sum of normalised curvature magnitudes across modes, weighted by
    1/mode_number.  *n_modes* controls how many modes contribute to
    the score — lower values focus positions on the modes being tuned.
    Peaks from all 8 modes are always used as candidates so that enough
    positions are available even when *n_modes* is small.

    Returns a flat list: centre (0.5) first, then symmetric pairs
    (lo, hi) in decreasing order of composite score.  Peaks within
    *merge_tol* of an already-selected one are skipped.
    """
    xi = np.linspace(0, 1, 10001)

    # Always compute curvature for all 8 modes (for candidate positions).
    n_all = len(_FREE_FREE_BETA_L)
    mode_curvatures = []
    for m in range(n_all):
        bL = _FREE_FREE_BETA_L[m]
        s = (np.cosh(bL) - np.cos(bL)) / (np.sinh(bL) - np.sin(bL))
        curv = np.abs(
            np.cosh(bL * xi) - np.cos(bL * xi)
            - s * (np.sinh(bL * xi) - np.sin(bL * xi))
        )
        curv /= curv.max()                 # normalise to 0-1
        mode_curvatures.append(curv)

    # Composite score uses only the first n_modes for ranking.
    composite = np.zeros_like(xi)
    for m in range(min(n_modes, n_all)):
        composite += mode_curvatures[m] / (m + 1)

    # Collect *individual mode* curvature peaks from ALL modes (interior,
    # 8–92%).  Each candidate is scored by its composite effectiveness
    # (which only weights the first n_modes).
    candidates = []
    for curv in mode_curvatures:
        for i in range(1, len(curv) - 1):
            if curv[i] > curv[i - 1] and curv[i] > curv[i + 1]:
                p = xi[i]
                if 0.08 < p < 0.92:
                    candidates.append((p, composite[i]))

    # Highest composite score first.
    candidates.sort(key=lambda t: -t[1])

    # Select positions: centre first, then pairs by composite score.
    result = [0.5]
    seen = {0.5}

    for pos, _ in candidates:
        pos = round(pos, 3)
        if abs(pos - 0.5) < 0.015:
            continue  # centre already included
        mirror = round(1.0 - pos, 3)
        lo, hi = min(pos, mirror), max(pos, mirror)

        # Skip if too close to an existing position.
        if any(abs(lo - s) < merge_tol for s in seen):
            continue

        seen.update([lo, hi])
        result.extend([lo, hi])

    return result


def _timoshenko_curvature_peaks(bar, material, n_modes=8, merge_tol=0.03):
    """Optimal cut positions from Timoshenko FEM curvature peaks.

    Runs a modal analysis on the uniform (uncut) bar and computes bending
    curvature κ = dψ/dx from the rotation DOFs.  Peak positions are ranked
    by a composite effectiveness score exactly as in
    ``_free_free_curvature_peaks``, but using the actual Timoshenko mode
    shapes which account for shear deformation and rotary inertia.

    For materials with a high E/G ratio (wood ≈ 14) the higher-mode
    curvature peaks shift noticeably compared to Euler-Bernoulli theory.

    All available modes (up to 8) are used to find candidate positions;
    *n_modes* controls how many modes contribute to the composite score
    that ranks them.
    """
    from .timoshenko import modal_analysis as _modal_analysis

    # Run FEM on the uniform bar (no undercuts).
    n_all = 8
    n_elem = 200  # fine enough for smooth curvature
    result = _modal_analysis(bar, material, n_elements=n_elem,
                             n_modes=n_all + 2)  # extra margin for rigid body filtering
    n_avail = min(n_all, len(result.frequencies))

    L = bar.length
    nodes = result.nodes
    # Element lengths
    L_e = np.diff(nodes)
    # Mid-element positions as fraction of length
    x_mid = (nodes[:-1] + L_e / 2) / L

    # Compute normalised curvature for all available modes.
    mode_curvatures = []
    for m in range(n_avail):
        mode = result.get_mode(m + 1)  # 1-indexed
        psi = mode.rotations            # rotation at each node
        # Curvature ≈ dψ/dx at element midpoints
        curv = np.abs(np.diff(psi) / L_e)
        mx = curv.max()
        if mx > 0:
            curv = curv / mx            # normalise to 0-1
        mode_curvatures.append(curv)

    # Composite score: weight only the first n_modes for ranking.
    composite = np.zeros_like(x_mid)
    for m in range(min(n_modes, n_avail)):
        composite += mode_curvatures[m] / (m + 1)

    # Collect individual-mode curvature peaks from ALL modes.
    candidates = []
    for curv in mode_curvatures:
        for i in range(1, len(curv) - 1):
            if curv[i] > curv[i - 1] and curv[i] > curv[i + 1]:
                p = float(x_mid[i])
                if 0.08 < p < 0.92:
                    candidates.append((p, float(composite[i])))

    candidates.sort(key=lambda t: -t[1])

    # Select: centre first, then symmetric pairs by score.
    result_pos = [0.5]
    seen = {0.5}
    for pos, _ in candidates:
        pos = round(pos, 3)
        if abs(pos - 0.5) < 0.015:
            continue
        mirror = round(1.0 - pos, 3)
        lo, hi = min(pos, mirror), max(pos, mirror)
        if any(abs(lo - s) < merge_tol for s in seen):
            continue
        seen.update([lo, hi])
        result_pos.extend([lo, hi])

    return result_pos


@dataclass
class UndercutConfig:
    """Configuration for narrow slot undercuts."""
    positions: list[float]          # Positions as fractions of length (0-1)
    width_mm: float = 1.0           # Width of each slot (narrow cuts)
    min_depth_mm: float = 0.0       # Minimum depth
    max_depth_fraction: float = 0.85  # Maximum depth as fraction of thickness
    profile: str = 'flat'           # 'flat' or 'parabolic'
    max_trim_mm: float = 0.0       # Max amount the bar may be shortened (mm)
    max_extend_mm: float = 0.0     # Max amount the bar may be lengthened (mm)
    depth_penalty_weight: float = 0.0       # Soft penalty on individual cut depths
    length_penalty_weight: float = 0.0      # Soft penalty on bar length changes
    total_depth_penalty_weight: float = 0.0 # Soft penalty on aggregate cut depth
    max_total_depth_mm: Optional[float] = None  # Hard limit on sum of all cut depths (mm)
    
    @classmethod
    def from_n_cuts(cls, n_cuts: int, width_mm: float = 1.0, *,
                    bar=None, material=None,
                    n_modes: Optional[int] = None) -> 'UndercutConfig':
        """Create config with *n_cuts* cuts at beam curvature peaks.

        Positions are computed from the bending-curvature peaks of a
        uniform free-free beam — the locations where material removal
        has the greatest effect on each mode's frequency.

        If *bar* and *material* are provided, a Timoshenko FEM analysis
        is used to find the curvature peaks.  This accounts for shear
        deformation and rotary inertia, which shifts higher-mode peaks
        noticeably for wood (E/G ≈ 14).  Without them, the analytical
        Euler-Bernoulli solution is used (universal, material-independent).

        *n_modes* controls how many vibration modes contribute to the
        composite effectiveness score that ranks cut positions.  Lower
        values focus positions on the modes you actually need to tune.
        Default is 8 (E-B) or the lesser of 8 and available modes
        (Timoshenko).

        The centre (0.50) is always included.  Additional cuts are added
        as symmetric pairs, prioritising the most effective positions.

        *n_cuts* must be odd (centre + symmetric pairs).  Even values
        are rounded up.

        Recommended values::

             5  (3 independent) — 2–3 mode control
             7  (4 independent) — 3 mode control
             9  (5 independent) — 3–4 mode control
            11  (6 independent) — 4–5 mode control
            13  (7 independent) — 5–6 mode control
            15  (8 independent) — 6–7 mode control
        """
        if n_cuts < 1:
            raise ValueError("n_cuts must be at least 1")
        if n_cuts % 2 == 0:
            n_cuts += 1

        nm = n_modes if n_modes is not None else 8

        if bar is not None and material is not None:
            all_positions = _timoshenko_curvature_peaks(bar, material,
                                                        n_modes=nm)
        else:
            all_positions = _free_free_curvature_peaks(n_modes=nm)

        positions = all_positions[:n_cuts]

        return cls(
            positions=sorted(positions),
            width_mm=width_mm,
            max_depth_fraction=0.85
        )


@dataclass
class OptimizationResult:
    """Results from the optimization."""
    success: bool
    optimized_geometry: BarGeometry
    achieved_frequencies: np.ndarray
    target_frequencies: np.ndarray
    undercut_depths_mm: np.ndarray
    frequency_errors_cents: np.ndarray
    ratio_errors: np.ndarray
    n_iterations: int
    message: str = ""
    optimized_length_mm: Optional[float] = None
    optim_n_elements: Optional[int] = None
    verified_frequencies: Optional[np.ndarray] = None
    verified_errors_cents: Optional[np.ndarray] = None
    verified_n_elements: Optional[int] = None

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("OPTIMIZATION RESULT")
        lines.append("=" * 60)
        lines.append(f"Status: {'SUCCESS' if self.success else 'FAILED'}")
        lines.append(f"Iterations: {self.n_iterations}")
        if self.message:
            lines.append(f"Message: {self.message}")
        lines.append("")

        if self.optimized_length_mm is not None:
            lines.append(f"BAR LENGTH: {self.optimized_length_mm:.2f} mm")
            lines.append("")

        lines.append("UNDERCUT DEPTHS:")
        bar_length_mm = self.optimized_geometry.length * 1000
        for i, depth in enumerate(self.undercut_depths_mm):
            pos = self.optimized_geometry.undercuts[i].center
            pos_mm = pos * bar_length_mm
            lines.append(f"  Cut {i+1} at {pos:.1%} ({pos_mm:.1f} mm): {depth:.2f} mm")
        total_depth = np.sum(self.undercut_depths_mm)
        max_depth = np.max(self.undercut_depths_mm)
        lines.append(f"  ---")
        lines.append(f"  Total cut depth: {total_depth:.1f} mm  "
                     f"(deepest: {max_depth:.2f} mm)")
        lines.append("")
        
        lines.append("FREQUENCIES:")
        has_verify = self.verified_frequencies is not None
        if has_verify:
            lines.append(f"  {'Mode':<8} {'Target':>10} {'Optim.':>10} {'Error':>8}"
                         f"  {'Hi-Res':>10} {'Error':>8}")
            lines.append(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8}"
                         f"  {'-'*10} {'-'*8}")
            for i in range(len(self.achieved_frequencies)):
                lines.append(
                    f"  Mode {i+1:<3}"
                    f" {self.target_frequencies[i]:>10.2f}"
                    f" {self.achieved_frequencies[i]:>10.2f}"
                    f" {self.frequency_errors_cents[i]:>+8.1f}¢"
                    f"  {self.verified_frequencies[i]:>10.2f}"
                    f" {self.verified_errors_cents[i]:>+8.1f}¢"
                )
            lines.append("")
            optim_el = self.optim_n_elements or '?'
            lines.append(f"  Optim. = optimiser mesh ({optim_el} elements)")
            lines.append(f"  Hi-Res = verification mesh ({self.verified_n_elements} elements)")
        else:
            lines.append(f"  {'Mode':<8} {'Target':>10} {'Achieved':>10} {'Error':>10}")
            lines.append(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
            for i in range(len(self.achieved_frequencies)):
                lines.append(f"  Mode {i+1:<3} {self.target_frequencies[i]:>10.2f} "
                            f"{self.achieved_frequencies[i]:>10.2f} {self.frequency_errors_cents[i]:>+10.1f}¢")
        lines.append("")

        # Achieved ratios
        f1 = self.achieved_frequencies[0]
        achieved_ratios = self.achieved_frequencies / f1
        target_ratios = self.target_frequencies / self.target_frequencies[0]
        lines.append("RATIOS:")
        lines.append(f"  Target:   {' : '.join(f'{r:.2f}' for r in target_ratios)}")
        lines.append(f"  Achieved: {' : '.join(f'{r:.2f}' for r in achieved_ratios)}")
        if has_verify:
            v_ratios = self.verified_frequencies / self.verified_frequencies[0]
            lines.append(f"  Hi-Res:   {' : '.join(f'{r:.2f}' for r in v_ratios)}")
        lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class _FastObjective:
    """Picklable objective function for multiprocessing.

    Pre-computes the depth influence matrix and FEM mesh so each evaluation
    is just a matrix-vector product + vectorized eigenvalue solve.

    When *sym_matrix* is provided the optimiser works in a reduced variable
    space where symmetric cut pairs share a single depth variable.

    When *optimize_length* is True the first element of the input vector is
    a length offset (metres) and element lengths are scaled accordingly.
    The depth influence matrix is recomputed for the current bar length so
    that the undercut profile weights stay correct.
    """

    def __init__(self, depth_matrix, base_thickness, width, L_e, E, G, rho,
                 n_modes, target_freqs, weights, ratios, sym_matrix=None,
                 optimize_length=False, base_length=None,
                 depth_penalty_weight=0.0, length_penalty_weight=0.0,
                 total_depth_penalty_weight=0.0, max_total_depth_mm=None,
                 positions=None, width_m=None, min_elements_per_cut=8,
                 max_element_m=0.005, cut_profile='flat'):
        # Static depth matrix (used when length is fixed).
        if sym_matrix is not None:
            self.eff_depth_matrix = depth_matrix @ sym_matrix
        else:
            self.eff_depth_matrix = depth_matrix
        self.sym_matrix = sym_matrix
        self.base_thickness = base_thickness
        self.width = width
        self.L_e = L_e
        self.E = E
        self.G = G
        self.rho = rho
        self.n_modes = n_modes
        self.target_freqs = target_freqs
        self.weights = weights
        self.ratios = ratios
        self.optimize_length = optimize_length
        self.base_length = base_length
        self.depth_penalty_weight = depth_penalty_weight
        self.length_penalty_weight = length_penalty_weight
        self.total_depth_penalty_weight = total_depth_penalty_weight
        self.max_total_depth_mm = max_total_depth_mm
        # For rebuilding mesh when length changes.
        self.positions = positions
        self.width_m = width_m
        self.min_elements_per_cut = min_elements_per_cut
        self.max_element_m = max_element_m
        self.cut_profile = cut_profile

    def __call__(self, x):
        if self.optimize_length:
            length_offset = x[0]
            depths = x[1:]
            cur_length = self.base_length + length_offset
            ratio = cur_length / self.base_length
            # Rebuild mesh at current bar length so element boundaries
            # align with undercut edges at this length.
            cur_nodes = _build_adaptive_nodes(
                self.positions, self.width_m, cur_length,
                min_per_cut=self.min_elements_per_cut,
                max_element_m=self.max_element_m * ratio)
            dm, L_e = _build_depth_matrix(
                self.positions, self.width_m, self.cut_profile,
                cur_length, cur_nodes)
            if self.sym_matrix is not None:
                h = self.base_thickness - (dm @ self.sym_matrix) @ depths
            else:
                h = self.base_thickness - dm @ depths
        else:
            length_offset = 0.0
            depths = x
            L_e = self.L_e
            h = self.base_thickness - self.eff_depth_matrix @ depths
        np.maximum(h, 1e-6, out=h)
        A = self.width * h
        I_val = self.width * h ** 3 / 12.0

        try:
            freqs = fast_compute_frequencies(
                L_e, self.E, self.G, self.rho, A, I_val, self.n_modes)
        except Exception:
            return 1e10

        if len(freqs) < self.n_modes:
            return 1e10

        cents_errors = 1200.0 * np.log2(freqs / self.target_freqs)
        cost = np.sum(self.weights * cents_errors ** 2)

        if len(freqs) >= 2:
            achieved_ratios = freqs / freqs[0]
            ratio_errors = (achieved_ratios - self.ratios) / self.ratios
            cost += 100.0 * np.sum(ratio_errors[1:] ** 2)

        # --- Penalty terms ---
        # Expand independent depths to full (per-cut) depths for penalties.
        if self.sym_matrix is not None:
            full_depths = self.sym_matrix @ depths
        else:
            full_depths = depths

        if self.depth_penalty_weight > 0:
            full_depths_mm = full_depths * 1000
            cost += self.depth_penalty_weight * np.sum(full_depths_mm ** 2)

        if self.total_depth_penalty_weight > 0:
            total_mm = np.sum(full_depths) * 1000
            cost += self.total_depth_penalty_weight * total_mm ** 2

        if self.length_penalty_weight > 0 and self.optimize_length:
            trim_mm = length_offset * 1000
            cost += self.length_penalty_weight * trim_mm ** 2

        if self.max_total_depth_mm is not None:
            total_mm = np.sum(full_depths) * 1000
            excess = total_mm - self.max_total_depth_mm
            if excess > 0:
                cost += 1e6 * excess ** 2

        return cost


def _build_symmetry_map(positions, tol=1e-6):
    """Identify symmetric position pairs and build a mapping matrix.

    Positions that are mirrors about 0.5 (i.e. ``pos_a + pos_b ≈ 1.0``)
    are grouped together so they share a single depth variable during
    optimisation.  This enforces the physical constraint that an idiophone
    bar's undercuts are symmetric about its centre.

    Returns:
        sym_matrix: (n_cuts, n_independent) array.
            ``full_depths = sym_matrix @ independent_depths``
        groups: list of lists – each inner list contains the cut indices
            that share one independent variable.
    """
    n = len(positions)
    assigned = [False] * n
    groups = []

    for i in range(n):
        if assigned[i]:
            continue
        mirror = 1.0 - positions[i]
        found_partner = False
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if abs(positions[j] - mirror) < tol:
                groups.append([i, j])
                assigned[i] = True
                assigned[j] = True
                found_partner = True
                break
        if not found_partner:
            groups.append([i])
            assigned[i] = True

    n_independent = len(groups)
    sym_matrix = np.zeros((n, n_independent))
    for g_idx, group in enumerate(groups):
        for cut_idx in group:
            sym_matrix[cut_idx, g_idx] = 1.0

    return sym_matrix, groups


def _build_adaptive_nodes(positions, width_m, bar_length, min_per_cut=8,
                          max_element_m=0.005):
    """Build a non-uniform node array that is refined inside undercuts.

    For narrow cuts the FEM needs several elements across each cut to
    resolve the abrupt thickness change accurately.  Between cuts a
    coarser mesh is sufficient.

    Args:
        positions: Cut centre positions as fractions of bar length.
        width_m: Cut width in metres.
        bar_length: Bar length in metres.
        min_per_cut: Minimum number of elements spanning each cut.
        max_element_m: Maximum element size in coarse regions (metres).

    Returns:
        1-D array of node positions (metres), sorted and unique.
    """
    cut_element = width_m / min_per_cut
    half_w = width_m / 2.0

    # Collect critical boundaries (cut edges + bar ends).
    boundaries = {0.0, bar_length}
    for pos in positions:
        center = pos * bar_length
        boundaries.add(max(0.0, center - half_w))
        boundaries.add(min(bar_length, center + half_w))
    boundaries = sorted(boundaries)

    # Merge boundaries that are closer together than the fine element size.
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] > cut_element * 0.5:
            merged.append(b)
    if merged[-1] != bar_length:
        merged[-1] = bar_length

    # Subdivide each interval with the appropriate element size.
    all_nodes = [merged[0]]
    for i in range(len(merged) - 1):
        start = merged[i]
        end = merged[i + 1]
        span = end - start

        mid = (start + end) / 2.0
        in_cut = any(abs(mid - p * bar_length) <= half_w + 1e-9
                     for p in positions)

        target = cut_element if in_cut else max_element_m
        n_sub = max(1, round(span / target))

        sub_nodes = np.linspace(start, end, n_sub + 1)
        all_nodes.extend(sub_nodes[1:].tolist())

    return np.array(all_nodes)


def _profile_weights(dx, half_w, profile):
    """Compute normalised (0-1) profile weights for an array of distances.

    Delegates to :func:`geometry.profile_weight` (the single source of truth
    for undercut profile shapes).

    *dx* is an array of signed distances from the undercut centre (metres).
    Returns an array the same shape as *dx* with weights in [0, 1].
    Points outside the undercut (|dx| > half_w) get weight 0.
    """
    return profile_weight(dx / half_w, profile)


def _build_depth_matrix(positions, width_m, profile, bar_length, nodes):
    """Build the depth influence matrix for vectorized thickness computation.

    Returns an (n_elements, n_cuts) matrix where entry [j, i] gives the
    profile weight of undercut i at element midpoint j.
    For flat profiles this is 0 or 1; for shaped profiles it's the
    profile envelope value.

    *nodes* is a 1-D array of FEM node positions (metres) — typically
    produced by :func:`_build_adaptive_nodes`.

    Thickness at any midpoint is: h = base_thickness - depth_matrix @ depths
    """
    x_mid = (nodes[:-1] + nodes[1:]) / 2.0  # element midpoints
    L_e = np.diff(nodes)
    n_elements = len(L_e)
    n_cuts = len(positions)

    depth_matrix = np.zeros((n_elements, n_cuts))
    half_w = width_m / 2.0

    for i in range(n_cuts):
        center_m = positions[i] * bar_length
        dx = x_mid - center_m
        depth_matrix[:, i] = _profile_weights(dx, half_w, profile)

    return depth_matrix, L_e


def optimize_bar(
    base_geometry: BarGeometry,
    material: Material,
    target: TuningTarget,
    undercut_config: UndercutConfig,
    n_elements: int = 100,
    method: str = 'differential_evolution',
    verbose: bool = True,
    progress_every: int = 20,
    callback: Optional[Callable] = None,
    workers: int = 1,
    min_elements_per_cut: int = 8,
    convergence_cents: Optional[float] = None,
    verify_n_elements: Optional[int] = None
) -> OptimizationResult:
    """
    Optimize undercut depths to achieve target frequencies.

    The FEM mesh is built adaptively: dense elements (at least
    *min_elements_per_cut*) inside each undercut for accuracy, and
    coarser elements (governed by *n_elements*) in between.

    Args:
        base_geometry: Initial bar geometry (without undercuts)
        material: Material properties
        target: Target frequencies and ratios
        undercut_config: Undercut positions and constraints
        n_elements: Controls coarse mesh spacing between cuts
        method: 'differential_evolution' (global) or 'SLSQP' (local gradient)
        verbose: Print progress
        progress_every: Print status every N evaluations (0 to disable)
        callback: Optional callback(iteration, frequencies, depths, cost)
        workers: Number of parallel workers for differential_evolution.
            1 = single-process (full progress tracking).
            -1 = use all CPU cores (faster, per-generation progress only).
        convergence_cents: Stop early when all mode frequencies are within
            this many cents of the target. None disables early stopping.
        verify_n_elements: If set, run a high-resolution verification after
            optimisation using this many coarse elements.  The summary will
            show both the optimiser-mesh and high-res results side by side.

    Returns:
        OptimizationResult with optimized geometry and achieved frequencies
    """
    n_cuts = len(undercut_config.positions)
    n_modes = len(target.ratios)

    # Maximum depth constraint
    max_depth = base_geometry.thickness * undercut_config.max_depth_fraction

    # Build adaptive FEM mesh: dense inside cuts, coarse between.
    base_length = base_geometry.length
    width_m = undercut_config.width_mm / 1000
    max_element_m = base_length / max(n_elements, 1)

    nodes = _build_adaptive_nodes(
        undercut_config.positions, width_m, base_length,
        min_per_cut=min_elements_per_cut,
        max_element_m=max_element_m
    )

    # Pre-compute depth influence matrix and element lengths
    depth_matrix, L_e = _build_depth_matrix(
        undercut_config.positions,
        width_m,
        undercut_config.profile,
        base_length,
        nodes
    )
    cut_positions = undercut_config.positions
    cut_profile = undercut_config.profile

    # Enforce symmetric cuts: mirror-pairs share one depth variable.
    sym_matrix, sym_groups = _build_symmetry_map(undercut_config.positions)
    effective_depth_matrix = depth_matrix @ sym_matrix
    n_independent = sym_matrix.shape[1]

    # Length optimisation: optionally prepend a length-offset variable.
    optimize_length = (undercut_config.max_trim_mm > 0
                       or undercut_config.max_extend_mm > 0)
    length_bound = (-undercut_config.max_trim_mm / 1000,
                    undercut_config.max_extend_mm / 1000)

    # Bounds: [length_offset (optional), depth1, depth2, ...]
    depth_bounds = [(undercut_config.min_depth_mm / 1000, max_depth)
                    for _ in range(n_independent)]
    bounds = ([length_bound] + depth_bounds) if optimize_length else depth_bounds

    base_thickness = base_geometry.thickness
    width = base_geometry.width
    E = material.E
    G = material.G
    rho = material.rho

    # Pre-compute cost function constants
    target_freqs = target.target_frequencies
    weights = np.array(target.weights[:n_modes], dtype=float)
    weights = weights / weights.sum()
    target_ratios = np.array(target.ratios[:n_modes])

    # Penalty config (captured in closures below).
    dpw = undercut_config.depth_penalty_weight
    lpw = undercut_config.length_penalty_weight
    tdpw = undercut_config.total_depth_penalty_weight
    mtd = undercut_config.max_total_depth_mm

    def _compute_cost_and_freqs(x):
        """Shared cost computation used by both single- and multi-process paths.

        *x* is the full optimisation vector: an optional length offset
        followed by the reduced (independent) depth variables.
        """
        if optimize_length:
            length_offset = x[0]
            depths = x[1:]
            cur_length = base_length + length_offset
            ratio = cur_length / base_length
            # Rebuild mesh at current bar length so element boundaries
            # align with undercut edges at this length.
            cur_nodes = _build_adaptive_nodes(
                cut_positions, width_m, cur_length,
                min_per_cut=min_elements_per_cut,
                max_element_m=max_element_m * ratio)
            dm, cur_L_e = _build_depth_matrix(
                cut_positions, width_m, cut_profile, cur_length, cur_nodes)
            h = base_thickness - (dm @ sym_matrix) @ depths
        else:
            length_offset = 0.0
            depths = x
            cur_L_e = L_e
            h = base_thickness - effective_depth_matrix @ depths
        np.maximum(h, 1e-6, out=h)
        A = width * h
        I_val = width * h ** 3 / 12.0

        freqs = fast_compute_frequencies(cur_L_e, E, G, rho, A, I_val, n_modes)
        if len(freqs) < n_modes:
            return 1e10, None

        cents_errors = 1200.0 * np.log2(freqs / target_freqs)
        cost = np.sum(weights * cents_errors ** 2)

        if len(freqs) >= 2:
            achieved_ratios = freqs / freqs[0]
            ratio_errors = (achieved_ratios - target_ratios) / target_ratios
            cost += 100.0 * np.sum(ratio_errors[1:] ** 2)

        # --- Penalty terms ---
        full_depths = sym_matrix @ depths

        if dpw > 0:
            full_depths_mm = full_depths * 1000
            cost += dpw * np.sum(full_depths_mm ** 2)

        if tdpw > 0:
            total_mm = np.sum(full_depths) * 1000
            cost += tdpw * total_mm ** 2

        if lpw > 0 and optimize_length:
            trim_mm = length_offset * 1000
            cost += lpw * trim_mm ** 2

        if mtd is not None:
            total_mm = np.sum(full_depths) * 1000
            excess = total_mm - mtd
            if excess > 0:
                cost += 1e6 * excess ** 2

        return cost, freqs

    # --- single-process: closure with full progress tracking ---
    eval_count = [0]
    best_cost = [np.inf]
    best_freqs = [None]
    best_x = [None]
    converged = [False]

    def objective(x: np.ndarray) -> float:
        eval_count[0] += 1

        if converged[0]:
            return best_cost[0]

        try:
            cost, freqs = _compute_cost_and_freqs(x)
        except Exception:
            return 1e10

        if freqs is None:
            return 1e10

        is_new_best = cost < best_cost[0]

        if is_new_best:
            best_cost[0] = cost
            best_freqs[0] = freqs.copy()
            best_x[0] = x.copy()

            if convergence_cents is not None:
                cents_errs = 1200.0 * np.log2(freqs / target_freqs)
                if np.all(np.abs(cents_errs) <= convergence_cents):
                    converged[0] = True

        show_progress = verbose and progress_every > 0 and (
            is_new_best or eval_count[0] % progress_every == 0
        )

        if show_progress:
            ratios = freqs / freqs[0]
            marker = "★ NEW BEST" if is_new_best else ""

            length_info = ""
            if optimize_length:
                cur_len_mm = (base_length + x[0]) * 1000
                length_info = f"L={cur_len_mm:.1f}mm  "

            ratio_str = ":".join(f"{r:.2f}" for r in ratios)
            print(f"[{eval_count[0]:4d}] {length_info}f1={freqs[0]:6.1f}Hz  "
                  f"ratios={ratio_str}  "
                  f"cost={cost:8.1f}  {marker}")
            sys.stdout.flush()

        if is_new_best and callback:
            callback(eval_count[0], freqs, x, cost)

        return cost

    n_total_elements = len(L_e)
    min_elem_mm = L_e.min() * 1000
    max_elem_mm = L_e.max() * 1000

    if verbose:
        print(f"Optimizing {n_cuts} undercuts ({n_independent} independent, "
              f"{n_cuts - n_independent} symmetric) for target "
              f"{target.f1_target:.1f} Hz")
        if optimize_length:
            print(f"Length optimisation: trim up to {undercut_config.max_trim_mm:.1f} mm, "
                  f"extend up to {undercut_config.max_extend_mm:.1f} mm  "
                  f"(base {base_length * 1000:.1f} mm)")
        print(f"Adaptive mesh: {n_total_elements} elements "
              f"(element size {min_elem_mm:.2f}–{max_elem_mm:.2f} mm)")
        note, _ = frequency_to_note(target.f1_target)
        print(f"Target note: {note}")
        print(f"Target ratios: {target.ratios}")
        if workers != 1:
            import os
            n_cpus = os.cpu_count() if workers == -1 else workers
            print(f"Using {n_cpus} parallel workers")
        penalties = []
        if dpw > 0:
            penalties.append(f"depth={dpw}")
        if lpw > 0:
            penalties.append(f"length={lpw}")
        if tdpw > 0:
            penalties.append(f"total_depth={tdpw}")
        if mtd is not None:
            penalties.append(f"max_total_depth={mtd:.1f}mm")
        if penalties:
            print(f"Penalties: {', '.join(penalties)}")
        print("")

    # When convergence_cents is set, disable DE's built-in population
    # convergence so it only stops via our per-mode cents callback.
    de_tol = 0 if convergence_cents is not None else 1e-4

    # Run optimization
    if method == 'differential_evolution':
        use_parallel = workers != 1

        if use_parallel:
            # Multiprocessing: use picklable objective, deferred updating
            fast_obj = _FastObjective(
                depth_matrix, base_thickness, width, L_e, E, G, rho,
                n_modes, target_freqs, weights, target_ratios,
                sym_matrix=sym_matrix,
                optimize_length=optimize_length,
                base_length=base_length,
                depth_penalty_weight=dpw,
                length_penalty_weight=lpw,
                total_depth_penalty_weight=tdpw,
                max_total_depth_mm=mtd,
                positions=cut_positions,
                width_m=width_m,
                min_elements_per_cut=min_elements_per_cut,
                max_element_m=max_element_m,
                cut_profile=cut_profile)

            de_gen_count = [0]

            def _de_callback(xk, convergence):
                de_gen_count[0] += 1
                freqs = None
                if verbose or convergence_cents is not None:
                    try:
                        cost, freqs = _compute_cost_and_freqs(xk)
                    except Exception:
                        pass

                if verbose and freqs is not None:
                    ratios = freqs / freqs[0]
                    ratio_str = ":".join(f"{r:.2f}" for r in ratios)
                    print(f"[gen {de_gen_count[0]:3d}] f1={freqs[0]:6.1f}Hz  "
                          f"ratios={ratio_str}  "
                          f"cost={cost:8.1f}  conv={convergence:.6f}")
                    sys.stdout.flush()

                if convergence_cents is not None and freqs is not None:
                    cents_errs = 1200.0 * np.log2(freqs / target_freqs)
                    if np.all(np.abs(cents_errs) <= convergence_cents):
                        converged[0] = True
                        if verbose:
                            print(f"\nConverged: all modes within "
                                  f"+/-{convergence_cents:.1f} cents of target")
                        return True

            result = differential_evolution(
                fast_obj,
                bounds=bounds,
                maxiter=500,
                tol=de_tol,
                seed=42,
                polish=True,
                workers=workers,
                mutation=(0.5, 1.0),
                recombination=0.7,
                popsize=10,
                updating='deferred',
                callback=_de_callback
            )
        else:
            # Single process: use closure with full progress, immediate updating
            def _sp_callback(xk, convergence):
                if converged[0]:
                    if verbose:
                        print(f"\nConverged: all modes within "
                              f"+/-{convergence_cents:.1f} cents of target")
                    return True

            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=500,
                tol=de_tol,
                seed=42,
                polish=True,
                workers=1,
                mutation=(0.5, 1.0),
                recombination=0.7,
                popsize=10,
                updating='immediate',
                callback=_sp_callback if convergence_cents is not None else None
            )

        optimal_result_x = result.x
        success = result.success or converged[0]
        message = result.message
        n_iter = result.nit

    elif method == 'SLSQP':
        # Local gradient-based optimizer
        x0_depths = [max_depth * 0.3 for _ in range(n_independent)]
        x0 = np.array(([0.0] + x0_depths) if optimize_length else x0_depths)
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )
        optimal_result_x = result.x
        success = result.success
        message = result.message
        n_iter = result.nit

        if converged[0] and verbose:
            print(f"\nConverged: all modes within "
                  f"+/-{convergence_cents:.1f} cents of target")

    else:
        raise ValueError(f"Unknown method: {method}")

    # If converged early, use the best tracked solution (the optimizer
    # may have continued evaluating after the flag was set).
    if converged[0] and best_x[0] is not None:
        optimal_result_x = best_x[0]

    # Split optimisation result into length offset and independent depths.
    if optimize_length:
        optimal_length_offset = optimal_result_x[0]
        optimal_independent = optimal_result_x[1:]
        optimized_length = base_length + optimal_length_offset
    else:
        optimal_independent = optimal_result_x
        optimized_length = base_length

    # Expand independent depths back to per-cut depths (symmetric pairs
    # receive the same value).
    optimal_depths = sym_matrix @ optimal_independent

    # Build final geometry (with optimised length if applicable)
    final_bar = BarGeometry(
        length=optimized_length,
        width=base_geometry.width,
        thickness=base_geometry.thickness,
        undercuts=[]
    )
    final_undercuts = [
        Undercut(
            center=undercut_config.positions[i],
            width=undercut_config.width_mm / 1000,
            depth=optimal_depths[i],
            profile=undercut_config.profile
        )
        for i in range(n_cuts)
    ]
    final_geometry = final_bar.copy_with_undercuts(final_undercuts)

    # Compute final frequencies using the same code path as the optimiser
    # to guarantee the summary matches the convergence check.
    _, final_freqs = _compute_cost_and_freqs(optimal_result_x)
    target_freqs_arr = target.target_frequencies

    # Compute errors
    cents_errors = 1200 * np.log2(final_freqs / target_freqs_arr)
    achieved_ratios = final_freqs / final_freqs[0]
    target_ratios_arr = np.array(target.ratios[:n_modes])
    ratio_errors = (achieved_ratios - target_ratios_arr) / target_ratios_arr

    # Optional high-resolution verification on a finer mesh.
    verified_freqs = None
    verified_cents = None
    if verify_n_elements is not None:
        verify_max_el = optimized_length / max(verify_n_elements, 1)
        verify_nodes = _build_adaptive_nodes(
            undercut_config.positions, width_m, optimized_length,
            min_per_cut=min_elements_per_cut,
            max_element_m=verify_max_el
        )
        vdm, v_L_e = _build_depth_matrix(
            undercut_config.positions, width_m, undercut_config.profile,
            optimized_length, verify_nodes
        )
        v_h = base_thickness - vdm @ optimal_depths
        np.maximum(v_h, 1e-6, out=v_h)
        v_A = width * v_h
        v_I = width * v_h ** 3 / 12.0
        verified_freqs = fast_compute_frequencies(
            v_L_e, E, G, rho, v_A, v_I, n_modes
        )
        verified_cents = 1200 * np.log2(verified_freqs / target_freqs_arr)
        if verbose:
            print(f"High-resolution verification: {len(v_L_e)} elements "
                  f"(element size {np.min(v_L_e)*1000:.2f}–"
                  f"{np.max(v_L_e)*1000:.2f} mm)")

    return OptimizationResult(
        success=success and all(abs(cents_errors) < 50),  # Within 50 cents
        optimized_geometry=final_geometry,
        achieved_frequencies=final_freqs,
        target_frequencies=target_freqs_arr,
        undercut_depths_mm=optimal_depths * 1000,
        frequency_errors_cents=cents_errors,
        ratio_errors=ratio_errors,
        n_iterations=n_iter,
        message=message,
        optimized_length_mm=optimized_length * 1000 if optimize_length else None,
        optim_n_elements=len(L_e),
        verified_frequencies=verified_freqs,
        verified_errors_cents=verified_cents,
        verified_n_elements=len(v_L_e) if verified_freqs is not None else None
    )


def compute_tuning_guide(
    result: OptimizationResult,
    base_geometry: BarGeometry,
    material: Material,
    undercut_config: UndercutConfig,
    n_elements: int = 60,
    min_elements_per_cut: int = 8,
    delta_mm: float = 0.1,
) -> str:
    """Compute a practical tuning guide for hand-finishing an optimised bar.

    For each independent cut (symmetric pairs grouped), this perturbs the
    depth by *delta_mm* and reports the resulting cents change for every mode.
    This tells the maker which cuts to deepen (or leave shy) to nudge each
    frequency in the desired direction.

    Args:
        result: A completed :class:`OptimizationResult`.
        base_geometry: The original bar geometry (pre-optimisation).
        material: Material properties.
        undercut_config: The undercut config used during optimisation.
        n_elements: Coarse mesh parameter (same as in optimise call).
        min_elements_per_cut: Fine mesh elements per cut.
        delta_mm: Perturbation size in mm for sensitivity calculation.

    Returns:
        A formatted multi-line string ready for printing.
    """
    n_cuts = len(undercut_config.positions)
    n_modes = len(result.achieved_frequencies)

    # Recover optimised bar length.
    if result.optimized_length_mm is not None:
        bar_length = result.optimized_length_mm / 1000
    else:
        bar_length = base_geometry.length

    width_m = undercut_config.width_mm / 1000
    max_element_m = bar_length / max(n_elements, 1)

    # Build FEM mesh at the optimised length.
    nodes = _build_adaptive_nodes(
        undercut_config.positions, width_m, bar_length,
        min_per_cut=min_elements_per_cut,
        max_element_m=max_element_m,
    )
    depth_matrix, L_e = _build_depth_matrix(
        undercut_config.positions, width_m,
        undercut_config.profile, bar_length, nodes,
    )
    sym_matrix, sym_groups = _build_symmetry_map(undercut_config.positions)

    base_thickness = base_geometry.thickness
    bar_width = base_geometry.width
    E, G, rho = material.E, material.G, material.rho

    # Optimal depths per cut (metres).
    optimal_depths = result.undercut_depths_mm / 1000

    def _freqs_at(depths_m):
        """Compute modal frequencies for a given depth vector."""
        h = base_thickness - depth_matrix @ depths_m
        np.maximum(h, 1e-6, out=h)
        A = bar_width * h
        I_val = bar_width * h ** 3 / 12.0
        return fast_compute_frequencies(L_e, E, G, rho, A, I_val, n_modes)

    # Reference frequencies at the optimised state.
    f_ref = _freqs_at(optimal_depths)

    # Sensitivity: cents change per mm of additional depth for each group.
    delta_m = delta_mm / 1000
    sensitivities = []  # list of arrays, one per group

    for g_idx, group in enumerate(sym_groups):
        perturbed = optimal_depths.copy()
        for cut_idx in group:
            perturbed[cut_idx] += delta_m
        f_pert = _freqs_at(perturbed)

        # cents change per mm
        cents_per_mm = 1200.0 * np.log2(f_pert / f_ref) / delta_mm
        sensitivities.append(cents_per_mm)

    # Also compute length sensitivity if length was optimised.
    length_sens = None
    if result.optimized_length_mm is not None:
        len_delta_mm = 0.5
        len_delta_m = len_delta_mm / 1000
        longer_length = bar_length + len_delta_m
        nodes_l = _build_adaptive_nodes(
            undercut_config.positions, width_m, longer_length,
            min_per_cut=min_elements_per_cut,
            max_element_m=max_element_m,
        )
        dm_l, Le_l = _build_depth_matrix(
            undercut_config.positions, width_m,
            undercut_config.profile, longer_length, nodes_l,
        )
        h_l = base_thickness - dm_l @ optimal_depths
        np.maximum(h_l, 1e-6, out=h_l)
        A_l = bar_width * h_l
        I_l = bar_width * h_l ** 3 / 12.0
        f_longer = fast_compute_frequencies(Le_l, E, G, rho, A_l, I_l, n_modes)
        length_sens = 1200.0 * np.log2(f_longer / f_ref) / len_delta_mm

    # --- Format the guide ---
    lines = []
    lines.append("=" * 70)
    lines.append("TUNING GUIDE")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Sensitivity: cents change per 1 mm deeper cut (negative = pitch drops)")
    lines.append("Start ~2 mm shy of target depth, then deepen to tune.")
    lines.append("")

    # Header
    mode_hdrs = [f"{'Mode '+str(i+1):>8s}" for i in range(n_modes)]
    hdr = f"  {'Cut':20s} {'Depth':>7s}  " + "  ".join(mode_hdrs)
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for g_idx, group in enumerate(sym_groups):
        positions = [undercut_config.positions[i] for i in group]
        depth_mm = result.undercut_depths_mm[group[0]]

        if len(group) == 1:
            label = f"{positions[0]:.1%} (centre)"
        else:
            label = f"{positions[0]:.1%} + {positions[1]:.1%}"

        sens = sensitivities[g_idx]
        vals = "  ".join(f"{s:+8.1f}" for s in sens)
        lines.append(f"  {label:20s} {depth_mm:6.1f}mm  {vals}")

    if length_sens is not None:
        lines.append("")
        vals = "  ".join(f"{s:+8.1f}" for s in length_sens)
        lines.append(f"  {'Bar length':20s} {result.optimized_length_mm:6.1f}mm  {vals}")
        lines.append(f"  {'':20s} {'':>7s}  (cents per 1 mm longer)")

    # --- Practical advice section ---
    lines.append("")
    lines.append("-" * 70)
    lines.append("HOW TO USE")
    lines.append("-" * 70)
    lines.append("")

    # For each mode, find the most influential cut.
    for m in range(n_modes):
        best_g = max(range(len(sym_groups)),
                     key=lambda g: abs(sensitivities[g][m]))
        group = sym_groups[best_g]
        positions = [undercut_config.positions[i] for i in group]
        s = sensitivities[best_g][m]

        if len(group) == 1:
            cut_label = f"centre cut ({positions[0]:.0%})"
        else:
            cut_label = f"cuts at {positions[0]:.0%}/{positions[1]:.0%}"

        direction = "drops" if s < 0 else "rises"
        lines.append(f"  Mode {m+1}: Most sensitive to {cut_label} "
                     f"({s:+.1f} ¢/mm — pitch {direction})")

    if length_sens is not None:
        lines.append("")
        lines.append("  Trimming the bar (shorter) raises all pitches; "
                     "all modes scale roughly together.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def spread_cuts(
    result: OptimizationResult,
    base_geometry: BarGeometry,
    material: Material,
    target: TuningTarget,
    original_config: UndercutConfig,
    n_per_side: int = 2,
    spacing_mm: Optional[float] = None,
    merge_tol_mm: Optional[float] = None,
    **optimize_kwargs,
) -> OptimizationResult:
    """Replace each optimised cut with multiple shallower cuts.

    After :func:`optimize_bar` finds ideal single-cut positions and
    depths, this function expands each position into a group of
    ``2 * n_per_side + 1`` cuts spread evenly around the ideal
    location.  The depths are then re-optimised so the expanded cuts
    collectively produce the same target frequencies.

    The result is a bar with many shallow cuts instead of a few deep
    ones, which is less sensitive to positioning errors and easier to
    produce in a workshop.

    Args:
        result: The output of a previous :func:`optimize_bar` call.
        base_geometry: The original bar geometry (pre-optimisation).
        material: Material properties.
        target: The tuning target (same as the first optimisation).
        original_config: The undercut config used in the first pass.
        n_per_side: Number of extra cuts on *each* side of every
            original position (total per group = ``2 * n_per_side + 1``).
        spacing_mm: Centre-to-centre distance between adjacent sub-cuts
            (mm).  Defaults to the cut width.
        merge_tol_mm: Sub-cuts from different groups closer than this
            are merged into one position.  Defaults to half the cut
            width.
        **optimize_kwargs: Forwarded to :func:`optimize_bar` (e.g.
            ``n_elements``, ``method``, ``workers``, ``verbose``,
            ``convergence_cents``, ``verify_n_elements``).

    Returns:
        A new :class:`OptimizationResult` with the spread-cut geometry.
    """
    if n_per_side < 1:
        raise ValueError("n_per_side must be at least 1")
    if spacing_mm is None:
        spacing_mm = original_config.width_mm
    if merge_tol_mm is None:
        merge_tol_mm = original_config.width_mm * 0.5

    # Use the optimised bar length from the first pass.
    if result.optimized_length_mm is not None:
        bar_length_m = result.optimized_length_mm / 1000
    else:
        bar_length_m = base_geometry.length

    spacing_frac = (spacing_mm / 1000) / bar_length_m
    merge_tol_frac = (merge_tol_mm / 1000) / bar_length_m

    # Expand each original position into a group of sub-cuts.
    original_positions = sorted(original_config.positions)
    expanded = []
    for pos in original_positions:
        for k in range(-n_per_side, n_per_side + 1):
            new_pos = pos + k * spacing_frac
            # Keep within a safe interior margin.
            if 0.02 <= new_pos <= 0.98:
                expanded.append(round(new_pos, 6))

    # Sort and merge positions that are too close together.
    expanded.sort()
    merged = [expanded[0]]
    for p in expanded[1:]:
        if p - merged[-1] >= merge_tol_frac:
            merged.append(p)
    new_positions = merged

    n_original = len(original_positions)
    n_spread = len(new_positions)

    # Build a new UndercutConfig with spread positions.
    new_config = UndercutConfig(
        positions=new_positions,
        width_mm=original_config.width_mm,
        min_depth_mm=original_config.min_depth_mm,
        max_depth_fraction=original_config.max_depth_fraction,
        profile=original_config.profile,
        # Fix bar length — don't re-optimise length in the spread pass.
        max_trim_mm=0.0,
        max_extend_mm=0.0,
        depth_penalty_weight=original_config.depth_penalty_weight,
        length_penalty_weight=0.0,
        total_depth_penalty_weight=original_config.total_depth_penalty_weight,
        max_total_depth_mm=original_config.max_total_depth_mm,
    )

    # Create the base bar at the optimised length.
    spread_bar = BarGeometry(
        length=bar_length_m,
        width=base_geometry.width,
        thickness=base_geometry.thickness,
        undercuts=[],
    )

    if 'verbose' not in optimize_kwargs:
        optimize_kwargs['verbose'] = True
    if optimize_kwargs.get('verbose', False):
        cuts_per = 2 * n_per_side + 1
        print(f"\nSPREADING CUTS: {n_original} original → {n_spread} spread "
              f"(up to {cuts_per} per group, spacing {spacing_mm:.1f} mm)")
        print(f"Spread positions: {[f'{p:.1%}' for p in new_positions]}")
        print("")

    # Re-optimise depths for the spread configuration.
    return optimize_bar(
        spread_bar, material, target, new_config,
        **optimize_kwargs,
    )


def find_initial_length(
    width_mm: float,
    thickness_mm: float,
    material: Material,
    target_f1: float,
    tolerance_hz: float = 1.0
) -> float:
    """
    Find the bar length that gives approximately the target fundamental.
    
    Uses a plain (uncut) bar and bisection search.
    
    Returns:
        Estimated length in mm
    """
    # Starting estimate from Euler-Bernoulli theory
    # f1 ≈ (beta1)^2 / (2*pi*L^2) * sqrt(EI / rho*A)
    # For free-free, beta1*L ≈ 4.73
    
    E = material.E
    rho = material.rho
    h = thickness_mm / 1000
    b = width_mm / 1000
    I = b * h**3 / 12
    A = b * h
    
    beta1_L = 4.73
    L_estimate = beta1_L * (E * I / (rho * A * (2 * np.pi * target_f1)**2))**0.25
    
    # Bisection search to refine
    L_low = L_estimate * 0.5
    L_high = L_estimate * 1.5
    
    for _ in range(20):
        L_mid = (L_low + L_high) / 2
        bar = create_bar(L_mid * 1000, width_mm, thickness_mm)
        freqs = compute_frequencies(bar, material, n_elements=50, n_modes=1)
        f1 = freqs[0]
        
        if abs(f1 - target_f1) < tolerance_hz:
            break
        
        if f1 > target_f1:
            L_low = L_mid
        else:
            L_high = L_mid
    
    return L_mid * 1000  # Return in mm


def quick_analysis(
    length_mm: float,
    width_mm: float,
    thickness_mm: float,
    material_name: str,
    n_modes: int = 5
) -> dict:
    """
    Quick analysis of a bar without optimization.
    
    Returns dict with frequencies, ratios, and mode info.
    """
    from .materials import get_material
    
    material = get_material(material_name)
    bar = create_bar(length_mm, width_mm, thickness_mm)
    
    result = modal_analysis(bar, material, n_elements=100, n_modes=n_modes)
    
    freqs = result.frequencies
    ratios = freqs / freqs[0]
    
    notes = [frequency_to_note(f) for f in freqs]
    
    return {
        'frequencies_hz': freqs,
        'ratios': ratios,
        'notes': [n[0] for n in notes],
        'cents_off': [n[1] for n in notes],
        'material': material,
        'geometry': bar,
        'modal_result': result
    }
