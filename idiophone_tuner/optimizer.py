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
from .geometry import BarGeometry, Undercut, create_bar
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
    'xylophone': (1.0, 3.0, 6.0),      # Western concert xylophone
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


@dataclass
class UndercutConfig:
    """Configuration for narrow slot undercuts."""
    positions: list[float]          # Positions as fractions of length (0-1)
    width_mm: float = 1.0           # Width of each slot (narrow cuts)
    min_depth_mm: float = 0.0       # Minimum depth
    max_depth_fraction: float = 0.85  # Maximum depth as fraction of thickness
    profile: str = 'flat'           # 'flat' or 'parabolic'
    
    @classmethod
    def single_center(cls, width_mm: float = 1.0) -> 'UndercutConfig':
        """Single center cut."""
        return cls(positions=[0.5], width_mm=width_mm)
    
    @classmethod
    def multi_point(cls, n_cuts: int = 5, width_mm: float = 1.0) -> 'UndercutConfig':
        """
        Multiple narrow cuts for tuning.
        
        More cuts = more control over mode ratios.
        Positions chosen to affect different modes:
        - Center (0.5): affects mode 1, 3 strongly
        - 0.3/0.7: affects mode 2
        - 0.15/0.85: affects mode 3
        """
        if n_cuts == 1:
            positions = [0.5]
        elif n_cuts == 3:
            positions = [0.5, 0.3, 0.7]
        elif n_cuts == 5:
            positions = [0.5, 0.35, 0.65, 0.2, 0.8]
        elif n_cuts == 7:
            positions = [0.5, 0.35, 0.65, 0.25, 0.75, 0.15, 0.85]
        elif n_cuts == 9:
            positions = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.12, 0.88]
        else:
            # Symmetric distribution
            half = n_cuts // 2
            positions = [0.5]
            for i in range(1, half + 1):
                offset = 0.15 + 0.3 * i / half
                positions.extend([0.5 - offset, 0.5 + offset])
            positions = positions[:n_cuts]
        
        return cls(
            positions=positions,
            width_mm=width_mm,
            max_depth_fraction=0.85
        )
    
    @classmethod
    def xylophone_physics(cls, width_mm: float = 1.0) -> 'UndercutConfig':
        """
        Physics-optimized positions for xylophone (1:3:6) tuning.
        
        Based on mode shape curvature analysis:
        
        Mode 1 (fundamental): Max curvature at CENTER (0.5)
            - Nodes at 0.224, 0.776
            
        Mode 2: Max curvature at ~0.25, 0.75 (between its nodes)
            - Nodes at 0.132, 0.5, 0.868
            
        Mode 3: Max curvature at ~0.17, 0.5, 0.83
            - Nodes at 0.094, 0.356, 0.644, 0.906
        
        To RAISE the ratios f2/f1 and f3/f1, we need f1 to drop MORE 
        than f2 and f3. Strategy:
        
        - Heavy cuts at 0.5 (max mode 1 curvature, also mode 3)
        - Cuts at 0.4, 0.6 (still high mode 1, less mode 2)  
        - Cuts at 0.3, 0.7 (mode 2 region - use sparingly)
        - Cuts at 0.2, 0.8 (transition)
        - Cuts near 0.15, 0.85 (mode 3 curvature, past mode 1 nodes)
        
        11 cuts gives good resolution for independent mode control.
        """
        positions = [
            0.5,              # Center - affects f1 and f3 strongly
            0.45, 0.55,       # Near center - strong f1 effect
            0.38, 0.62,       # f1 region, some f2
            0.30, 0.70,       # f2 curvature region  
            0.22, 0.78,       # Near f1 nodes - minimal f1 effect, affects f2/f3
            0.14, 0.86,       # f3 curvature region, past f1 nodes
        ]
        
        return cls(
            positions=positions,
            width_mm=width_mm,
            max_depth_fraction=0.85
        )
    
    @classmethod
    def xylophone_dense(cls, width_mm: float = 1.0) -> 'UndercutConfig':
        """
        Dense cut pattern for maximum tuning control.
        
        15 cuts at 5% intervals from 15% to 85% of bar length.
        Symmetric about center.
        """
        positions = [0.5]
        for offset in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            positions.extend([0.5 - offset, 0.5 + offset])
        
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
        
        lines.append("UNDERCUT DEPTHS:")
        for i, depth in enumerate(self.undercut_depths_mm):
            pos = self.optimized_geometry.undercuts[i].center
            lines.append(f"  Cut {i+1} at {pos:.1%}: {depth:.2f} mm")
        lines.append("")
        
        lines.append("FREQUENCIES:")
        lines.append(f"  {'Mode':<8} {'Target':>10} {'Achieved':>10} {'Error':>10}")
        lines.append(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for i in range(len(self.achieved_frequencies)):
            note, cents = frequency_to_note(self.achieved_frequencies[i])
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
        lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class _FastObjective:
    """Picklable objective function for multiprocessing.

    Pre-computes the depth influence matrix and FEM mesh so each evaluation
    is just a matrix-vector product + vectorized eigenvalue solve.

    When *sym_matrix* is provided the optimiser works in a reduced variable
    space where symmetric cut pairs share a single depth variable.
    ``effective_depth_matrix = depth_matrix @ sym_matrix`` so the input
    vector has length ``n_independent`` rather than ``n_cuts``.
    """

    def __init__(self, depth_matrix, base_thickness, width, L_e, E, G, rho,
                 n_modes, target_freqs, weights, ratios, sym_matrix=None):
        if sym_matrix is not None:
            self.depth_matrix = depth_matrix @ sym_matrix
        else:
            self.depth_matrix = depth_matrix
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

    def __call__(self, depths):
        h = self.base_thickness - self.depth_matrix @ depths
        np.maximum(h, 1e-6, out=h)
        A = self.width * h
        I_val = self.width * h ** 3 / 12.0

        try:
            freqs = fast_compute_frequencies(
                self.L_e, self.E, self.G, self.rho, A, I_val, self.n_modes)
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


def _build_depth_matrix(positions, width_m, profile, bar_length, n_elements):
    """Build the depth influence matrix for vectorized thickness computation.

    Returns an (n_elements, n_cuts) matrix where entry [j, i] gives the
    profile weight of undercut i at element midpoint j.
    For flat profiles this is 0 or 1; for shaped profiles it's the
    profile envelope value.

    Thickness at any midpoint is: h = base_thickness - depth_matrix @ depths
    """
    nodes = np.linspace(0, bar_length, n_elements + 1)
    x_mid = (nodes[:-1] + nodes[1:]) / 2.0  # meters
    L_e = np.diff(nodes)
    n_cuts = len(positions)

    depth_matrix = np.zeros((n_elements, n_cuts))
    half_w = width_m / 2.0

    for i in range(n_cuts):
        center_m = positions[i] * bar_length
        dx = x_mid - center_m
        mask = np.abs(dx) <= half_w
        if not np.any(mask):
            continue

        if profile == 'flat':
            depth_matrix[mask, i] = 1.0
        elif profile == 'parabolic':
            u = dx[mask] / half_w
            depth_matrix[mask, i] = 1.0 - u ** 2
        elif profile == 'cosine':
            u = dx[mask] / half_w
            depth_matrix[mask, i] = 0.5 * (1.0 + np.cos(np.pi * u))
        elif profile == 'elliptical':
            u = dx[mask] / half_w
            depth_matrix[mask, i] = np.sqrt(1.0 - u ** 2)

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
    workers: int = 1
) -> OptimizationResult:
    """
    Optimize undercut depths to achieve target frequencies.

    Args:
        base_geometry: Initial bar geometry (without undercuts)
        material: Material properties
        target: Target frequencies and ratios
        undercut_config: Undercut positions and constraints
        n_elements: FEM mesh density
        method: 'differential_evolution' (global) or 'SLSQP' (local gradient)
        verbose: Print progress
        progress_every: Print status every N evaluations (0 to disable)
        callback: Optional callback(iteration, frequencies, depths, cost)
        workers: Number of parallel workers for differential_evolution.
            1 = single-process (full progress tracking).
            -1 = use all CPU cores (faster, per-generation progress only).

    Returns:
        OptimizationResult with optimized geometry and achieved frequencies
    """
    n_cuts = len(undercut_config.positions)
    n_modes = len(target.ratios)

    # Maximum depth constraint
    max_depth = base_geometry.thickness * undercut_config.max_depth_fraction

    # Pre-compute depth influence matrix and element lengths
    depth_matrix, L_e = _build_depth_matrix(
        undercut_config.positions,
        undercut_config.width_mm / 1000,
        undercut_config.profile,
        base_geometry.length,
        n_elements
    )

    # Enforce symmetric cuts: mirror-pairs share one depth variable.
    sym_matrix, sym_groups = _build_symmetry_map(undercut_config.positions)
    effective_depth_matrix = depth_matrix @ sym_matrix
    n_independent = sym_matrix.shape[1]

    # Bounds for optimization – one per independent variable (meters)
    bounds = [(undercut_config.min_depth_mm / 1000, max_depth)
              for _ in range(n_independent)]

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

    def _compute_cost_and_freqs(depths):
        """Shared cost computation used by both single- and multi-process paths.

        *depths* is in the reduced (independent) variable space.
        """
        h = base_thickness - effective_depth_matrix @ depths
        np.maximum(h, 1e-6, out=h)
        A = width * h
        I_val = width * h ** 3 / 12.0

        freqs = fast_compute_frequencies(L_e, E, G, rho, A, I_val, n_modes)
        if len(freqs) < n_modes:
            return 1e10, None

        cents_errors = 1200.0 * np.log2(freqs / target_freqs)
        cost = np.sum(weights * cents_errors ** 2)

        if len(freqs) >= 2:
            achieved_ratios = freqs / freqs[0]
            ratio_errors = (achieved_ratios - target_ratios) / target_ratios
            cost += 100.0 * np.sum(ratio_errors[1:] ** 2)

        return cost, freqs

    # --- single-process: closure with full progress tracking ---
    eval_count = [0]
    best_cost = [np.inf]
    best_freqs = [None]
    best_depths = [None]

    def objective(depths: np.ndarray) -> float:
        eval_count[0] += 1

        try:
            cost, freqs = _compute_cost_and_freqs(depths)
        except Exception:
            return 1e10

        if freqs is None:
            return 1e10

        is_new_best = cost < best_cost[0]

        if is_new_best:
            best_cost[0] = cost
            best_freqs[0] = freqs.copy()
            best_depths[0] = depths.copy()

        show_progress = verbose and progress_every > 0 and (
            is_new_best or eval_count[0] % progress_every == 0
        )

        if show_progress:
            ratios = freqs / freqs[0]
            marker = "★ NEW BEST" if is_new_best else ""

            print(f"[{eval_count[0]:4d}] f1={freqs[0]:6.1f}Hz  "
                  f"ratios=1:{ratios[1]:.2f}:{ratios[2]:.2f}  "
                  f"cost={cost:8.1f}  {marker}")
            sys.stdout.flush()

        if is_new_best and callback:
            callback(eval_count[0], freqs, depths, cost)

        return cost

    if verbose:
        print(f"Optimizing {n_cuts} undercuts ({n_independent} independent, "
              f"{n_cuts - n_independent} symmetric) for target "
              f"{target.f1_target:.1f} Hz")
        note, _ = frequency_to_note(target.f1_target)
        print(f"Target note: {note}")
        print(f"Target ratios: {target.ratios}")
        if workers != 1:
            import os
            n_cpus = os.cpu_count() if workers == -1 else workers
            print(f"Using {n_cpus} parallel workers")
        print("")

    # Run optimization
    if method == 'differential_evolution':
        use_parallel = workers != 1

        if use_parallel:
            # Multiprocessing: use picklable objective, deferred updating
            fast_obj = _FastObjective(
                depth_matrix, base_thickness, width, L_e, E, G, rho,
                n_modes, target_freqs, weights, target_ratios,
                sym_matrix=sym_matrix)

            de_gen_count = [0]

            def _de_callback(xk, convergence):
                de_gen_count[0] += 1
                if verbose:
                    try:
                        cost, freqs = _compute_cost_and_freqs(xk)
                        if freqs is not None:
                            ratios = freqs / freqs[0]
                            print(f"[gen {de_gen_count[0]:3d}] f1={freqs[0]:6.1f}Hz  "
                                  f"ratios=1:{ratios[1]:.2f}:{ratios[2]:.2f}  "
                                  f"cost={cost:8.1f}  conv={convergence:.6f}")
                            sys.stdout.flush()
                    except Exception:
                        pass

            result = differential_evolution(
                fast_obj,
                bounds=bounds,
                maxiter=500,
                tol=1e-4,
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
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=500,
                tol=1e-4,
                seed=42,
                polish=True,
                workers=1,
                mutation=(0.5, 1.0),
                recombination=0.7,
                popsize=10,
                updating='immediate'
            )

        optimal_independent = result.x
        success = result.success
        message = result.message
        n_iter = result.nit

    elif method == 'SLSQP':
        # Local gradient-based optimizer
        x0 = np.array([max_depth * 0.3 for _ in range(n_independent)])
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )
        optimal_independent = result.x
        success = result.success
        message = result.message
        n_iter = result.nit

    else:
        raise ValueError(f"Unknown method: {method}")

    # Expand independent depths back to per-cut depths (symmetric pairs
    # receive the same value).
    optimal_depths = sym_matrix @ optimal_independent

    # Build final geometry
    final_undercuts = [
        Undercut(
            center=undercut_config.positions[i],
            width=undercut_config.width_mm / 1000,
            depth=optimal_depths[i],
            profile=undercut_config.profile
        )
        for i in range(n_cuts)
    ]
    final_geometry = base_geometry.copy_with_undercuts(final_undercuts)

    # Compute final frequencies
    final_freqs = compute_frequencies(final_geometry, material, n_elements, n_modes)
    target_freqs_arr = target.target_frequencies

    # Compute errors
    cents_errors = 1200 * np.log2(final_freqs / target_freqs_arr)
    achieved_ratios = final_freqs / final_freqs[0]
    target_ratios_arr = np.array(target.ratios[:n_modes])
    ratio_errors = (achieved_ratios - target_ratios_arr) / target_ratios_arr

    return OptimizationResult(
        success=success and all(abs(cents_errors) < 50),  # Within 50 cents
        optimized_geometry=final_geometry,
        achieved_frequencies=final_freqs,
        target_frequencies=target_freqs_arr,
        undercut_depths_mm=optimal_depths * 1000,
        frequency_errors_cents=cents_errors,
        ratio_errors=ratio_errors,
        n_iterations=n_iter,
        message=message
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
