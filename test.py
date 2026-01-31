"""
Example: Optimise an idiophone bar with all available controls.

This file demonstrates every tuneable parameter in the optimiser.
Adjust the values below to suit your bar, material, and workshop constraints.
"""

from idiophone_tuner import (
    create_bar, get_material, optimize_bar, compute_tuning_guide,
    TuningTarget, UndercutConfig
)

# =========================================================================
# Progress callback (optional)
# =========================================================================
# Called each time the optimiser finds a new best solution.
# Useful for logging or live plotting.

def my_progress(iteration, frequencies, x, cost):
    ratios = frequencies / frequencies[0]
    x_mm = x * 1000
    ratio_str = ":".join(f"{r:.2f}" for r in ratios)
    print(f"  [{iteration}] f1={frequencies[0]:.1f}Hz  "
          f"ratios={ratio_str}  cost={cost:.1f}")
    print(f"          x: {[f'{v:.1f}' for v in x_mm]}")


# =========================================================================
# Material
# =========================================================================
# Built-in: 'sapele', 'padauk', 'rosewood_honduran', 'maple_hard',
#           'ash', 'bubinga', 'purpleheart',
#           'aluminum_6061', 'aluminum_7075', 'bronze_phosphor',
#           'steel_1095', 'brass'
# You can override any property after loading.

material = get_material('sapele')
material.E   = 10.65e9      # Young's modulus (Pa)  — override from testing
material.rho = 662           # Density (kg/m³)       — override from testing
# material.G = 0.73e9       # Shear modulus (Pa)     — rarely needs changing


# =========================================================================
# Bar geometry
# =========================================================================

bar = create_bar(
    length_mm=480,           # Bar length (mm)
    width_mm=32,             # Bar width (mm)
    thickness_mm=24           # Bar thickness / height (mm)
)


# =========================================================================
# Tuning target
# =========================================================================
# Standard ratio presets:
#   xylophone   = (1, 3, 6)        3-mode concert xylophone
#   xylophone_4 = (1, 3, 6, 10)    4-mode xylophone
#   marimba     = (1, 4, 10)        concert marimba
#   vibraphone  = (1, 4)            vibraphone (2 modes)
#   glockenspiel= (1, 2.76, 5.4)   metal glock
#   gamelan     = (1, 2)            gamelan (octave-ish)
#
# weights: relative importance of each mode in the cost function.
#   None = automatic (1, 1/2, 1/3, …) — weights fundamental most.
#   (1, 1, 1) = equal importance for all modes.

target = TuningTarget.from_note(
    'F4',                        # Target fundamental (note name or use f1_target=349.23)
    ratios=(1, 3, 6),           # Desired frequency ratios
    weights=None                 # Per-mode weighting (None = auto)
)


# =========================================================================
# Undercut configuration
# =========================================================================
# Cut pattern presets:
#   .single_center(width_mm)           1 cut
#   .multi_point(n_cuts, width_mm)     1/3/5/7/9 symmetric cuts
#   .xylophone_physics(width_mm)       11 cuts for 1:3:6   (recommended)
#   .xylophone_physics_4mode(width_mm) 15 cuts for 1:3:6:10
#   .xylophone_dense(width_mm)         15 cuts at 5% intervals

config = UndercutConfig.xylophone_physics(width_mm=1.0)

# --- Cut depth constraints ---
config.min_depth_mm       = 0.0    # Floor: minimum cut depth (mm)
config.max_depth_fraction = 0.85   # Ceiling: max depth as fraction of thickness
                                   #   e.g. 0.85 × 24 mm = 20.4 mm max

# --- Cut profile shape ---
# 'flat'       — vertical-sided slot (CNC router / saw)
# 'parabolic'  — smooth arch (hand-carved)
# 'cosine'     — smooth with gentle edges
# 'elliptical' — elliptical arch
config.profile = 'flat'

# --- Bar length constraints ---
# Allow the optimiser to shorten or lengthen the bar to help hit f1.
# Set both to 0 to keep the bar at its original length.
config.max_trim_mm   = 200.0      # Max bar shortening allowed (mm)
config.max_extend_mm = 20.0       # Max bar lengthening allowed (mm)

# --- Optimisation penalties (soft constraints) ---
# These add cost terms that bias the optimiser toward preferred solutions.
# At 0.0 (default) they have no effect.  Increase to steer the solution.
#
# depth_penalty_weight:
#   Penalises deep individual cuts — adds  weight × Σ(depth_mm²)  to cost.
#   Higher values → shallower cuts spread across more positions.
#   Typical range: 0.0 – 1.0
#
# length_penalty_weight:
#   Penalises bar length changes — adds  weight × trim_mm²  to cost.
#   Higher values → keeps the bar closer to its original length.
#   Typical range: 0.0 – 0.5
#
# total_depth_penalty_weight:
#   Penalises total material removal — adds  weight × (Σ depth_mm)²  to cost.
#   Higher values → less total cutting overall.
#   Typical range: 0.0 – 0.1
#
# Tip: to prefer shallower cuts over extreme bar shortening, set
#   depth_penalty_weight  = 0.1   (gentle bias toward shallow cuts)
#   length_penalty_weight = 0.05  (mild resistance to trimming)
# The optimiser will still hit the target frequencies but will favour
# solutions that are less aggressive on both axes.

config.depth_penalty_weight       = 0.0
config.length_penalty_weight      = 0.0
config.total_depth_penalty_weight = 0.0

# --- Total depth hard limit ---
# Hard cap on the sum of all cut depths (mm).  Solutions that exceed
# this are heavily penalised and effectively forbidden.
# None = no limit.
config.max_total_depth_mm = None


# =========================================================================
# Run optimisation
# =========================================================================

if __name__ == '__main__':
    result = optimize_bar(
        bar, material, target, config,

        # -- Mesh resolution --
        n_elements=120,              # Coarse element count between cuts
                                     # Higher = more accurate, slower
        min_elements_per_cut=8,      # Fine elements spanning each cut width
                                     # Increase for very narrow cuts

        # -- Optimisation method --
        method='differential_evolution',  # 'differential_evolution' (global, robust)
                                          # 'SLSQP' (local gradient, fast but needs
                                          #          good starting point)

        # -- Output --
        verbose=True,                # Print progress to stdout
        progress_every=50,           # Status line every N evaluations (0 = off)
        callback=my_progress,        # Custom callback on each improvement (or None)

        # -- Parallelism --
        workers=1,                   # 1 = single-process (full per-eval progress)
                                     # -1 = all CPUs (faster, per-generation progress)
                                     # N  = use N worker processes

        # -- Early stopping --
        convergence_cents=1.0        # Stop when every mode is within ±N cents
                                     # None = run to completion / DE convergence
    )


    # =========================================================================
    # Results
    # =========================================================================

    print(result.summary())


    # =========================================================================
    # Tuning guide for hand-finishing
    # =========================================================================
    # Shows how each cut affects each mode (cents per mm of additional depth).
    # Useful for fine-tuning after CNC roughing.

    guide = compute_tuning_guide(
        result, bar, material, config,
        n_elements=60,               # Mesh resolution for sensitivity calc
        min_elements_per_cut=8,
        delta_mm=0.1                 # Perturbation step for finite-difference
    )
    print(guide)
