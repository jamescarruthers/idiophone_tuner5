from idiophone_tuner import (
    create_bar, get_material, optimize_bar,
    TuningTarget, UndercutConfig
)

def my_progress(iteration, frequencies, x, cost):
    """Called on every improvement."""
    ratios = frequencies / frequencies[0]
    x_mm = x * 1000
    ratio_str = ":".join(f"{r:.2f}" for r in ratios)

    print(f"[{iteration}] f1={frequencies[0]:.1f}Hz, ratios={ratio_str}, cost={cost:.1f}")
    print(f"        x: {[f'{v:.1f}' for v in x_mm]}")

material = get_material('sapele')
bar = create_bar(480, 32, 24)
target_3mode = TuningTarget.from_note('F4', ratios=(1, 3, 6))

# --- Test 1: 3-mode tuning with 3-mode cuts (11 cuts, 6 independent) ---
print("=" * 70)
print("TEST 1: 3-mode target with 3-mode cuts (11 cuts, 6 independent)")
print("=" * 70)
config_3 = UndercutConfig.xylophone_physics(width_mm=2.0)
config_3.max_trim_mm = 20.0
config_3.max_extend_mm = 20.0

result_3 = optimize_bar(bar, material, target_3mode, config_3, n_elements=60,
                        verbose=True, callback=my_progress, progress_every=50)
print(result_3.summary())

# --- Test 2: 3-mode tuning with 4-mode cuts (15 cuts, 8 independent) ---
print("\n")
print("=" * 70)
print("TEST 2: 3-mode target with 4-mode cuts (15 cuts, 8 independent)")
print("=" * 70)
config_4 = UndercutConfig.xylophone_physics_4mode(width_mm=2.0)
config_4.max_trim_mm = 20.0
config_4.max_extend_mm = 20.0

result_4 = optimize_bar(bar, material, target_3mode, config_4, n_elements=60,
                        verbose=True, callback=my_progress, progress_every=50)
print(result_4.summary())

# --- Comparison ---
print("\n")
print("=" * 70)
print("COMPARISON")
print("=" * 70)
import numpy as np
for label, r in [("3-mode cuts", result_3), ("4-mode cuts", result_4)]:
    errs = r.frequency_errors_cents
    max_err = np.max(np.abs(errs))
    rms_err = np.sqrt(np.mean(errs**2))
    ratios = r.achieved_frequencies / r.achieved_frequencies[0]
    ratio_str = ":".join(f"{v:.3f}" for v in ratios)
    length_str = f"  length={r.optimized_length_mm:.1f}mm" if r.optimized_length_mm else ""
    print(f"  {label:12s}: ratios={ratio_str}  max_err={max_err:.1f}¢  rms={rms_err:.1f}¢{length_str}")
