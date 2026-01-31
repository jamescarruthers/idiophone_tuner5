from idiophone_tuner import (
    create_bar, get_material, optimize_bar, compute_tuning_guide,
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

material.E = 10.65e9  # Pa
material.rho = 662  # kg/m³

bar = create_bar(480, 32, 24)
target = TuningTarget.from_note('F4', ratios=(1, 3, 6))

config = UndercutConfig.xylophone_physics(width_mm=1.0)
config.max_trim_mm = 200.0
config.max_extend_mm = 20.0

result = optimize_bar(bar, material, target, config, n_elements=120,
                      verbose=True, callback=my_progress, progress_every=50,
                      convergence_cents=1.0)
print(result.summary())

guide = compute_tuning_guide(result, bar, material, config, n_elements=60)
print(guide)
