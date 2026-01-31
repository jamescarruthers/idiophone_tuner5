from idiophone_tuner import (
    create_bar, get_material, optimize_bar,
    TuningTarget, UndercutConfig
)

def my_progress(iteration, frequencies, depths, cost):
    """Called on every improvement."""
    ratios = frequencies / frequencies[0]
    depths_mm = depths * 1000
    
    print(f"[{iteration}] f1={frequencies[0]:.1f}Hz, ratios={ratios[1]:.2f}:{ratios[2]:.2f}, cost={cost:.1f}")
    print(f"        depths: {[f'{d:.1f}' for d in depths_mm]}")

material = get_material('sapele')
bar = create_bar(480, 32, 24)
target = TuningTarget.from_note('F4', ratios=(1, 3, 6))

# Use the physics-based 11-cut config
config = UndercutConfig.xylophone_physics(width_mm=2.0)

# Or the dense 15-cut config
# config = UndercutConfig.xylophone_dense(width_mm=1.0)

result = optimize_bar(bar, material, target, config, n_elements=480, verbose=True, callback=my_progress, progress_every=20)
print(result.summary())

