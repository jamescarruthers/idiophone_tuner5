"""
Idiophone Bar Tuner

FEM-based optimization of idiophone bars (xylophones, marimbas, glockenspiels).
Uses Timoshenko beam theory to accurately predict frequencies including
shear deformation and rotary inertia effects.

Quick start:
    from idiophone_tuner import quick_analysis, optimize_bar
    from idiophone_tuner import create_bar, get_material, TuningTarget, UndercutConfig
    
    # Analyze a bar
    result = quick_analysis(350, 32, 24, 'sapele')
    print(result['frequencies_hz'])
    
    # Optimize for F4 xylophone tuning
    bar = create_bar(350, 32, 24)
    material = get_material('sapele')
    target = TuningTarget.from_note('F4', ratios=(1, 3, 6))
    config = UndercutConfig.multi_point(n_cuts=3)
    result = optimize_bar(bar, material, target, config)
    print(result.summary())
"""

from .materials import (
    Material, 
    get_material, 
    list_materials, 
    custom_material,
    SAPELE, PADAUK, ROSEWOOD_HONDURAN, MAPLE_HARD,
    ALUMINUM_6061, BRONZE_PHOSPHOR, STEEL_1095, BRASS
)

from .geometry import (
    BarGeometry,
    Undercut,
    create_bar,
    add_undercut,
    xylophone_undercut_positions,
    marimba_undercut_positions
)

from .timoshenko import (
    modal_analysis,
    compute_frequencies,
    ModalAnalysisResult,
    ModeResult,
    euler_bernoulli_free_free_frequencies
)

from .optimizer import (
    TuningTarget,
    UndercutConfig,
    OptimizationResult,
    optimize_bar,
    compute_tuning_guide,
    find_initial_length,
    quick_analysis,
    note_to_frequency,
    frequency_to_note,
    TUNING_RATIOS
)

__version__ = '0.1.0'
__author__ = 'Claude'

__all__ = [
    # Materials
    'Material', 'get_material', 'list_materials', 'custom_material',
    'SAPELE', 'PADAUK', 'ROSEWOOD_HONDURAN', 'MAPLE_HARD',
    'ALUMINUM_6061', 'BRONZE_PHOSPHOR', 'STEEL_1095', 'BRASS',
    
    # Geometry
    'BarGeometry', 'Undercut', 'create_bar', 'add_undercut',
    'xylophone_undercut_positions', 'marimba_undercut_positions',
    
    # FEM
    'modal_analysis', 'compute_frequencies', 
    'ModalAnalysisResult', 'ModeResult',
    'euler_bernoulli_free_free_frequencies',
    
    # Optimization
    'TuningTarget', 'UndercutConfig', 'OptimizationResult',
    'optimize_bar', 'compute_tuning_guide', 'find_initial_length', 'quick_analysis',
    'note_to_frequency', 'frequency_to_note', 'TUNING_RATIOS',
]
