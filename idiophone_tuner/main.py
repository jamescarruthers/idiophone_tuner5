#!/usr/bin/env python3
"""
Idiophone Bar Tuner - CLI Interface

Optimize bar dimensions and undercuts to achieve target frequencies and ratios.
"""

import argparse
import numpy as np
from typing import Optional

from .materials import get_material, list_materials, Material
from .geometry import create_bar, BarGeometry
from .timoshenko import modal_analysis, compute_frequencies, euler_bernoulli_free_free_frequencies
from .optimizer import (
    TuningTarget, UndercutConfig, optimize_bar, spread_cuts,
    find_initial_length, quick_analysis,
    note_to_frequency, frequency_to_note, TUNING_RATIOS
)


def analyze_command(args):
    """Analyze a bar and report its natural frequencies."""
    result = quick_analysis(
        args.length, args.width, args.thickness, 
        args.material, n_modes=args.modes
    )
    
    print("\n" + "=" * 60)
    print("BAR ANALYSIS")
    print("=" * 60)
    print(f"Material: {result['material'].name}")
    print(f"  E = {result['material'].E/1e9:.2f} GPa")
    print(f"  G = {result['material'].G/1e9:.2f} GPa")
    print(f"  ρ = {result['material'].rho:.0f} kg/m³")
    print(f"\nDimensions: {args.length:.1f} × {args.width:.1f} × {args.thickness:.1f} mm")
    print("")
    
    print("NATURAL FREQUENCIES:")
    print(f"  {'Mode':<6} {'Freq (Hz)':>10} {'Note':>8} {'Cents':>8} {'Ratio':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    
    for i in range(len(result['frequencies_hz'])):
        print(f"  {i+1:<6} {result['frequencies_hz'][i]:>10.2f} "
              f"{result['notes'][i]:>8} {result['cents_off'][i]:>+8.1f} "
              f"{result['ratios'][i]:>8.2f}")
    
    print("")
    
    # Show comparison with standard tunings
    print("COMPARISON WITH STANDARD TUNINGS:")
    for name, ratios in TUNING_RATIOS.items():
        if ratios[2] is not None:
            match = all(abs(result['ratios'][i] - ratios[i]) < 0.2 for i in range(min(3, len(result['ratios']))))
            status = "✓ close" if match else "  "
            print(f"  {name:12}: 1:{ratios[1]:.1f}:{ratios[2]:.1f} {status}")
        else:
            print(f"  {name:12}: 1:{ratios[1]:.1f}:—")
    
    print("=" * 60)


def optimize_command(args):
    """Optimize undercuts to achieve target frequency and ratios."""
    material = get_material(args.material)
    
    # Parse target frequency
    if args.note:
        target_f1 = note_to_frequency(args.note)
        print(f"Target note: {args.note} = {target_f1:.2f} Hz")
    else:
        target_f1 = args.frequency
    
    # Parse ratios
    if args.ratios:
        ratios = tuple(float(x) for x in args.ratios.split(':'))
    elif args.tuning:
        ratios = TUNING_RATIOS[args.tuning]
        # Filter out None values
        ratios = tuple(r for r in ratios if r is not None)
    else:
        ratios = (1.0, 3.0, 6.0)  # Default xylophone
    
    print(f"Target ratios: {':'.join(f'{r:.1f}' for r in ratios)}")
    
    # Find initial length if not provided
    if args.length:
        length_mm = args.length
    else:
        print("\nFinding optimal starting length...")
        length_mm = find_initial_length(
            args.width, args.thickness, material, target_f1
        )
        print(f"Estimated length: {length_mm:.1f} mm")
    
    # Create base geometry
    base_bar = create_bar(length_mm, args.width, args.thickness)
    
    # Set up undercut configuration
    undercut_config = UndercutConfig.multi_point(
        n_cuts=args.cuts,
        width_mm=args.cut_width
    )
    
    # Set up target
    target = TuningTarget(
        f1_target=target_f1,
        ratios=ratios
    )
    
    print(f"\nOptimizing {args.cuts} undercuts (width={args.cut_width}mm)...")
    print(f"Undercut positions: {[f'{p:.1%}' for p in undercut_config.positions]}")
    print("")
    
    # Run optimization
    result = optimize_bar(
        base_bar,
        material,
        target,
        undercut_config,
        n_elements=args.elements,
        verbose=True
    )
    
    print("\n" + result.summary())

    # Spread cuts if requested
    if args.spread:
        spread_result = spread_cuts(
            result, base_bar, material, target, undercut_config,
            n_per_side=args.spread,
            spacing_mm=args.spread_spacing if args.spread_spacing else None,
            n_elements=args.elements,
            verbose=True
        )
        print("\n" + spread_result.summary())
        result = spread_result  # Use spread result for plotting

    # Show the thickness profile
    if args.plot:
        try:
            plot_result(result, base_bar)
        except ImportError:
            print("(matplotlib not available for plotting)")


def plot_result(result, base_bar):
    """Plot the optimized bar profile and mode shapes."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot thickness profile
    ax1 = axes[0]
    x_base, h_base = base_bar.get_profile(200)
    x_opt, h_opt = result.optimized_geometry.get_profile(200)
    
    ax1.fill_between(x_base * 1000, 0, h_base * 1000, alpha=0.3, label='Original')
    ax1.fill_between(x_opt * 1000, 0, h_opt * 1000, alpha=0.5, label='Optimized')
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Thickness (mm)')
    ax1.set_title('Bar Cross-Section Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Note undercut positions
    for i, uc in enumerate(result.optimized_geometry.undercuts):
        x_pos = uc.center * result.optimized_geometry.length * 1000
        ax1.axvline(x_pos, color='red', linestyle='--', alpha=0.5)
        ax1.annotate(f'Cut {i+1}\n{result.undercut_depths_mm[i]:.1f}mm',
                    (x_pos, h_base.max() * 1000 * 0.9),
                    ha='center', fontsize=8)
    
    ax1.set_ylim(0, base_bar.thickness * 1000 * 1.1)
    
    # Info text
    info = (f"Material: {result.optimized_geometry.width*1000:.0f}×{base_bar.thickness*1000:.0f}mm\n"
            f"Length: {result.optimized_geometry.length*1000:.1f}mm\n"
            f"f1: {result.achieved_frequencies[0]:.1f} Hz ({result.frequency_errors_cents[0]:+.0f}¢)\n"
            f"Ratios: {':'.join(f'{r:.2f}' for r in result.achieved_frequencies/result.achieved_frequencies[0])}")
    ax1.text(0.02, 0.98, info, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2 = axes[1]
    ax2.text(0.5, 0.5, 'Mode shapes - run with modal_analysis() for detailed shapes',
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_axis_off()
    
    plt.tight_layout()
    plt.savefig('optimization_result.png', dpi=150)
    print("\nPlot saved to optimization_result.png")
    plt.show()


def materials_command(args):
    """List available materials."""
    print("\nAVAILABLE MATERIALS:")
    print("=" * 60)
    
    from .materials import MATERIALS
    
    # Get unique materials
    seen = set()
    for name, mat in sorted(MATERIALS.items()):
        if mat.name not in seen:
            seen.add(mat.name)
            print(f"\n{mat.name}:")
            print(f"  E = {mat.E/1e9:.2f} GPa, G = {mat.G/1e9:.2f} GPa, ρ = {mat.rho:.0f} kg/m³")
            if mat.description:
                print(f"  {mat.description}")


def estimate_length_command(args):
    """Estimate bar length for a target note."""
    material = get_material(args.material)
    
    if args.note:
        target_f1 = note_to_frequency(args.note)
    else:
        target_f1 = args.frequency
    
    length_mm = find_initial_length(
        args.width, args.thickness, material, target_f1
    )
    
    # Verify
    bar = create_bar(length_mm, args.width, args.thickness)
    freqs = compute_frequencies(bar, material, n_elements=100, n_modes=3)
    
    print(f"\nFor {args.material}, {args.width}×{args.thickness}mm cross-section:")
    print(f"  Target: {target_f1:.2f} Hz", end="")
    if args.note:
        print(f" ({args.note})")
    else:
        print("")
    print(f"  Recommended length: {length_mm:.1f} mm")
    print(f"  Actual f1 (uncut): {freqs[0]:.2f} Hz")
    print(f"  Initial ratios: 1:{freqs[1]/freqs[0]:.2f}:{freqs[2]/freqs[0]:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Idiophone Bar Tuner - FEM-based bar optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze an existing bar
  python -m idiophone_tuner analyze --length 350 --width 32 --thickness 24 --material sapele

  # Find length for a target note  
  python -m idiophone_tuner estimate --note F4 --width 32 --thickness 24 --material sapele

  # Optimize undercuts for xylophone tuning
  python -m idiophone_tuner optimize --note F4 --width 32 --thickness 24 --material sapele --tuning xylophone

  # Custom ratios
  python -m idiophone_tuner optimize --frequency 349.2 --width 32 --thickness 24 --material sapele --ratios 1:3:6

  # List materials
  python -m idiophone_tuner materials
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a bar')
    analyze_parser.add_argument('--length', type=float, required=True, help='Bar length (mm)')
    analyze_parser.add_argument('--width', type=float, required=True, help='Bar width (mm)')
    analyze_parser.add_argument('--thickness', type=float, required=True, help='Bar thickness (mm)')
    analyze_parser.add_argument('--material', type=str, required=True, help='Material name')
    analyze_parser.add_argument('--modes', type=int, default=5, help='Number of modes to compute')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize undercuts')
    opt_parser.add_argument('--length', type=float, help='Bar length (mm) - auto-calculated if not provided')
    opt_parser.add_argument('--width', type=float, required=True, help='Bar width (mm)')
    opt_parser.add_argument('--thickness', type=float, required=True, help='Bar thickness (mm)')
    opt_parser.add_argument('--material', type=str, required=True, help='Material name')
    opt_parser.add_argument('--note', type=str, help='Target note (e.g., F4, C#5)')
    opt_parser.add_argument('--frequency', type=float, help='Target fundamental frequency (Hz)')
    opt_parser.add_argument('--ratios', type=str, help='Target ratios (e.g., "1:3:6")')
    opt_parser.add_argument('--tuning', type=str, choices=list(TUNING_RATIOS.keys()),
                           help='Standard tuning preset')
    opt_parser.add_argument('--cuts', type=int, default=3, help='Number of undercuts')
    opt_parser.add_argument('--cut-width', type=float, default=3.0, help='Undercut width (mm)')
    opt_parser.add_argument('--elements', type=int, default=100, help='FEM elements')
    opt_parser.add_argument('--spread', type=int, default=0, metavar='N',
                           help='Spread each cut into multiple shallow cuts '
                                '(N extra cuts per side, e.g. 2 = 5 cuts per group)')
    opt_parser.add_argument('--spread-spacing', type=float, default=None, metavar='MM',
                           help='Spacing between spread sub-cuts (mm, default=cut width)')
    opt_parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    # Materials command
    mat_parser = subparsers.add_parser('materials', help='List available materials')
    
    # Estimate length command
    est_parser = subparsers.add_parser('estimate', help='Estimate length for target frequency')
    est_parser.add_argument('--width', type=float, required=True, help='Bar width (mm)')
    est_parser.add_argument('--thickness', type=float, required=True, help='Bar thickness (mm)')
    est_parser.add_argument('--material', type=str, required=True, help='Material name')
    est_parser.add_argument('--note', type=str, help='Target note (e.g., F4)')
    est_parser.add_argument('--frequency', type=float, help='Target frequency (Hz)')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'optimize':
        if not args.note and not args.frequency:
            parser.error("Must specify --note or --frequency")
        optimize_command(args)
    elif args.command == 'materials':
        materials_command(args)
    elif args.command == 'estimate':
        if not args.note and not args.frequency:
            parser.error("Must specify --note or --frequency")
        estimate_length_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
