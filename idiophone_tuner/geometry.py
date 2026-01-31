"""
Geometry definitions for idiophone bars.

Handles rectangular cross-sections with variable thickness along length
to model undercuts for tuning.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import math


def profile_weight(u, profile):
    """Compute normalised (0-1) profile weight(s) for normalised position *u*.

    *u* can be a scalar or numpy array.  Values with ``|u| > 1`` (outside
    the undercut) get weight 0.  This is the single source of truth for
    all undercut profile shapes.
    """
    u = np.asarray(u, dtype=float)
    scalar = u.ndim == 0
    u = np.atleast_1d(u)

    weights = np.zeros_like(u)
    mask = np.abs(u) <= 1.0
    if np.any(mask):
        if profile == 'flat':
            weights[mask] = 1.0
        elif profile == 'parabolic':
            weights[mask] = 1.0 - u[mask] ** 2
        elif profile == 'cosine':
            weights[mask] = 0.5 * (1.0 + np.cos(np.pi * u[mask]))
        elif profile == 'elliptical':
            weights[mask] = np.sqrt(1.0 - u[mask] ** 2)
        else:
            raise ValueError(f"Unknown profile type: {profile}")

    return float(weights[0]) if scalar else weights


@dataclass
class Undercut:
    """
    Defines a single undercut (notch or arch) in the bar.
    
    The undercut is centered at position `center` (as fraction of length, 0-1),
    has a given `width` (in meters), and `depth` (in meters, measured from bottom).
    
    Profile types:
    - 'flat': constant depth across width (for notch cuts)
    - 'parabolic': smooth parabolic profile, zero at edges (traditional arch)
    - 'cosine': cosine profile, smooth transition (good for CNC)
    - 'elliptical': elliptical arch profile
    """
    center: float          # Position as fraction of length (0-1)
    width: float           # Width in meters
    depth: float           # Maximum depth in meters
    profile: str = 'flat'  # 'flat', 'parabolic', 'cosine', 'elliptical'
    
    def depth_at(self, x_rel: float, bar_length: float) -> float:
        """
        Get undercut depth at position x_rel (fraction of bar length).

        Returns 0 if outside the undercut region.
        """
        center_m = self.center * bar_length
        half_width = self.width / 2
        x_m = x_rel * bar_length
        u = (x_m - center_m) / half_width
        return self.depth * profile_weight(u, self.profile)


@dataclass
class BarGeometry:
    """
    Defines the geometry of an idiophone bar.
    
    The bar has constant width and a thickness that varies along the length
    due to undercuts. The coordinate system has x along the length of the bar.
    """
    length: float          # Length in meters
    width: float           # Width in meters (constant)
    thickness: float       # Base thickness in meters
    undercuts: list[Undercut] = field(default_factory=list)
    
    def thickness_at(self, x_rel: float) -> float:
        """
        Get thickness at position x_rel (fraction of length, 0-1).
        
        Accounts for all undercuts.
        """
        total_depth = sum(u.depth_at(x_rel, self.length) for u in self.undercuts)
        return max(self.thickness - total_depth, 1e-6)  # Prevent zero thickness
    
    def area_at(self, x_rel: float) -> float:
        """Cross-sectional area at position x_rel."""
        return self.width * self.thickness_at(x_rel)
    
    def I_at(self, x_rel: float) -> float:
        """Second moment of area (I_zz) at position x_rel."""
        h = self.thickness_at(x_rel)
        b = self.width
        return b * h**3 / 12
    
    def shear_area_at(self, x_rel: float) -> float:
        """
        Effective shear area at position x_rel.
        
        For rectangular sections: A_shear = kappa * A
        where kappa = 5/6 is the shear correction factor.
        """
        kappa = 5 / 6  # Rectangular section
        return kappa * self.area_at(x_rel)
    
    def get_profile(self, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the thickness profile along the bar.
        
        Returns:
            x: Position array (meters)
            h: Thickness array (meters)
        """
        x_rel = np.linspace(0, 1, n_points)
        x = x_rel * self.length
        h = np.array([self.thickness_at(xr) for xr in x_rel])
        return x, h
    
    def copy_with_undercuts(self, undercuts: list[Undercut]) -> 'BarGeometry':
        """Create a copy of this geometry with different undercuts."""
        return BarGeometry(
            length=self.length,
            width=self.width,
            thickness=self.thickness,
            undercuts=undercuts
        )


def create_bar(
    length_mm: float,
    width_mm: float,
    thickness_mm: float
) -> BarGeometry:
    """
    Create a bar geometry with dimensions in millimeters.
    
    Args:
        length_mm: Bar length in mm
        width_mm: Bar width in mm
        thickness_mm: Bar thickness in mm
    
    Returns:
        BarGeometry instance (internally uses meters)
    """
    return BarGeometry(
        length=length_mm / 1000,
        width=width_mm / 1000,
        thickness=thickness_mm / 1000
    )


def add_undercut(
    bar: BarGeometry,
    center_fraction: float,
    width_mm: float,
    depth_mm: float,
    profile: str = 'flat'
) -> BarGeometry:
    """
    Add an undercut to a bar geometry.
    
    Args:
        bar: Existing bar geometry
        center_fraction: Position of undercut center (0-1 along length)
        width_mm: Undercut width in mm
        depth_mm: Undercut depth in mm
        profile: 'flat' or 'parabolic'
    
    Returns:
        New BarGeometry with the undercut added
    """
    new_undercut = Undercut(
        center=center_fraction,
        width=width_mm / 1000,
        depth=depth_mm / 1000,
        profile=profile
    )
    return BarGeometry(
        length=bar.length,
        width=bar.width,
        thickness=bar.thickness,
        undercuts=bar.undercuts + [new_undercut]
    )


# =============================================================================
# Standard tuning configurations
# =============================================================================

def xylophone_undercut_positions() -> dict[str, float]:
    """
    Standard undercut positions for xylophone tuning (1:3:6 ratio).
    
    These are approximate antinode/node positions for mode 1, 2, 3.
    Positions are fractions of bar length.
    """
    return {
        'center': 0.5,           # Mode 1 antinode, mode 2 antinode
        'quarter_left': 0.224,   # Near mode 1 node
        'quarter_right': 0.776,  # Near mode 1 node
        'eighth_left': 0.132,    # Mode 2 antinode region
        'eighth_right': 0.868,   # Mode 2 antinode region
    }


def marimba_undercut_positions() -> dict[str, float]:
    """
    Standard undercut positions for marimba tuning (1:4:10 ratio).
    """
    return {
        'center': 0.5,
        'inner_left': 0.27,
        'inner_right': 0.73,
        'outer_left': 0.15,
        'outer_right': 0.85,
    }


def vibraphone_undercut_positions() -> dict[str, float]:
    """
    Standard undercut positions for vibraphone tuning (1:4 ratio, mode 2 only).
    """
    return {
        'center': 0.5,
        'node_left': 0.224,
        'node_right': 0.776,
    }
