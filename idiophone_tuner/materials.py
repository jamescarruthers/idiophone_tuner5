"""
Material properties database for idiophone bar optimization.

Properties required for Timoshenko beam analysis:
- E: Young's modulus (Pa)
- G: Shear modulus (Pa) 
- rho: Density (kg/m³)

For anisotropic materials like wood, we use the longitudinal properties
(along the grain) which dominate bending behavior.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Material:
    """Material properties for beam analysis."""
    name: str
    E: float           # Young's modulus (Pa)
    G: float           # Shear modulus (Pa)
    rho: float         # Density (kg/m³)
    description: str = ""
    
    @property
    def nu(self) -> float:
        """Poisson's ratio (derived from E and G)."""
        return self.E / (2 * self.G) - 1


# =============================================================================
# Wood materials (longitudinal properties along grain)
# =============================================================================

SAPELE = Material(
    name="sapele",
    E=11.7e9,          # ~11.7 GPa longitudinal
    G=0.73e9,          # Shear modulus
    rho=640,           # kg/m³
    description="Sapele (Entandrophragma cylindricum) - African hardwood, good tone"
)

PADAUK = Material(
    name="padauk",
    E=11.0e9,
    G=0.69e9,
    rho=750,
    description="African Padauk - dense, bright tone"
)

ROSEWOOD_HONDURAN = Material(
    name="honduran_rosewood",
    E=14.0e9,
    G=0.88e9,
    rho=900,
    description="Honduran Rosewood (Dalbergia stevensonii) - classic marimba"
)

ROSEWOOD_INDIAN = Material(
    name="indian_rosewood",
    E=11.5e9,
    G=0.72e9,
    rho=830,
    description="Indian Rosewood (Dalbergia latifolia)"
)

MAPLE_HARD = Material(
    name="hard_maple",
    E=12.6e9,
    G=0.79e9,
    rho=705,
    description="Hard Maple (Acer saccharum) - bright, clear tone"
)

ASH = Material(
    name="ash",
    E=12.0e9,
    G=0.75e9,
    rho=670,
    description="White Ash (Fraxinus americana)"
)

BUBINGA = Material(
    name="bubinga",
    E=12.8e9,
    G=0.80e9,
    rho=890,
    description="Bubinga (Guibourtia) - dense African hardwood"
)

PURPLEHEART = Material(
    name="purpleheart",
    E=14.5e9,
    G=0.91e9,
    rho=880,
    description="Purpleheart (Peltogyne) - very stiff, bright"
)

# =============================================================================
# Metal materials
# =============================================================================

ALUMINUM_6061 = Material(
    name="aluminum_6061",
    E=68.9e9,
    G=26.0e9,
    rho=2700,
    description="Aluminum 6061-T6 - glockenspiel bars"
)

ALUMINUM_7075 = Material(
    name="aluminum_7075",
    E=71.7e9,
    G=26.9e9,
    rho=2810,
    description="Aluminum 7075-T6 - stiffer, higher overtones"
)

BRONZE_PHOSPHOR = Material(
    name="phosphor_bronze",
    E=110e9,
    G=41e9,
    rho=8800,
    description="Phosphor Bronze - bell-like tone"
)

STEEL_1095 = Material(
    name="steel_1095",
    E=205e9,
    G=80e9,
    rho=7850,
    description="1095 Carbon Steel - bright, long sustain"
)

BRASS = Material(
    name="brass",
    E=100e9,
    G=37e9,
    rho=8500,
    description="Yellow Brass (C26000) - warm tone"
)


# =============================================================================
# Material registry
# =============================================================================

MATERIALS = {
    # Woods
    "sapele": SAPELE,
    "padauk": PADAUK,
    "honduran_rosewood": ROSEWOOD_HONDURAN,
    "indian_rosewood": ROSEWOOD_INDIAN,
    "rosewood": ROSEWOOD_HONDURAN,  # Default rosewood
    "maple": MAPLE_HARD,
    "hard_maple": MAPLE_HARD,
    "ash": ASH,
    "bubinga": BUBINGA,
    "purpleheart": PURPLEHEART,
    
    # Metals
    "aluminum": ALUMINUM_6061,
    "aluminum_6061": ALUMINUM_6061,
    "aluminum_7075": ALUMINUM_7075,
    "bronze": BRONZE_PHOSPHOR,
    "phosphor_bronze": BRONZE_PHOSPHOR,
    "steel": STEEL_1095,
    "brass": BRASS,
}


def get_material(name: str) -> Material:
    """Get material by name (case-insensitive)."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in MATERIALS:
        available = ", ".join(sorted(set(MATERIALS.keys())))
        raise ValueError(f"Unknown material '{name}'. Available: {available}")
    return MATERIALS[key]


def list_materials() -> list[str]:
    """List all available material names."""
    return sorted(set(MATERIALS.keys()))


def custom_material(
    name: str,
    E: float,
    G: Optional[float] = None,
    rho: float = 1000,
    nu: Optional[float] = None
) -> Material:
    """
    Create a custom material.
    
    Args:
        name: Material name
        E: Young's modulus (Pa)
        G: Shear modulus (Pa) - if not provided, calculated from nu
        rho: Density (kg/m³)
        nu: Poisson's ratio - used to calculate G if G not provided
    
    Returns:
        Material instance
    """
    if G is None:
        if nu is None:
            nu = 0.3  # Default Poisson's ratio
        G = E / (2 * (1 + nu))
    
    return Material(name=name, E=E, G=G, rho=rho, description="Custom material")
