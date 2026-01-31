"""
Timoshenko beam finite element solver for modal analysis.

Implements 2-node beam elements with 4 DOF per node (w, theta at each node)
including shear deformation and rotary inertia effects.

Key references:
- Friedman & Kosmatka (1993) - Exact Timoshenko stiffness matrix
- Dawe (1978) - Numerical studies of Timoshenko beams
"""

import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Optional

from .materials import Material
from .geometry import BarGeometry


@dataclass
class ModeResult:
    """Results for a single vibration mode."""
    frequency: float           # Natural frequency in Hz
    omega: float               # Angular frequency in rad/s
    mode_shape: np.ndarray     # Mode shape (displacement at each node)
    rotations: np.ndarray      # Rotations at each node


@dataclass
class ModalAnalysisResult:
    """Complete modal analysis results."""
    frequencies: np.ndarray    # Natural frequencies in Hz (ascending)
    omegas: np.ndarray         # Angular frequencies in rad/s
    mode_shapes: np.ndarray    # Mode shapes (n_dof x n_modes)
    nodes: np.ndarray          # Node positions along beam (meters)
    n_elements: int
    
    def get_mode(self, mode_number: int) -> ModeResult:
        """
        Get results for a specific mode (1-indexed for physical modes).
        
        Mode 1 is the first bending mode (fundamental), etc.
        Note: Rigid body modes (if any) are already filtered out.
        """
        idx = mode_number - 1
        if idx < 0 or idx >= len(self.frequencies):
            raise ValueError(f"Mode {mode_number} not available. Have {len(self.frequencies)} modes.")
        
        # Extract displacement and rotation DOFs
        full_shape = self.mode_shapes[:, idx]
        n_nodes = len(self.nodes)
        
        # DOFs are ordered: w0, theta0, w1, theta1, ...
        displacements = full_shape[0::2]
        rotations = full_shape[1::2]
        
        return ModeResult(
            frequency=self.frequencies[idx],
            omega=self.omegas[idx],
            mode_shape=displacements,
            rotations=rotations
        )
    
    def get_antinode_positions(self, mode_number: int) -> np.ndarray:
        """
        Get antinode positions (local maxima of |displacement|) for a mode.
        
        Returns positions as fractions of bar length (0-1).
        """
        mode = self.get_mode(mode_number)
        shape = np.abs(mode.mode_shape)
        
        # Find local maxima
        antinodes = []
        for i in range(1, len(shape) - 1):
            if shape[i] > shape[i-1] and shape[i] > shape[i+1]:
                antinodes.append(self.nodes[i])
        
        # Also check endpoints
        if shape[0] > shape[1]:
            antinodes.insert(0, self.nodes[0])
        if shape[-1] > shape[-2]:
            antinodes.append(self.nodes[-1])
        
        L = self.nodes[-1]
        return np.array(antinodes) / L
    
    def get_node_positions(self, mode_number: int, threshold: float = 0.1) -> np.ndarray:
        """
        Get node positions (zero crossings / local minima) for a mode.
        
        Returns positions as fractions of bar length (0-1).
        """
        mode = self.get_mode(mode_number)
        shape = mode.mode_shape
        L = self.nodes[-1]
        
        nodes = []
        
        # Find zero crossings
        for i in range(len(shape) - 1):
            if shape[i] * shape[i+1] < 0:
                # Linear interpolation for crossing point
                t = -shape[i] / (shape[i+1] - shape[i])
                x = self.nodes[i] + t * (self.nodes[i+1] - self.nodes[i])
                nodes.append(x / L)
        
        return np.array(nodes)


def timoshenko_element_matrices(
    L: float,
    E: float,
    G: float,
    rho: float,
    A: float,
    I: float,
    kappa: float = 5/6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Timoshenko beam element stiffness and mass matrices.
    
    Uses the exact Timoshenko stiffness matrix (Friedman & Kosmatka, 1993)
    and consistent mass matrix with rotary inertia.
    
    Args:
        L: Element length
        E: Young's modulus
        G: Shear modulus
        rho: Density
        A: Cross-sectional area
        I: Second moment of area
        kappa: Shear correction factor (5/6 for rectangular)
    
    Returns:
        K: 4x4 stiffness matrix
        M: 4x4 consistent mass matrix
    
    DOF ordering: [w1, theta1, w2, theta2]
    where w is transverse displacement and theta is rotation.
    """
    # Shear parameter
    phi = 12 * E * I / (kappa * G * A * L**2)
    
    # Stiffness matrix (exact Timoshenko formulation)
    coeff = E * I / (L**3 * (1 + phi))
    
    K = coeff * np.array([
        [12,           6*L,          -12,          6*L],
        [6*L,          (4+phi)*L**2, -6*L,         (2-phi)*L**2],
        [-12,          -6*L,         12,           -6*L],
        [6*L,          (2-phi)*L**2, -6*L,         (4+phi)*L**2]
    ])
    
    # Consistent mass matrix with rotary inertia
    # Standard terms + rotary inertia corrections
    m = rho * A * L
    r2 = I / A  # Radius of gyration squared
    
    # Translational mass matrix
    M_trans = m / 420 / (1 + phi)**2 * np.array([
        [156 + 294*phi + 140*phi**2,
         (22 + 38.5*phi + 17.5*phi**2)*L,
         54 + 126*phi + 70*phi**2,
         -(13 + 31.5*phi + 17.5*phi**2)*L],
        [(22 + 38.5*phi + 17.5*phi**2)*L,
         (4 + 7*phi + 3.5*phi**2)*L**2,
         (13 + 31.5*phi + 17.5*phi**2)*L,
         -(3 + 7*phi + 3.5*phi**2)*L**2],
        [54 + 126*phi + 70*phi**2,
         (13 + 31.5*phi + 17.5*phi**2)*L,
         156 + 294*phi + 140*phi**2,
         -(22 + 38.5*phi + 17.5*phi**2)*L],
        [-(13 + 31.5*phi + 17.5*phi**2)*L,
         -(3 + 7*phi + 3.5*phi**2)*L**2,
         -(22 + 38.5*phi + 17.5*phi**2)*L,
         (4 + 7*phi + 3.5*phi**2)*L**2]
    ])
    
    # Rotary inertia contribution
    M_rot = rho * I / 30 / L / (1 + phi)**2 * np.array([
        [36,           (3 - 15*phi)*L,        -36,          (3 - 15*phi)*L],
        [(3 - 15*phi)*L, (4 + 5*phi + 10*phi**2)*L**2, -(3 - 15*phi)*L, (-1 - 5*phi + 5*phi**2)*L**2],
        [-36,          -(3 - 15*phi)*L,       36,           -(3 - 15*phi)*L],
        [(3 - 15*phi)*L, (-1 - 5*phi + 5*phi**2)*L**2, -(3 - 15*phi)*L, (4 + 5*phi + 10*phi**2)*L**2]
    ])
    
    M = M_trans + M_rot
    
    return K, M


def assemble_global_matrices(
    geometry: BarGeometry,
    material: Material,
    n_elements: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble global stiffness and mass matrices for a beam.
    
    Args:
        geometry: Bar geometry (may have variable cross-section)
        material: Material properties
        n_elements: Number of elements
    
    Returns:
        K: Global stiffness matrix
        M: Global mass matrix
        nodes: Node positions (meters)
    """
    n_nodes = n_elements + 1
    n_dof = 2 * n_nodes  # 2 DOF per node (w, theta)
    
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    
    # Node positions
    nodes = np.linspace(0, geometry.length, n_nodes)
    
    kappa = 5 / 6  # Rectangular section
    
    for i in range(n_elements):
        # Element length
        L_e = nodes[i + 1] - nodes[i]
        
        # Cross-section properties at element midpoint
        x_mid = (nodes[i] + nodes[i + 1]) / 2 / geometry.length
        A = geometry.area_at(x_mid)
        I = geometry.I_at(x_mid)
        
        # Element matrices
        K_e, M_e = timoshenko_element_matrices(
            L=L_e,
            E=material.E,
            G=material.G,
            rho=material.rho,
            A=A,
            I=I,
            kappa=kappa
        )
        
        # Assembly: DOFs for element i are [2i, 2i+1, 2i+2, 2i+3]
        dofs = [2*i, 2*i + 1, 2*i + 2, 2*i + 3]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += K_e[a, b]
                M[dofs[a], dofs[b]] += M_e[a, b]
    
    return K, M, nodes


def modal_analysis(
    geometry: BarGeometry,
    material: Material,
    n_elements: int = 100,
    n_modes: int = 10,
    boundary_conditions: str = 'free-free',
    use_sparse: bool = True
) -> ModalAnalysisResult:
    """
    Perform modal analysis on a beam.
    
    Args:
        geometry: Bar geometry
        material: Material properties
        n_elements: Number of finite elements
        n_modes: Number of modes to compute
        boundary_conditions: 'free-free' (only option for idiophones)
        use_sparse: Use sparse solver (faster for large n_elements)
    
    Returns:
        ModalAnalysisResult with frequencies and mode shapes
    """
    if boundary_conditions != 'free-free':
        raise NotImplementedError("Only free-free boundary conditions implemented")
    
    # Assemble global matrices
    K, M, nodes = assemble_global_matrices(geometry, material, n_elements)
    
    # Solve generalized eigenvalue problem: K @ phi = omega^2 @ M @ phi
    # For free-free, we have 2 rigid body modes (translation, rotation)
    # which appear as near-zero eigenvalues
    
    if use_sparse and n_elements > 50:
        # Use sparse solver - much faster for large systems
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        K_sparse = csr_matrix(K)
        M_sparse = csr_matrix(M)
        
        # Request extra modes to account for rigid body modes we'll filter
        n_request = min(n_modes + 4, K.shape[0] - 1)
        
        # Use shift-invert mode for better convergence on small eigenvalues
        # sigma=1.0 shifts away from zero (rigid body modes)
        try:
            eigenvalues, eigenvectors = eigsh(
                K_sparse, k=n_request, M=M_sparse, 
                sigma=1.0, which='LM',
                tol=1e-6
            )
        except Exception:
            # Fallback to dense if sparse fails
            eigenvalues, eigenvectors = eigh(K, M)
    else:
        # Dense solver for small systems
        eigenvalues, eigenvectors = eigh(K, M)
    
    # Filter out rigid body modes (eigenvalues close to zero)
    threshold = 1.0  # rad/s threshold for rigid body modes
    mask = np.sqrt(np.maximum(eigenvalues, 0)) > threshold
    
    omegas = np.sqrt(np.maximum(eigenvalues[mask], 0))
    modes = eigenvectors[:, mask]
    
    # Sort by frequency (should already be sorted, but ensure)
    sort_idx = np.argsort(omegas)
    omegas = omegas[sort_idx]
    modes = modes[:, sort_idx]
    
    # Truncate to requested number of modes
    n_available = min(n_modes, len(omegas))
    omegas = omegas[:n_available]
    modes = modes[:, :n_available]
    
    # Convert to frequencies
    frequencies = omegas / (2 * np.pi)
    
    # Normalize mode shapes (max displacement = 1)
    for i in range(modes.shape[1]):
        displacements = modes[0::2, i]
        max_disp = np.max(np.abs(displacements))
        if max_disp > 0:
            modes[:, i] /= max_disp
    
    return ModalAnalysisResult(
        frequencies=frequencies,
        omegas=omegas,
        mode_shapes=modes,
        nodes=nodes,
        n_elements=n_elements
    )


def compute_frequencies(
    geometry: BarGeometry,
    material: Material,
    n_elements: int = 100,
    n_modes: int = 3
) -> np.ndarray:
    """
    Convenience function to just get the first n mode frequencies.

    Returns:
        Array of frequencies in Hz [f1, f2, f3, ...]
    """
    result = modal_analysis(geometry, material, n_elements, n_modes)
    return result.frequencies[:n_modes]


def fast_compute_frequencies(
    L_e: np.ndarray,
    E: float,
    G: float,
    rho: float,
    A: np.ndarray,
    I_val: np.ndarray,
    n_modes: int = 3
) -> np.ndarray:
    """
    Compute natural frequencies using pre-computed element properties.

    Fully vectorized — no Python loops for matrix construction or assembly.
    Designed for use in optimization loops where geometry changes each call
    but the mesh topology is fixed.

    Args:
        L_e: Element lengths (n_elements,)
        E, G, rho: Material scalars
        A: Cross-section areas per element (n_elements,)
        I_val: Second moments of area per element (n_elements,)
        n_modes: Number of modes to return

    Returns:
        Array of natural frequencies in Hz (n_modes,)
    """
    n_elements = len(L_e)
    n_dof = 2 * (n_elements + 1)

    kappa = 5.0 / 6.0
    phi = 12.0 * E * I_val / (kappa * G * A * L_e ** 2)
    phi2 = phi * phi
    denom = 1.0 + phi
    L2 = L_e * L_e

    # === Stiffness matrices (n_elements, 4, 4) ===
    ck = E * I_val / (L_e ** 3 * denom)

    v12 = 12.0 * ck
    v6L = 6.0 * L_e * ck
    v4pL2 = (4.0 + phi) * L2 * ck
    v2pL2 = (2.0 - phi) * L2 * ck

    K_all = np.empty((n_elements, 4, 4))
    K_all[:, 0, 0] = v12
    K_all[:, 0, 1] = v6L
    K_all[:, 0, 2] = -v12
    K_all[:, 0, 3] = v6L
    K_all[:, 1, 0] = v6L
    K_all[:, 1, 1] = v4pL2
    K_all[:, 1, 2] = -v6L
    K_all[:, 1, 3] = v2pL2
    K_all[:, 2, 0] = -v12
    K_all[:, 2, 1] = -v6L
    K_all[:, 2, 2] = v12
    K_all[:, 2, 3] = -v6L
    K_all[:, 3, 0] = v6L
    K_all[:, 3, 1] = v2pL2
    K_all[:, 3, 2] = -v6L
    K_all[:, 3, 3] = v4pL2

    # === Translational mass (n_elements, 4, 4) ===
    mt = rho * A * L_e / 420.0 / (denom * denom)

    a00 = 156.0 + 294.0 * phi + 140.0 * phi2
    a01 = (22.0 + 38.5 * phi + 17.5 * phi2) * L_e
    a02 = 54.0 + 126.0 * phi + 70.0 * phi2
    a03 = (13.0 + 31.5 * phi + 17.5 * phi2) * L_e
    a11 = (4.0 + 7.0 * phi + 3.5 * phi2) * L2
    a13 = (3.0 + 7.0 * phi + 3.5 * phi2) * L2

    M_t = np.empty((n_elements, 4, 4))
    M_t[:, 0, 0] = mt * a00
    M_t[:, 0, 1] = mt * a01
    M_t[:, 0, 2] = mt * a02
    M_t[:, 0, 3] = -mt * a03
    M_t[:, 1, 0] = M_t[:, 0, 1]
    M_t[:, 1, 1] = mt * a11
    M_t[:, 1, 2] = mt * a03
    M_t[:, 1, 3] = -mt * a13
    M_t[:, 2, 0] = M_t[:, 0, 2]
    M_t[:, 2, 1] = M_t[:, 1, 2]
    M_t[:, 2, 2] = M_t[:, 0, 0]
    M_t[:, 2, 3] = -M_t[:, 0, 1]
    M_t[:, 3, 0] = M_t[:, 0, 3]
    M_t[:, 3, 1] = M_t[:, 1, 3]
    M_t[:, 3, 2] = M_t[:, 2, 3]
    M_t[:, 3, 3] = M_t[:, 1, 1]

    # === Rotary inertia mass (n_elements, 4, 4) ===
    mr = rho * I_val / (30.0 * L_e * denom * denom)

    pL = (3.0 - 15.0 * phi) * L_e
    r11 = (4.0 + 5.0 * phi + 10.0 * phi2) * L2
    r13 = (-1.0 - 5.0 * phi + 5.0 * phi2) * L2

    M_r = np.empty((n_elements, 4, 4))
    M_r[:, 0, 0] = mr * 36.0
    M_r[:, 0, 1] = mr * pL
    M_r[:, 0, 2] = -mr * 36.0
    M_r[:, 0, 3] = mr * pL
    M_r[:, 1, 0] = M_r[:, 0, 1]
    M_r[:, 1, 1] = mr * r11
    M_r[:, 1, 2] = -mr * pL
    M_r[:, 1, 3] = mr * r13
    M_r[:, 2, 0] = M_r[:, 0, 2]
    M_r[:, 2, 1] = M_r[:, 1, 2]
    M_r[:, 2, 2] = M_r[:, 0, 0]
    M_r[:, 2, 3] = -mr * pL
    M_r[:, 3, 0] = M_r[:, 0, 3]
    M_r[:, 3, 1] = M_r[:, 1, 3]
    M_r[:, 3, 2] = M_r[:, 2, 3]
    M_r[:, 3, 3] = M_r[:, 1, 1]

    M_all = M_t + M_r

    # === Assembly via vectorized indexing ===
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))

    idx = np.arange(n_elements)
    dofs = np.column_stack([2 * idx, 2 * idx + 1, 2 * idx + 2, 2 * idx + 3])
    rows = np.repeat(dofs, 4, axis=1)         # (n_elements, 16)
    cols = np.tile(dofs, (1, 4))               # (n_elements, 16)

    np.add.at(K, (rows.ravel(), cols.ravel()), K_all.reshape(-1))
    np.add.at(M, (rows.ravel(), cols.ravel()), M_all.reshape(-1))

    # === Eigenvalue solve ===
    if n_elements > 50:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        n_request = min(n_modes + 4, n_dof - 1)
        try:
            eigenvalues, _ = eigsh(
                csr_matrix(K), k=n_request, M=csr_matrix(M),
                sigma=1.0, which='LM', tol=1e-6
            )
        except Exception:
            eigenvalues, _ = eigh(K, M)
    else:
        eigenvalues, _ = eigh(K, M)

    # Filter rigid body modes and return frequencies
    omegas = np.sqrt(np.maximum(eigenvalues, 0))
    omegas = omegas[omegas > 1.0]         # threshold for rigid body modes
    omegas = np.sort(omegas)[:n_modes]

    return omegas / (2.0 * np.pi)


# =============================================================================
# Analytical solutions for validation
# =============================================================================

def euler_bernoulli_free_free_frequencies(
    L: float,
    E: float,
    I: float,
    rho: float,
    A: float,
    n_modes: int = 3
) -> np.ndarray:
    """
    Analytical frequencies for uniform Euler-Bernoulli beam (free-free).
    
    Useful for validation. Note: Timoshenko frequencies will be lower
    due to shear deformation, especially for higher modes and thick beams.
    """
    # Beta*L values for free-free beam (from characteristic equation)
    beta_L = [4.7300, 7.8532, 10.9956, 14.1372, 17.2788, 20.4204, 23.5619, 26.7035]
    
    EI = E * I
    m = rho * A  # mass per length
    
    freqs = []
    for i in range(min(n_modes, len(beta_L))):
        beta = beta_L[i] / L
        omega = beta**2 * np.sqrt(EI / m)
        freqs.append(omega / (2 * np.pi))
    
    return np.array(freqs)
