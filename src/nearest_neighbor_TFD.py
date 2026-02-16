import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import inv, sqrtm, block_diag, eigh, expm, sqrtm, schur, det
from thewalrus.symplectic import xpxp_to_xxpp

def symplectic_form(n):
    """Returns the 2n × 2n symplectic form Omega"""
    return np.block([
        [np.zeros((n, n),dtype=np.float64), np.eye(n,dtype=np.float64)],
        [-np.eye(n,dtype=np.float64), np.zeros((n, n),dtype=np.float64)]
    ])

def symplectic_direct_sum(S1,S2):
    n = S1.shape[0]
    A1 = S1[0:n//2,0:n//2]
    B1 = S1[0:n//2,n//2:n]
    C1 = S1[n//2:n,0:n//2]
    D1 = S1[n//2:n,n//2:n]

    A2 = S2[0:n//2,0:n//2]
    B2 = S2[0:n//2,n//2:n]
    C2 = S2[n//2:n,0:n//2]
    D2 = S2[n//2:n,n//2:n]
    
    A_block = block_diag(A1,A2)
    B_block = block_diag(B1,B2)
    C_block = block_diag(C1,C2)
    D_block = block_diag(D1,D2)   

    S_tot = np.block([
        [A_block,B_block],
        [C_block,D_block]
    ])
    return S_tot


def williamson_decomposition(Gamma):
    """Williamson decomposition: Gamma = S D S.T"""
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    eigvals, eigvecs = np.linalg.eig(1j * Omega @ Gamma)
    idx = np.argsort(np.abs(eigvals.real))
    eigvecs = eigvecs[:, idx]
    ν = np.sort(np.abs(eigvals.real))[::2]
    D = np.diag(np.repeat(ν, 2))
    
    Gamma_sym = ((Gamma + Gamma.T) / 2).astype(np.float64)
    Gamma_sqrt = sqrtm(Gamma_sym)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.repeat(ν, 2)))
    S = Gamma_sqrt @ D_inv_sqrt
    return S, D, ν

def williamson(G):
    G = 0.5 * (G + G.T)  # symmetrize
    n = G.shape[0] // 2
    Omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])
    eigvals, U = eigh(1j * Omega @ G)
    # Symplectic eigenvalues (positive pairs only)
    ν = np.sort(np.abs(eigvals.real))[::2]
    D = np.diag(np.repeat(ν, 2))
    # Construct S using G = S D S^T => S = G^{1/2} D^{-1/2}
    sqrtG = sqrtm(G)
    sqrtDinv = np.diag(1 / np.sqrt(np.repeat(ν, 2)))
    S = sqrtG @ sqrtDinv
    return S, D, ν

def williamson2(G):
    n = G.shape[0] // 2
    G = 0.5 * (G + G.T)  # symmetrize
    Omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])
    sqrtG = sqrtm(G)
    psi = inv(sqrtG) @ Omega @ inv(sqrtG)
    tilde_O, R = schur(psi, output='real')
    pi1 = np.eye(2*n)
    for i in range(n):
        if tilde_O[2*i,2*i+1]<0:
            pi1[2*i:2*i+2,2*i:2*i+2]=np.block([[0,1],[1,0]])
    phi = np.zeros((n,n))
    nus = []
    for i in range(n):
        phi[i,i] = np.abs(tilde_O[2*i,2*i+1])
        nus.append(1/np.abs(tilde_O[2*i,2*i+1]))
    sqrt_phi = sqrtm(phi)
    sqrt_phi_phi = np.block([
            [inv(sqrt_phi),np.zeros((n,n))],
            [np.zeros((n,n)),inv(sqrt_phi)]
            ])

    inv_phi = inv(phi)

    D = np.block([
            [inv_phi,np.zeros((n,n))],
            [np.zeros((n,n)),inv_phi]
            ])
    O = xpxp_to_xxpp(tilde_O @ pi1)

    S = sqrtG @ O @ sqrt_phi_phi


    """
    eigvals_sympl = np.linalg.eigvals(1j * Omega @ Gamma)
    nu = np.sort(np.abs(eigvals_sympl.real))[::2]
    D = np.diag(np.repeat(nu, 2))
    """
    
    return S, D, nus

def williamson3(G):
    G = 0.5 * (G + G.T)  # symmetrize
    n = G.shape[0] // 2
    Omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])

    sqrtG = sqrtm(G)
    eigvals, U = eigh(sqrtG @ Omega @ sqrtG)
    # Symplectic eigenvalues (positive pairs only)
    ν = np.sort(np.abs(eigvals.real))[::2]
    D = np.diag(np.repeat(ν, 2))
    # Construct S using G = S D S^T => S = G^{1/2} D^{-1/2}
    sqrtDinv = np.diag(1 / np.sqrt(np.repeat(ν, 2)))
    S = sqrtG @ U @ sqrtDinv
    return S, D, ν


def williamson_strawberry(V):
    tol=1e-11
    r"""Williamson decomposition of positive-definite (real) symmetric matrix.

    See :ref:`williamson`.

    Note that it is assumed that the symplectic form is

    .. math:: \Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the identity matrix and :math:`0` is the zero matrix.

    See https://math.stackexchange.com/questions/1171842/finding-the-symplectic-matrix-in-williamsons-theorem/2682630#2682630

    Args:
        V (array[float]): positive definite symmetric (real) matrix
        tol (float): the tolerance used when checking if the matrix is symmetric: :math:`|V-V^T| \leq` tol

    Returns:
        tuple[array,array]: ``(Db, S)`` where ``Db`` is a diagonal matrix
            and ``S`` is a symplectic matrix such that :math:`V = S^T Db S`
    """
    (n, m) = V.shape

    if n != m:
        raise ValueError("The input matrix is not square")

    diffn = np.linalg.norm(V - np.transpose(V))

    if diffn >= tol:
        raise ValueError("The input matrix is not symmetric")

    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])
    vals = np.linalg.eigvalsh(V)

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = []

    # In what follows I construct a permutation matrix p  so that the Schur matrix has
    # only positive elements above the diagonal
    # Also the Schur matrix uses the x_1,p_1, ..., x_n,p_n  ordering thus I use rotmat to
    # go to the ordering x_1, ..., x_n, p_1, ... , p_n

    for i in range(n):
        if s1[2 * i, 2 * i + 1] > 0:
            seq.append(I)
        else:
            seq.append(X)

    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    dd = xpxp_to_xxpp(s1t)
    perm_indices = xpxp_to_xxpp(np.arange(2 * n))
    Ktt = Kt[:, perm_indices]
    Db = np.diag([1 / dd[i, i + n] for i in range(n)] + [1 / dd[i, i + n] for i in range(n)])
    S = Mm12 @ Ktt @ sqrtm(Db)


    eigvals, U = eigh(sqrtm(V) @ omega @ sqrtm(V))
    v = np.sort(np.abs(eigvals.real))[::2]
    S = np.linalg.inv(S).T
    return  S, Db, v


def symplectic_eigenvalues(Gamma):
    """
    Compute the symplectic eigenvalues ν_i of a covariance matrix Γ.
    """
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * Omega @ Gamma)
    ν = np.sort(np.abs(eigvals))[::2]  # Take only one of each ν_i pair
    return ν

def symplectic_eigenvalues(Gamma):
    """
    Compute the symplectic eigenvalues ν_i of a covariance matrix Γ.
    """
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * Omega @ Gamma)
    ν = np.sort(np.abs(eigvals))[::2]  # Take only one of each ν_i pair
    return ν

def build_thermal_state_from_modular_hamiltonian(K, tol=1e-8):
    """
    Given a modular Hamiltonian K (real symmetric, 2n x 2n),
    returns the corresponding thermal Gaussian state's covariance matrix Gamma.

    K = S^{-T} E S^{-1}  ⇒  Gamma = S D S^T,  with D = 0.5 * coth(E/2)
    """
    # Ensure K is Hermitian
    K = 0.5 * (K + K.T)
    
    # Diagonalize K to get E and S
    eigvals, O = eigh(K)
    
    # Construct symplectic spectrum: epsilon_i = modular energy
    E = np.diag(eigvals)
    
    # Compute symplectic eigenvalues ν_i = 0.5 coth(ε_i / 2)
    epsilons = eigvals
    nu = np.zeros_like(epsilons)
    for i, eps in enumerate(epsilons):
        if np.abs(eps) < tol:
            nu[i] = 0.5  # Pure mode limit: coth(0) → ∞, but ν → 0.5
        else:
            nu[i] = 0.5 * 1.0 / np.tanh(0.5 * eps)
    
    # Build D matrix (repeated symplectic spectrum)
    D = np.diag(np.repeat(nu, 1))  # no double since epsilons already doubled for 2x2 blocks

    # Gamma = O D O^T
    Gamma = O @ D @ O.T

    # Symmetrize and return

    return 0.5 * (Gamma + Gamma.T), nu, epsilons


def build_quadratic_thermal_covariance(m2, k, N, beta):
    """
    Thermal covariance for
    H = 1/2 p^2 + 1/2 x^T V x
    with ring Laplacian.
    """

    # Build potential matrix V
    V = np.zeros((N, N))
    for i in range(N):
        V[i, i] = m2 + 2*k
        V[i, (i+1)%N] = -k
        V[i, (i-1)%N] = -k

    # Diagonalize V = O diag(omega^2) O^T
    omega2, O = np.linalg.eigh(V)
    omega = np.sqrt(omega2)

    # Thermal variances in normal modes
    coth = lambda x: 1/np.tanh(x)

    var_q = (1/(2*omega)) * coth(beta*omega/2)
    var_p = (omega/2) * coth(beta*omega/2)

    Gamma_xx = O @ np.diag(var_q) @ O.T
    Gamma_pp = O @ np.diag(var_p) @ O.T

    Gamma = np.block([
        [Gamma_xx, np.zeros((N,N))],
        [np.zeros((N,N)), Gamma_pp]
    ])

    return 0.5*(Gamma + Gamma.T)


def thermal_covariance_from_quadratic(H):
    """
    Given H (2n x 2n) symmetric positive matrix defining
    K = 1/2 r^T H r, returns covariance matrix Gamma for rho ∝ e^{-K} (β = 1).
    """
    # Diagonalize H: H = U Λ U^T
    eigvals, U = np.linalg.eigh(H)
    # sqrt(Λ)
    sqrtL = np.diag(np.sqrt(eigvals))
    inv_sqrtL = np.diag(1.0 / np.sqrt(eigvals))
    
    # Form coth(½ sqrtΛ) diag
    coth_diag = np.diag([np.cosh(0.5 * l) / np.sinh(0.5 * l) for l in np.sqrt(eigvals)])
    # Or equivalently: coth(½ √λ) = (e^{√λ} + 1)/(e^{√λ} - 1)
    
    # Build in eigenbasis: Gamma_eig = inv_sqrtL @ coth_diag @ inv_sqrtL
    Gamma_eig = inv_sqrtL @ coth_diag @ inv_sqrtL
    
    # Transform back
    Gamma = U @ Gamma_eig @ U.T
    return 0.5 * (Gamma + Gamma.T)


def heisenberg_evolution_operator(H, t, n):
    Omega = symplectic_form(n)
    return expm(Omega @ H * t)

def operator_spread_over_time(H, t_list, op_index=0):
    """
    Computes the Heisenberg evolution of operator r_op_index over time.
    
    Returns:
        coeffs_t: list of arrays of coefficients at each time
    """
    n = H.shape[0] // 2  # number of modes
    coeffs_t = []

    for t in t_list:
        S_t = heisenberg_evolution_operator(H, t, n)
        r0 = np.zeros(2 * n)
        r0[op_index] = 1.0  # evolve x_{op_index}(t)

        evolved = S_t @ r0
        coeffs_t.append(evolved)

    return np.array(coeffs_t)  # shape: (len(t_list), 2n)

def plot_light_cone(coeffs_t, title="Operator Spread"):
    """
    coeffs_t: (T x 2n) array
    """
    T, dim = coeffs_t.shape
    n = dim // 2

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # |x_i| coefficients over time
    im1 = axs[0].imshow(np.abs(coeffs_t[:, :n]), aspect='auto', cmap='inferno', origin='lower')
    axs[0].set_ylabel('Time step')
    axs[0].set_title('Contribution to x_i')

    # |p_i| coefficients over time
    im2 = axs[1].imshow(np.abs(coeffs_t[:, n:]), aspect='auto', cmap='inferno', origin='lower')
    axs[1].set_ylabel('Time step')
    axs[1].set_xlabel('Mode index')
    axs[1].set_title('Contribution to p_i')

    fig.colorbar(im1, ax=axs[0], orientation='vertical', label='|Coefficient|')
    fig.colorbar(im2, ax=axs[1], orientation='vertical', label='|Coefficient|')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

import numpy as np
from scipy.linalg import sqrtm

def build_tfd_purification(Gamma_A, tol=1e-8):
    """
    Construct a thermofield double (TFD) purification of a mixed Gaussian state Gamma_A.
    
    Parameters:
        Gamma_A: (2n x 2n) covariance matrix of the mixed state (single copy)
    
    Returns:
        Gamma_TFD: (4n x 4n) pure state covariance matrix of the TFD on two copies
    """
    from scipy.linalg import eigh
    from numpy.linalg import inv

    # 1. Williamson decomposition: Gamma_A = S D S^T
    S, D, ν = williamson(Gamma_A)

    # 2. Build off-diagonal entanglement block
    sqrt_ent = np.sqrt(D @ D - 0.25 * np.eye(len(D)))

    # 3. Full 4n × 4n pure covariance matrix
    big_S = np.block([
        [S, np.zeros_like(S)],
        [np.zeros_like(S), S]
    ])

    Gamma_pure_block = np.block([
        [D, sqrt_ent],
        [sqrt_ent, D]
    ])

    Gamma_TFD = big_S @ Gamma_pure_block @ big_S.T
    Gamma_TFD = 0.5 * (Gamma_TFD + Gamma_TFD.T)  # ensure symmetry

    return Gamma_TFD

def extract_subsystem_covariance(Gamma, indices):
    indices = np.array(indices)
    x_idx = indices
    p_idx = indices + Gamma.shape[0] // 2
    full_idx = np.concatenate([x_idx, p_idx])
    return Gamma[np.ix_(full_idx, full_idx)]

def trace_out_subsystem(Gamma, keep_indices):
    """
    Return the reduced covariance matrix for a Gaussian state
    by keeping only modes in keep_indices (x and p interleaved).

    keep_indices: list or array of mode indices to keep (0 to n-1)
    Assumes Gamma is in the (x_0,...x_n, p_0,...p_n) basis
    """
    n = Gamma.shape[0] // 2
    x_idx = np.array(keep_indices)
    p_idx = x_idx + n
    full_idx = np.concatenate([x_idx, p_idx])
    return Gamma[np.ix_(full_idx, full_idx)]

def von_neumann_entropy_alt(Gamma):
    n = Gamma.shape[0] // 2
    Omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])
    eigvals = np.linalg.eigvals(1j * Omega @ Gamma)
    nu = np.sort(np.abs(eigvals))[::2]
    nu = np.clip(nu, 0.5000000001, None)
    return sum((nu + 0.5)*np.log(nu + 0.5) - (nu - 0.5)*np.log(nu - 0.5))

def mutual_information(Gamma, idx_L, idx_R):
    S_L = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_L))
    S_R = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_R))
    S_LR = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_L + idx_R))
    return S_L + S_R - S_LR


def insert_two_mode_state_direct_sum(Gamma_system, insert_idx, Gamma_insert_2mode):
    """
    Inserts a 2-mode state (inserted + observer) into Gamma_system by:
    - Removing the inserted mode from Gamma_system entirely
    - Performing a direct sum with Gamma_insert_2mode
    - Permuting quadratures so inserted mode goes to insert_idx,
      observer goes to the end.

    Parameters:
        Gamma_system: (2n x 2n) real symmetric covariance matrix
        insert_idx: index (0 <= i < n) of mode to be replaced
        Gamma_insert_2mode: (4 x 4) covariance matrix of [inserted, observer]

    Returns:
        Gamma_extended: (2n x 2n) covariance matrix with inserted + observer
    """
    assert Gamma_insert_2mode.shape == (4, 4), "Gamma_insert_2mode must be 4×4"
    n = Gamma_system.shape[0] // 2
    assert Gamma_system.shape == (2*n, 2*n)


    # Permute the rows and columns
    Gamma_direct_sum = np.zeros((2*n+2,2*n+2))
    Gamma_direct_sum[0:2*n,0:2*n] = Gamma_system
    Gamma_permuted = Gamma_direct_sum.copy()
    Gamma_permuted[n+1:2*n+1,:]  = Gamma_direct_sum[n:2*n,:]
    Gamma_permuted[:,n+1:2*n+1] = Gamma_direct_sum[:,n:2*n]
    Gamma_permuted[n+1:2*n+1,n+1:2*n+1] = Gamma_system[n:2*n+1,n:2*n+1]
    Gamma_permuted[insert_idx,:]=0
    Gamma_permuted[:,insert_idx]=0
    Gamma_permuted[n,:]=0
    Gamma_permuted[:,n]=0
    Gamma_permuted[insert_idx+n+1,:]=0
    Gamma_permuted[:,insert_idx+n+1]=0
    Gamma_permuted[insert_idx,insert_idx] = Gamma_2mode[0,0]
    Gamma_permuted[insert_idx,n] = Gamma_2mode[0,1]
    Gamma_permuted[n,insert_idx] = Gamma_2mode[1,0]
    Gamma_permuted[n,n] = Gamma_2mode[1,1]
    Gamma_permuted[insert_idx+n+1,insert_idx+n+1]=Gamma_2mode[2,2]
    Gamma_permuted[insert_idx+n+1,2*n+1] = Gamma_2mode[2,3]
    Gamma_permuted[2*n+1,insert_idx+n+1] = Gamma_2mode[3,2]
    Gamma_permuted[2*n+1,2*n+1]=Gamma_2mode[3,3]

    return(.5*(Gamma_permuted+Gamma_permuted.T))



def pad_matrix_for_observer(H_sys, observer_modes=1):
    """
    Pad a (2n x 2n) Hamiltonian matrix H_sys by adding `observer_modes` that evolve trivially.
    Inserts observer position(s) after all x-quadratures and observer momentum(s) after all p-quadratures.

    Assumes canonical ordering [x_0, ..., x_{n-1}, p_0, ..., p_{n-1}]
    
    Returns:
        H_padded : (2(n + m) x 2(n + m)) np.array
    """
    assert H_sys.shape[0] == H_sys.shape[1], "H_sys must be square"
    n_sys = H_sys.shape[0] // 2
    m = observer_modes
    n_total = n_sys + m

    # Create full zero matrix
    H_padded = np.zeros((2 * n_total, 2 * n_total))

    # Fill top-left x block
    H_padded[0:n_sys, 0:n_sys] = H_sys[0:n_sys, 0:n_sys]                    # x-x
    H_padded[0:n_sys, n_total:n_total + n_sys] = H_sys[0:n_sys, n_sys:]    # x-p
    H_padded[n_total:n_total + n_sys, 0:n_sys] = H_sys[n_sys:, 0:n_sys]    # p-x
    H_padded[n_total:n_total + n_sys, n_total:n_total + n_sys] = H_sys[n_sys:, n_sys:]  # p-p

    return H_padded
    
def extract_mode_block(Gamma, mode_index):
    """
    Extract the 2×2 covariance matrix (x, p) block for one mode from full Gamma.
    Assumes Gamma is in (x0, ..., xn, p0, ..., pn) ordering.
    """
    n = Gamma.shape[0] // 2
    x_i = mode_index
    p_i = mode_index + n
    return Gamma[np.ix_([x_i, p_i], [x_i, p_i])]

def compute_MI_with_observer(Gamma, observer_idx, target_indices):
    # Gamma: 2n x 2n covariance matrix
    Gamma_obs = extract_mode_block(Gamma, observer_idx)
    Gamma_target = trace_out_subsystem(Gamma, target_indices)
    Gamma_joint = extract_subsystem_covariance(Gamma, target_indices + [observer_idx])
    
    S_obs = von_neumann_entropy_alt(Gamma_obs)
    S_target = von_neumann_entropy_alt(Gamma_target)
    S_joint = von_neumann_entropy_alt(Gamma_joint)
    
    return S_obs + S_target - S_joint


def total_mutual_information_with_observer(Gamma_total,n_total,idx_observer):
    all_indices = np.arange(n_total)  # all physical modes including observer
    ab_indices = np.setdiff1d(all_indices, [idx_observer])

    Gamma_C     = trace_out_subsystem(Gamma_total, [idx_observer])
    Gamma_AB    = trace_out_subsystem(Gamma_total, ab_indices)
    Gamma_ABC   = Gamma_total

    S_C    = von_neumann_entropy_alt(Gamma_C)
    S_AB   = von_neumann_entropy_alt(Gamma_AB)
    S_ABC  = von_neumann_entropy_alt(Gamma_ABC)

    I_C_AB = S_C + S_AB - S_ABC

    return I_C_AB

def extract_teleported_mode(Gamma, input_index_left, b):
    """
    Extract the corresponding mode on the right boundary.
    Assumes ordering: x_L, x_R, p_L, p_R.
    """
    n_total = Gamma.shape[0] // 2
    x_idx = input_index_left + b  # shift by boundary length to get right side
    p_idx = x_idx + n_total
    return Gamma[np.ix_([x_idx, p_idx], [x_idx, p_idx])]

def reorder_to_block_form(Gamma):
    """
    Reorders 2-mode covariance matrix from [x0,p0,x1,p1] to [x0,x1,p0,p1]
    """
    perm = [0, 2, 1, 3]
    return Gamma[np.ix_(perm, perm)]

def two_mode_squeezed_state(r):
    """
    Returns 4x4 covariance matrix for a two-mode squeezed vacuum.
    Mode 0: inserted into system
    Mode 1: external observer
    """
    ch = np.cosh(2 * r)
    sh = np.sinh(2 * r)
    Z = np.diag([1, -1])
    
    cov = 0.5 * np.block([
        [ch * np.eye(2),     sh * Z],
        [sh * Z,             ch * np.eye(2)]
    ])

    cov = reorder_to_block_form(cov)
    return cov


import numpy as np

def build_harmonic_chain_hamiltonian(N, m_squared=1.0, k=1.0):
    """
    Constructs a 2N x 2N positive-definite Hamiltonian matrix for a
    harmonic oscillator chain with nearest-neighbor interactions.

    Parameters:
        N (int): Number of oscillators
        m_squared (float): On-site mass term (positive)
        k (float): Coupling strength (positive)

    Returns:
        H (2N x 2N ndarray): Phase-space Hamiltonian matrix
    """
    # Potential energy (position-position block)
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = m_squared + 2 * k  # on-site + two neighbors
        if i > 0:
            K[i, i-1] = -k
        if i < N - 1:
            K[i, i+1] = -k

    # Kinetic energy (momentum block): mass = 1 → identity
    M = np.eye(N)

    # Full Hamiltonian in (x1,...,xN, p1,...,pN) basis
    H = np.block([
        [K,           np.zeros_like(K)],
        [np.zeros_like(M),  M]
    ])
    return H


def build_tfd_covariance(Gamma_A):
    """
    Given a (2n x 2n) thermal covariance matrix Gamma_A,
    returns the (4n x 4n) covariance matrix of the TFD purification.
    """
    S, D, nus = williamson(Gamma_A)
    n = len(nus)


    # Build blocks of 2-mode squeezed states in canonical basis
    blocks = []
    for nu in nus:
        if nu < 0.5:
            raise ValueError("Invalid symplectic eigenvalue < 0.5")
        delta = np.sqrt(nu**2 - 0.25)
        xpxp = np.array([
            [nu, 0, delta, 0],
            [0, nu, 0, -delta],
            [delta, 0, nu, 0],
            [0, -delta, 0, nu]
        ])
        P = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        block = P @ xpxp @ P.T
        blocks.append(block)

    Gamma_TFD_diag = block_diag(*blocks)  # 4n x 4n

    # Build S_total = S ⊕ S
    S_total = np.block([
        [S,           np.zeros_like(S)],
        [np.zeros_like(S), S]
    ])

    # Transform back to full basis
    Gamma_TFD = S_total @ Gamma_TFD_diag @ S_total.T
    return 0.5 * (Gamma_TFD + Gamma_TFD.T)  # ensure symmetry

def gaussian_purification(V):
    """
    Given a mixed Gaussian state with covariance V (2n x 2n),
    construct a purification (4n x 4n) using Weedbrook et al. Eq. (50)
    """
    S, D, nus = williamson_strawberry(V)

    Z = np.array([[1, 0], [0, -1]])
    C_blocks = [np.sqrt(nu**2 - .25) * Z for nu in nus]
    C = block_diag(*C_blocks)
    Gamma_2mode_list = []
    for nu in nus:
        Gamma_2mode = np.block([
            [nu * np.eye(2), np.sqrt(nu**2 - 1/4) * Z ],
            [np.sqrt(nu**2 - 1/4) * Z.T, nu * np.eye(2) ]
        ])
        Gamma_2mode_list.append(Gamma_2mode)

    # SC S^T block
    #S_C_ST = S @ C @ S.T

    # Final purified covariance matrix
    """
    V_purified = np.block([
        [V,         S_C_ST],
        [S_C_ST.T,  V]
    ])
    """
    V_pure2 = np.block([
        [D,  C],
        [C,  D]
    ])

    V_pure3 = xpxp_to_xxpp(V_pure2)


    V_pure = block_diag(*Gamma_2mode_list)


    S_trans = np.block([
        [S,np.zeros((S.shape[0],S.shape[0] ))],
        [np.zeros((S.shape[0],S.shape[0])), S]
    ])

    V_pure_phys = S_trans @ V_pure @ S_trans.T



    V_purified = np.block([
        [S @ D @ S.T,         S @ C @ S.T],
        [S @ C.T @ S.T,  S @ D @ S.T]
    ])



    """                
    V_purified2 = np.block([
        [S @ D @ S.T,         S @ C],
        [C.T @ S.T,  V]
    ])
    """

    # Symmetrize just in case
    #V_purified = 0.5 * (V_purified + V_purified.T)
    #V_purified = rearrange_blocks(V_purified)

    S_xxpp, Db_xxpp, nus = williamson_strawberry(V)
    alphas = np.sqrt(nus**2 - 0.25)

    C_top = np.diag(alphas)
    C_bottom = np.diag(-alphas)
    C_xxpp = np.block([
        [C_top,               np.zeros_like(C_top)],
        [np.zeros_like(C_bottom), C_bottom]
    ])   # 2n x 2n

    D_xxpp = Db_xxpp   # this is already diag(nu_1,...,nu_n, nu_1,...,nu_n)

    V_pure_will_xxpp = np.block([
        [D_xxpp, C_xxpp],
        [C_xxpp, D_xxpp]
    ])

    S_total = symplectic_direct_sum(S_xxpp.T, S_xxpp.T)  # or S_xxpp ⊕ I if you prefer
    V_pure_phys_xxpp = S_total @ V_pure_will_xxpp @ S_total.T

   
    return V_pure_phys_xxpp



def is_thermal(Gamma, tol=1e-10):
    """Checks if a covariance matrix corresponds to a thermal state."""
    Gamma = 0.5 * (Gamma + Gamma.T)  # Ensure symmetry
    n = Gamma.shape[0] // 2
    
    # Williamson decomposition
    S, D, nu = williamson(Gamma)
    
    # Transform to mode basis: should be block-diagonal with nu_i * I_2
    Gamma_diag = np.linalg.inv(S) @ Gamma @ np.linalg.inv(S.T)

    # Check if each 2x2 block is proportional to identity
    is_block_diag = True
    for i in range(n):
        block = Gamma_diag[2*i:2*i+2, 2*i:2*i+2]
        if not np.allclose(block, nu[i] * np.eye(2), atol=tol):
            is_block_diag = False
            break

    return is_block_diag, nu

def gaussian_fidelity_mixed(Gamma1, Gamma2):
    """
    Computes the fidelity between two 1-mode mixed Gaussian states
    assuming zero displacement (centered states).
    
    Parameters:
        Gamma1, Gamma2: 2x2 real symmetric covariance matrices
    
    Returns:
        Fidelity F ∈ [0, 1]
    """
    det1 = np.linalg.det(Gamma1)
    det2 = np.linalg.det(Gamma2)
    det_sum = np.linalg.det(Gamma1 + Gamma2)

    delta = (det1 - 0.25) * (det2 - 0.25)

    F = 1.0 / (np.sqrt(det_sum + delta) - np.sqrt(delta))
    return F

def insert_unentangled_mode(Gamma, mode_index, Gamma_insert):
    """
    Replace a single mode (x_i, p_i) in the covariance matrix with a new unentangled mode.

    Parameters:
        Gamma : (2n x 2n) np.array
            Original covariance matrix (x_0, ..., x_{n-1}, p_0, ..., p_{n-1})
        mode_index : int
            The index of the mode (0 <= i < n) to replace
        Gamma_insert : (2x2) np.array (optional)
            Covariance matrix for the inserted mode. If None, defaults to vacuum state.

    Returns:
        Gamma_new : (2n x 2n) np.array
            New covariance matrix with the mode replaced
    """
    n = Gamma.shape[0] // 2
    assert Gamma.shape == (2*n, 2*n), "Gamma must be 2n x 2n"
    assert 0 <= mode_index < n, "Invalid mode index"

    # Default inserted mode: vacuum (ν = 0.5, identity block)
    if Gamma_insert is None:
        Gamma_insert = 0.5 * np.eye(2)

    # Identify row/column indices for mode i
    x_i = mode_index
    p_i = mode_index + n
    idx_remove = [x_i, p_i]

    # Create new Gamma by replacing x_i and p_i rows/cols
    Gamma_new = Gamma.copy()

    # Zero out off-diagonal coupling to/from x_i and p_i
    Gamma_new[idx_remove, :] = 0
    Gamma_new[:, idx_remove] = 0

    # Insert new 2x2 unentangled block
    Gamma_new[np.ix_(idx_remove, idx_remove)] = Gamma_insert

    return Gamma_new

def rearrange_blocks(Gamma):
    N = Gamma.shape[0]
    Gamma_rearranged = np.copy(Gamma)
    # top left block
    Gamma_rearranged[N//4:N//2,0:N//4] = Gamma[N//2:3*N//4,0:N//4]
    Gamma_rearranged[0:N//4,N//4:N//2] = Gamma[0:N//4,N//2:3*N//4]
    Gamma_rearranged[N//4:N//2,N//4:N//2] = Gamma[N//2:3*N//4,N//2:3*N//4]
    # bottom left block
    Gamma_rearranged[N//2:3*N//4,0:N//4] = Gamma[N//4:N//2,0:N//4]
    Gamma_rearranged[3*N//4:N,N//4:N//2] = Gamma[3*N//4:N,N//2:3*N//4]
    # top right block
    Gamma_rearranged[0:N//4,N//2:3*N//4] = Gamma[0:N//4,N//4:N//2]
    Gamma_rearranged[N//4:N//2,3*N//4:N] = Gamma[N//2:3*N//4,3*N//4:N]
    #bottom right block
    Gamma_rearranged[N//2:3*N//4,N//2:3*N//4] = Gamma[N//4:N//2,N//4:N//2]
    Gamma_rearranged[3*N//4:N,N//2:3*N//4] = Gamma[3*N//4:N,N//4:N//2]
    Gamma_rearranged[N//2:3*N//4,3*N//4:N] = Gamma[N//4:N//2,3*N//4:N]
    return Gamma_rearranged 

def fidelity(V1,V2):
    n = V1.shape[0] // 2
    omega = symplectic_form(n)
    V_aux = omega.T @ inv(V1 + V2) @ (1/4 * omega + V2 @ omega @ V1)
    F_tot4 = det(2 * (sqrtm(np.eye(2*n)+1/4 * np.linalg.matrix_power(V_aux @ omega,-2))+np.eye(2*n)) @ V_aux)
    F_tot = F_tot4**.25
    F0 = F_tot/(det(V1+V2))**.25
    return F0

def score_from_covariance(Gamma, j, idx_obs, n_total):
    """
    Gamma in ordering [q_all, p_all] where q_all length = n_total.
    j and idx_obs are mode indices in q-ordering (0..n_total-1).
    returns ||Cov([q_j,p_j],[q_obs,p_obs])||_F^2
    """
    # indices in full Gamma
    rows = [j, j+n_total]
    cols = [idx_obs, idx_obs+n_total]
    C = Gamma[np.ix_(rows, cols)]  # 2x2 cross-cov
    return float(np.sum(C*C))

def signal_map_from_covariance(Gamma, idx_obs, n_total, exclude_obs=True):
    last = n_total-1 if exclude_obs else n_total
    scores = np.zeros(last)
    for j in range(last):
        scores[j] = score_from_covariance(Gamma, j, idx_obs, n_total)
    return scores




########
#Build TFD Hamiltonian
########


N = 20
k = 5
m_squared = 13
HL = np.zeros((N,N))
for i in range(N):
    if i < N//2-1:
        HL[i, i] = m_squared + 2 * k  # on-site + two neighbors
        HL[i,i+1] = -k
        HL[i+1, i] = -k
        #HL[i,i+2] = -k
        #HL[i+2,i] = -k
    #if i == N//2 - 2:
        #HL[i,1] = -k
        #HL[1,i] = -k        
    if i == N//2-1:
        HL[i,0] = -k
        HL[0,i] = -k 
        HL[i,i] = m_squared + 2 * k 
    if i > N//2-1:
        HL[i,i] = 1

"""
plt.imshow(np.abs(HL),vmax = ".5")
plt.colorbar()
plt.title("Nearest-Neighbor Modular Hamiltonian")
plt.show()
"""

mode_indices = np.linspace(0,HL.shape[0],HL.shape[0])
plt.plot(mode_indices, eigh(HL)[0])
plt.title("Nearest-neighbor Coupling Modular Hamiltonian Eigenvalues")
plt.xlabel("quadrature")
plt.ylabel("eigenvalue")
plt.show()



#HL = build_harmonic_chain_hamiltonian(4, m_squared=1.5, k=1.0)

Gamma_reconstructed, nu, eps_reconstructed = build_thermal_state_from_modular_hamiltonian(HL)

#Gamma_reconstructed = thermal_covariance_from_quadratic(HL)
print(is_thermal(Gamma_reconstructed))


#Gamma_TFD = build_tfd_covariance(Gamma_reconstructed)
Gamma_TFD = gaussian_purification(Gamma_reconstructed)

S_tot = von_neumann_entropy_alt(Gamma_TFD)





###########
# investigate spreading
###########

t0 = 4

t_list = np.linspace(0, t0, 100)  # 100 time steps from t=0 to t=10
coeffs_t = operator_spread_over_time(HL, t_list, op_index=0)  # evolve x_0(t)
plot_light_cone(coeffs_t, title="Light Cone of $x_0(t)$")





n = Gamma_TFD.shape[0] //2
b = Gamma_TFD.shape[0]//4
n_L = b

HL_full = np.zeros((2*n, 2*n))
HL_full[np.ix_(range(b), range(b))] = HL[:b, :b]                     # x-x
HL_full[np.ix_(range(b), range(n, n + b))] = HL[:b, b:]             # x-p
HL_full[np.ix_(range(n, n + b), range(b))] = HL[b:, :b]             # p-x
HL_full[np.ix_(range(n, n + b), range(n, n + b))] = HL[b:, b:]      # p-p


HR_full = np.zeros((2*n, 2*n))
HR_full[np.ix_(range(b, 2*b), range(b, 2*b))] = HL[:b, :b]
HR_full[np.ix_(range(b, 2*b), range(n + b, n + 2*b))] = HL[:b, b:]
HR_full[np.ix_(range(n + b, n + 2*b), range(b, 2*b))] = HL[b:, :b]
HR_full[np.ix_(range(n + b, n + 2*b), range(n + b, n + 2*b))] = HL[b:, b:]

H_LR = HL_full+HR_full

HL_full_padded = pad_matrix_for_observer(HL_full)

HR_full_padded = pad_matrix_for_observer(HR_full)


H_LR_padded = HL_full_padded+HR_full_padded

# Symplectic form
Omega = symplectic_form(n)

# Evolve backward in time


S_back = expm(-1 * Omega @ HL_full * t0)
Gamma_back = S_back @ Gamma_TFD @ S_back.T

T = 150
dt = t0/T


I_0 = mutual_information(Gamma_TFD, idx_L=list(range(b)), idx_R=list(range(n_L, n)))
I_mut = []

Gamma_LR = Gamma_TFD
times_back=np.linspace(0,t0,T)

for t in times_back:
    S_back_dt = expm(-1 * Omega @ HL_full * dt)
    Gamma_LR = S_back_dt @ Gamma_LR @ S_back_dt.T
    I = mutual_information(Gamma_LR, idx_L=list(range(n_L)), idx_R=list(range(n_L, n)))
    I_mut.append(I)



###########
# insert quantum information on one side
###########




# define a squeezed input state
#insert_idx = Gamma_LR.shape[0] // 8
insert_idx = 1

Gamma_2mode = two_mode_squeezed_state(r=1)


Gamma_with_observer = insert_two_mode_state_direct_sum(Gamma_back, insert_idx, Gamma_2mode)

###
# set squeezing
###
s = 1
Gamma_squeezed = 0.5 * np.array([[np.exp(-2*s), 0],
                                 [0, np.exp(2*s)]])

Gamma_insert_wigner = insert_unentangled_mode(Gamma_back, insert_idx, Gamma_insert=Gamma_squeezed)


observer_idx = Gamma_with_observer.shape[0] // 2 - 1 # new mode is last






#######
# evolve forwards in time
#######
n_total = (Gamma_with_observer.shape[0]) // 2  # now n+1
Omega_padded = symplectic_form(n_total)

#Omega = symplectic_form(Gamma_TFD.shape[0]//2)


S_forward = expm(Omega_padded @ HL_full_padded * t0)
Gamma_forward = S_forward @ Gamma_with_observer @ S_forward.T




I_obs_L = []
I_obs_R = []
I_tot = []
I_insert = []
I_telep = []
#I_telep_1 = []
#I_telep_4 = []
#I_telep_7 = []
#I_telep_11 = []



teleport_idx = insert_idx + Gamma_TFD.shape[0]//4



times_obs_forward = np.linspace(0,t0,T)
Gamma_LR = Gamma_with_observer
Gamma_LR_no_insert = Gamma_back
Gamma_LR_wigner = Gamma_insert_wigner

S_forward_no_insert = expm(Omega @ HL_full * dt)
S_forward_dt = expm(1 * Omega_padded @ HL_full_padded * dt)

scores_array = np.zeros((3*T,N))


for s,t in enumerate(times_obs_forward):
    Gamma_LR = S_forward_dt @ Gamma_LR @ S_forward_dt.T
    Gamma_LR_no_insert = S_forward_no_insert @ Gamma_LR_no_insert @ S_forward_no_insert.T
    Gamma_LR_wigner = S_forward_no_insert @ Gamma_LR_wigner @ S_forward_no_insert.T
    I_L = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L)))
    I_R = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L, n)))
    I_obs_L.append(I_L)
    I_obs_R.append(I_R)
    I = mutual_information(Gamma_LR, idx_L=list(range(n_L)), idx_R=list(range(n_L, n)))
    I_mut.append(I)
    I_tot.append(total_mutual_information_with_observer(Gamma_LR,n_total,observer_idx))
    I_insert.append(compute_MI_with_observer(Gamma_LR, observer_idx, [insert_idx]))
    I_telep.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx]))
    #I_telep_1.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+1]))
    #I_telep_4.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+4]))
    #I_telep_7.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+7]))
    #I_telep_11.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+11]))
    scores = signal_map_from_covariance(Gamma_LR, observer_idx, n_total=N+1, exclude_obs=True)
    scores_array[int(s),:] = scores/np.linalg.norm(scores)

initial_mut_info= []
for i in range(Gamma_TFD.shape[0]//2):
    initial_mut_info.append(compute_MI_with_observer(Gamma_LR, observer_idx, [i]))

plt.plot(np.arange(Gamma_TFD.shape[0]//2),initial_mut_info,color='k')
plt.axvline(insert_idx,color="blue",linestyle="dashed")
plt.axvline(teleport_idx,color="red",linestyle="dashed")
plt.xlabel("site")
plt.ylabel("mutual info with observer")
plt.title("mutual information with observer before coupling")
plt.legend()
plt.show()




#######
# couple the two sides
#######

n_total = Gamma_TFD.shape[0] // 2


bdy_len = Gamma_TFD.shape[0]//4

bdy_1_idx = np.arange(bdy_len)
bdy_2_idx = np.arange(bdy_len,2*bdy_len)

#carrier_indices = np.arange(0, bdy_len)  # don't skip teleportation qubit

carrier_indices1 = np.arange(0,insert_idx)
carrier_indices2 = np.arange(insert_idx+1,bdy_len)
carrier_indices = np.concatenate((carrier_indices1,carrier_indices2))

#carrier_indices = np.arange(insert_idx+6,bdy_len)

#carrier_indices = np.array([0,1,2,3,6,7,8,9,10,11,12,13,14,15])


#carrier_indices = np.array([1,2,3,7,8,10,11,12,14])

def idx_x(j): return j
def idx_p(j): return j + n_total


H = np.zeros((2*n_total, 2*n_total))
mu = 1
omega0 = np.sqrt(m_squared + 2*k)


"""
j = insert_idx

x_L = bdy_1_idx[j]
x_R = bdy_2_idx[j]
# x coupling
H[x_L, x_R] = H[x_R, x_L] = mu / 2
# p coupling
H[x_L + n_total, x_R + n_total] = H[x_R + n_total, x_L + n_total] = mu / 2
"""


for j in carrier_indices:
    x_L = bdy_1_idx[j]
    x_R = bdy_2_idx[j]
    # x coupling
    H[x_L, x_R] = H[x_R, x_L] = omega0*mu / 2
    # p coupling
    H[x_L + n_total, x_R + n_total] = H[x_R + n_total, x_L + n_total] = mu / (2*omega0)



"""
for i in carrier_indices:
    x_L = bdy_1_idx[i]
    for j in carrier_indices:
        x_R = bdy_2_idx[j]
        # x coupling
        H[x_L, x_R] = H[x_R, x_L] = mu / 2
        # p coupling
        H[x_L + n_total, x_R + n_total] = H[x_R + n_total, x_L + n_total] = mu / 2
"""


H_padded = pad_matrix_for_observer(H)

H += H_LR
H_padded += H_LR_padded


t_couple = 3
dt_couple = t_couple/T
S_coupling = expm(Omega_padded @ H_padded * t_couple)

Gamma_coupled = S_coupling @ Gamma_forward @ S_coupling.T



S_couple_no_insert = expm(Omega @ H * dt_couple)


times_obs_coupling = np.linspace(t0,t0+t_couple,T)
for s,t in enumerate(times_obs_coupling):
    S_couple_dt = expm(1 * Omega_padded @ H_padded * dt_couple)
    Gamma_LR = S_couple_dt @ Gamma_LR @ S_couple_dt.T
    Gamma_LR_no_insert = S_couple_no_insert @ Gamma_LR_no_insert @ S_couple_no_insert.T
    Gamma_LR_wigner = S_couple_no_insert @ Gamma_LR_wigner @ S_couple_no_insert.T
    I_L = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L)))
    I_R = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L, n)))
    I_obs_L.append(I_L)
    I_obs_R.append(I_R)
    I = mutual_information(Gamma_LR, idx_L=list(range(n_L)), idx_R=list(range(n_L, n)))
    I_mut.append(I)
    I_tot.append(total_mutual_information_with_observer(Gamma_LR,n_total,observer_idx))
    I_insert.append(compute_MI_with_observer(Gamma_LR, observer_idx, [insert_idx]))
    I_telep.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx]))
    #I_telep_1.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+1]))
    #I_telep_4.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+4]))
    #I_telep_7.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+7]))
    #I_telep_11.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+11]))
    scores = signal_map_from_covariance(Gamma_LR, observer_idx, n_total=N+1, exclude_obs=True)
    scores_array[int(T+s),:] = scores/np.linalg.norm(scores)

######
# evolve state forwards in time with KR
######





S_final = expm(Omega_padded @ HR_full_padded * t0)
Gamma_final = S_final @ Gamma_coupled @ S_final.T


tf = 6*t0


S_final_no_insert = expm(1 * Omega @ HR_full* dt)
times_obs_final = np.linspace(t0+t_couple,2*t0+t_couple,T)
for s,t in enumerate(times_obs_forward):
    S_final_dt = expm(1 * Omega_padded @ HR_full_padded * dt)
    Gamma_LR = S_final_dt @ Gamma_LR @ S_final_dt.T
    Gamma_LR_no_insert = S_final_no_insert @ Gamma_LR_no_insert @ S_final_no_insert.T
    Gamma_LR_wigner = S_final_no_insert @ Gamma_LR_wigner @ S_final_no_insert.T
    I_L = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L)))
    I_R = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L, n)))
    I_obs_L.append(I_L)
    I_obs_R.append(I_R)
    I = mutual_information(Gamma_LR, idx_L=list(range(n_L)), idx_R=list(range(n_L, n)))
    I_mut.append(I)
    I_tot.append(total_mutual_information_with_observer(Gamma_LR,n_total,observer_idx))
    I_insert.append(compute_MI_with_observer(Gamma_LR, observer_idx, [insert_idx]))
    I_telep.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx]))
    #I_telep_1.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+1]))
    #I_telep_4.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+4]))
    #I_telep_7.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+7]))
    #I_telep_11.append(compute_MI_with_observer(Gamma_LR, observer_idx, [teleport_idx+11]))
    scores = signal_map_from_covariance(Gamma_LR, observer_idx, n_total=N+1, exclude_obs=True)
    scores_array[int(2*T+s),:] = scores/np.linalg.norm(scores) 


times_score = [0,T/2,T-1,3/2*T,2*T-1,5/2*T,3*T-1]

for t in range(len(times_score)):
    plt.plot(np.arange(N),scores_array[int(times_score[t]),:],label=f"t={times_score[t]}")
plt.xlabel("site")
plt.ylabel("correlation with observer")
plt.legend()
plt.show()




times_bdy_info = np.concatenate((times_back,times_obs_forward+times_back[-1],times_obs_coupling+times_back[-1],times_obs_final+times_back[-1]))
#times = np.concatenate((times1,times2,times3))

plt.plot(times_bdy_info,I_mut,"k",label = "I_mut")
#plt.axvline(times_back[-1],color="green",label = "back")
#plt.axvline(times_obs_forward[-1]+times_back[-1],color="red",label = "forward")
#plt.axvline(times_obs_coupling[-1]+times_back[-1],color = "blue",label = "couple")
#plt.axvline(times_obs_final[-1]+times_back[-1],color = "orange",label = "final")

plt.xlabel("time (t*||K_L||)")
plt.ylabel("mutual information")
plt.legend()
plt.show()


final_mut_info= []
for i in range(Gamma_TFD.shape[0]//2):
    final_mut_info.append(compute_MI_with_observer(Gamma_LR, observer_idx, [i]))

plt.plot(np.arange(Gamma_TFD.shape[0]//2),final_mut_info,color='k')
plt.axvline(insert_idx,color="blue",linestyle="dashed")
plt.axvline(teleport_idx,color="red",linestyle="dashed")
plt.xlabel("site")
plt.ylabel("mutual info with observer")
plt.title("mutual information with observer after coupling")
plt.legend()
plt.show()



times = np.concatenate((times_obs_forward,times_obs_coupling,times_obs_final))

#plt.rc('font', size=28) 
#linewidth=4

plt.plot(times,I_obs_L,"k",label = "mutual info with left")
plt.plot(times,I_obs_R,"r",label = "mutual info with right")
#plt.plot(times,I_tot,"blue",linewidth=4,label = "total mutual info")
#plt.plot(times,I_insert,"green",label = "mutual info with insert")
#plt.plot(times,I_telep,"orange",label = "mutual info with teleported")
#plt.plot(times,I_telep_1,"magenta",label = "mutual info with teleported+1")
#plt.plot(times,I_telep_4,"cyan",label = "mutual info with teleported+4")
#plt.plot(times,I_telep_7,"lime",label = "mutual info with teleported+7")
#plt.plot(times,I_telep_11,"darkviolet",label = "mutual info with teleported+11")

plt.axvline(t0)


#plt.axvline(t0*2,label = "kick")
plt.xlabel("time")
plt.ylabel("mutual info with observer")
plt.title("Mutual Information vs. Time")
plt.legend()
plt.show()

"""
teleported_idx = bdy_len + insert_idx


mut_info_insert_regions = []
mut_info_telep_regions = []

lengths_array = np.linspace(1,Gamma_LR.shape[0] // 8,Gamma_LR.shape[0] // 8)
for i in range(1,lengths_array.shape[0]):
    #if i == 0:
    #segment_telep = [teleported_idx]
    segment_insert = np.arange(insert_idx - i, insert_idx + i+1)
    segment_insert = np.ndarray.tolist(segment_insert)
    mut_info_insert_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, segment_insert))
    segment_telep = np.arange(teleported_idx - i, teleported_idx + i+1)
    segment_telep = np.ndarray.tolist(segment_telep)
    mut_info_telep_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, segment_telep))

mut_info_insert_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L))))
mut_info_telep_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L,n))))


full_lengths_array = 2 * lengths_array + 1
full_lengths_array[-1] = 64    
plt.plot(full_lengths_array,mut_info_insert_regions,color='k',label = "insert side")
plt.plot(full_lengths_array,mut_info_telep_regions,color='red', label ="teleport side")
plt.axhline(compute_MI_with_observer(Gamma_LR,observer_idx,list(range(n_total))), color = "blue", label = "total mutual info with observer")
plt.xlabel("length of segment")
plt.ylabel("mutual info with observer")
plt.title("mutual information of segments centered around teleported site")
plt.legend()
plt.show()

"""


mut_info_insert_regions = []
mut_info_telep_regions = []

lengths_array = np.linspace(1,Gamma_TFD.shape[0] // 8,Gamma_TFD.shape[0] // 8)
center_idx = Gamma_TFD.shape[0] // 8

for i in range(1,lengths_array.shape[0]):
    #if i == 0:
    #segment_telep = [teleported_idx]
    if center_idx - i >= 0 and center_idx + i < Gamma_TFD.shape[0]//4:
        segment_insert = np.arange(center_idx - i, center_idx + i+1)
    if center_idx - i < 0:
        diff = np.abs(center_idx - i)
        segment_insert_1 = np.arange(Gamma_TFD.shape[0]//4-diff,Gamma_TFD.shape[0]//4)
        segment_insert_2 = np.arange(0,center_idx+i+1)
        segment_insert = np.concatenate((segment_insert_1,segment_insert_2))
    if center_idx + i >= Gamma_TFD.shape[0]//4:
        diff = center_idx + i - Gamma_TFD.shape[0]//4
        segment_insert_1 = np.arange(center_idx-i,Gamma_TFD.shape[0]//4)
        segment_insert_2 = np.arange(0,diff+1)
        segment_insert = np.concatenate((segment_insert_1,segment_insert_2))
    segment_insert = np.ndarray.tolist(segment_insert)
    mut_info_insert_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, segment_insert))
    if center_idx  - i >= 0  and center_idx  + i < Gamma_TFD.shape[0]//4:
        segment_telep = np.arange(center_idx + Gamma_TFD.shape[0]//4 - i, center_idx + Gamma_TFD.shape[0]//4 + i+1)
    if center_idx - i < 0 :
        diff = np.abs(center_idx - i)
        segment_telep_1 = np.arange(Gamma_TFD.shape[0]//2-diff,Gamma_TFD.shape[0]//2)
        segment_telep_2 = np.arange(Gamma_TFD.shape[0] // 4 ,center_idx + Gamma_TFD.shape[0] // 4 + i+1)
        segment_telep = np.concatenate((segment_telep_1,segment_telep_2))
    if center_idx + i >= Gamma_TFD.shape[0]//4:
        diff = center_idx + i - Gamma_TFD.shape[0]//4
        segment_telep_1 = np.arange(center_idx + Gamma_TFD.shape[0] // 4 - i,Gamma_TFD.shape[0]//2)
        segment_telep_2 = np.arange(Gamma_TFD.shape[0] // 4,Gamma_TFD.shape[0] //4 + diff+1)
        segment_telep = np.concatenate((segment_telep_1,segment_telep_2))
    segment_telep = np.ndarray.tolist(segment_telep)
    mut_info_telep_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, segment_telep))



mut_info_insert_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L))))
mut_info_telep_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L,n))))


full_lengths_array = 2 * lengths_array + 1
full_lengths_array[-1] = N // 2

#plt.rc('font', size=28) 
#linewidth=4
plt.plot(full_lengths_array,mut_info_insert_regions,color='k',label = "insert side")
plt.plot(full_lengths_array,mut_info_telep_regions,color='red', label ="teleport side")
plt.axhline(compute_MI_with_observer(Gamma_LR,observer_idx,list(range(n_total))),color = "blue", label = "total mutual info with observer")
plt.xlabel("length of segment")
plt.ylabel("mutual info with observer")
plt.title("Mutual Information of Segments")
plt.legend()
plt.show()


Gamma_input = extract_mode_block(Gamma_2mode,0)

flip = np.array([[-1,0],[0,-1]])

site_fidelity = []
site_fidelity_flip = []

for j in range(N//2,N):
    site_fidelity.append(gaussian_fidelity_mixed(Gamma_input,extract_subsystem_covariance(Gamma_LR,[j])))

plt.plot(np.arange(N//2,N),site_fidelity)
plt.xlabel("site")
plt.ylabel("fidelity")
plt.show()

Gamma_teleported = extract_mode_block(Gamma_LR_wigner, teleported_idx-1)
Gamma_teleported2 = extract_mode_block(Gamma_LR_no_insert, teleported_idx-1)


print(0.5 * np.eye(2))
Gamma_out_real = 0.5 * (Gamma_teleported + Gamma_teleported.conj().T)
print(Gamma_out_real)

Gamma_out_real2 = 0.5 * (Gamma_teleported2 + Gamma_teleported2.conj().T)
#print(Gamma_out_real2)



telep_eigvals = symplectic_eigenvalues(Gamma_out_real)
print("telep eigvals = ", telep_eigvals)


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_wigner_ellipse(Gamma_mode, ax, label='', color='blue'):
    from scipy.linalg import eigh
    W = Gamma_mode[:2, :2].real  # just x, p block
    vals, vecs = eigh(W)
    width, height = 2 * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                      edgecolor=color, fc='None', lw=2, label=label)
    ax.add_patch(ellipse)

fig, ax = plt.subplots()
#plot_wigner_ellipse(np.array([[0.5, 0], [0, 0.5]]), ax, label='Vacuum', color='blue')
plot_wigner_ellipse(Gamma_squeezed, ax, label='Input', color='green')
plot_wigner_ellipse(Gamma_out_real, ax, label='Output', color='red')
plot_wigner_ellipse(Gamma_out_real2, ax, label='No Input', color='k')
#plot_wigner_ellipse(Gamma_out_shift, ax, label='No Input', color='orange')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel("Position Quadrature")
ax.set_ylabel("Momentum Quadrature")
ax.set_aspect('equal')
ax.legend()
plt.title("Input vs Output Wigner Ellipses")
plt.grid(True)
plt.show()





Gamma_output = Gamma_out_real


f = fidelity(Gamma_squeezed,Gamma_out_real)
print("f=",f)

fidelity = gaussian_fidelity_mixed(Gamma_input, Gamma_output)
print(f"Fidelity of teleportation: {fidelity:.4f}")

print(Gamma_out_real)

print("done")
