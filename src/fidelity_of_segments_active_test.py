import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, expm, sqrtm, schur, block_diag, eigh, det, polar
from thewalrus.symplectic import xpxp_to_xxpp, sympmat

def symplectic_form(n):
    """Returns the 2n × 2n symplectic form Omega"""
    return np.block([
        [np.zeros((n, n),dtype=np.float64), np.eye(n,dtype=np.float64)],
        [-np.eye(n,dtype=np.float64), np.zeros((n, n),dtype=np.float64)]
    ])


def extract_subsystem_covariance(Gamma, indices):
    indices = np.array(indices)
    x_idx = indices
    p_idx = indices + Gamma.shape[0] // 2
    full_idx = np.concatenate([x_idx, p_idx])
    return Gamma[np.ix_(full_idx, full_idx)]

def von_neumann_entropy_alt(Gamma):
    n = Gamma.shape[0] // 2
    Omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])
    eigvals = np.linalg.eigvals(1j * Gamma @ Omega)
    nu = np.sort(np.abs(eigvals))[::2]
    nu = np.clip(nu, 0.500001, None)
    return sum((nu + 0.5)*np.log(nu + 0.5) - (nu - 0.5)*np.log(nu - 0.5))

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

    if diffn >= 10**(-5):
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
    return np.linalg.inv(S).T, Db, v



def symplectic_eigenvalues(Gamma):
    """
    Compute the symplectic eigenvalues ν_i of a covariance matrix Γ.
    """
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * Gamma @ Omega)
    ν = np.sort(np.abs(eigvals))[::2]  # Take only one of each ν_i pair
    return ν


def momentum_projection_matrix(m):
    P = np.zeros((2*m, 2*m))
    P[m:, m:] = np.eye(m)
    return P


def momentum_measured_1(Gamma,un_set,meas_set):
    na = un_set.shape[0]//2
    nb = meas_set.shape[0]//2

    Gamma_AA = np.zeros((4*na,4*na))
    for i in range(4):
        for j in range(4):
            Gamma_AA[i*na:(i+1)*na,j*na:(j+1)*na] = Gamma[(i+1)*nb+i*na:(i+1)*nb+(i+1)*na,(j+1)*nb+j*na:(j+1)*nb+(j+1)*na]
        
    Gamma_BB = np.zeros((4*nb,4*nb))
    for i in range(4):
        for j in range(4):
            Gamma_BB[i*nb:(i+1)*nb,j*nb:(j+1)*nb] = Gamma[i*nb+i*na:(i+1)*nb+i*na,j*nb+j*na:(j+1)*nb+j*na]

    Gamma_AB = np.zeros((4*na,4*nb))
    for i in range(4):
        for j in range(4):
            Gamma_AB[i*na:(i+1)*na,j*nb:(j+1)*nb] = Gamma[(i+1)*nb+i*na:(i+1)*nb+(i+1)*na,j*nb+j*na:(j+1)*nb+j*na]

    m = Gamma_BB.shape[0]//2
    P = momentum_projection_matrix(m)
    V_bdy = Gamma_AA - Gamma_AB @ np.linalg.pinv(P @ Gamma_BB @ P) @ Gamma_AB.T

    return V_bdy




def mutual_information(Gamma, idx_L, idx_R):
    S_L = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_L))
    S_R = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_R))
    S_LR = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_L + idx_R))
    return S_L + S_R - S_LR

def construct_modular_hamiltonian_with_pinning(Gamma, epsilon_max=15, tol=1e-6):
    """
    Constructs the modular Hamiltonian K for a mixed Gaussian state Γ,
    assigning very high energy to pure modes (ν ≈ 0.5).
    """
    S, D, V = williamson_strawberry(Gamma)
    delta = 1e-5
    # Modular energies
    epsilons = []
    for v in V:
        if np.abs(v - 0.5) < tol or v < .5:
            epsilons.append(epsilon_max)  # Pin pure modes

        else:
            #eps = np.log((v + 0.5) / (v - 0.5))
            eps = 2*np.arctanh(1/(2*v))

            epsilons.append(eps)
    
    E_diag = np.diag(np.repeat(epsilons, 2))
    # Modular Hamiltonian: K = S^{-T} E S^{-1}
    S_inv = inv(S)
    K = S_inv.T @ E_diag @ S_inv
    return K

def insert_two_mode_state_direct_sum(Gamma_system, insert_idx, Gamma_2mode):
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
    assert Gamma_2mode.shape == (4, 4), "Gamma_insert_2mode must be 4×4"
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

def covmat_to_hamil(V, tol=1e-5):  # pragma: no cover
    #V = .5*(V + V.T)
    r"""Converts a covariance matrix to a Hamiltonian.

    Given a covariance matrix V of a Gaussian state :math:`\rho` in the xp ordering,
    finds a positive matrix :math:`H` such that

    .. math:: \rho = \exp(-Q^T H Q/2)/Z

    where :math:`Q = (x_1,\dots,x_n,p_1,\dots,p_n)` are the canonical
    operators, and Z is the partition function.

    For more details, see https://arxiv.org/abs/1507.01941

    Args:
        V (array): Gaussian covariance matrix
        tol (int): the number of decimal places to use when determining if the matrix is symmetric

    Returns:
        array: positive definite Hamiltonian matrix
    """
    (n, m) = V.shape
    if n != m:
        raise ValueError("Input matrix must be square")
    if np.linalg.norm(V - np.transpose(V)) >= tol:
        raise ValueError("The input matrix is not symmetric")

    n = n // 2
    omega = sympmat(n)

    vals = np.linalg.eigvalsh(V)
    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    W = 1j *  omega @ V
    l, v = np.linalg.eig(W)
    H = (1j * omega @ (v @ np.diag(np.arctanh(1.0 / 2*l.real)) @ np.linalg.inv(v))).real

    return H

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

def gaussian_purification(V):
    """
    Given a mixed Gaussian state with covariance V (2n x 2n),
    construct a purification (4n x 4n) using Weedbrook et al. Eq. (50)
    """
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

def build_ring_potential(N, k, m2):
    """V in H = 1/2 p^T p + 1/2 x^T V x for a periodic ring."""
    V = np.zeros((N, N), dtype=float)
    for i in range(N):
        V[i, i] = m2 + 2.0 * k
        V[i, (i + 1) % N] = -k
        V[i, (i - 1) % N] = -k

    return 0.5 * (V + V.T)


def thermal_cov_one_side_from_modes(O, omega, beta):
    """
    One-side thermal covariance (2N×2N) in xxpp ordering [x1..xN, p1..pN].
    """
    N = len(omega)
    nu = 0.5 * _coth(0.5 * beta * omega)          # symplectic spectrum of each normal mode
    var_x = nu / omega                             # <x^2>
    var_p = nu * omega                             # <p^2>

    Gamma_xx = O @ np.diag(var_x) @ O.T
    Gamma_pp = O @ np.diag(var_p) @ O.T
    Gamma = np.block([[Gamma_xx, np.zeros((N, N))],
                      [np.zeros((N, N)), Gamma_pp]])
    return 0.5 * (Gamma + Gamma.T), nu

def _coth(x):
    # stable-ish coth for moderate x
    return 1.0 / np.tanh(x)

def tfd_cov_ring_from_normal_modes(N, k, m2, V, beta, eps_omega=1e-15):
    """
    Construct the *pure* TFD covariance matrix for the ring Hamiltonian
        H = 1/2 p^T p + 1/2 x^T V x
    at inverse temperature beta, using the normal-mode diagonalization of V.

    Output ordering (4N×4N) is:
        [x_L(1..N), x_R(1..N), p_L(1..N), p_R(1..N)]   (xxpp with LR split)

    This construction is an *analytic* Gaussian purification mode-by-mode, so in exact arithmetic
    symplectic eigenvalues of the 4N-mode state are exactly 0.5.
    """
    #V = build_ring_potential(N, k, m2)

    # V = O diag(omega^2) O^T
    omega2, O = np.linalg.eigh(V)
    omega2 = np.clip(omega2, eps_omega, None)
    omega = np.sqrt(omega2)

    # Thermal invariants per normal mode
    nu = 0.5 * _coth(0.5 * beta * omega)                 # >= 0.5
    alpha = np.sqrt(np.maximum(nu * nu - 0.25, 0.0))     # correlations for purification

    # In normal-mode basis, build 4N×4N covariance for TFD:
    # blocks in xxpp with LR split:
    #   xx: [ diag(nu/ω)     diag(alpha/ω)
    #         diag(alpha/ω)  diag(nu/ω)     ]
    #
    #   pp: [ diag(nu*ω)     diag(-alpha*ω)
    #         diag(-alpha*ω) diag(nu*ω)     ]
    #
    #   xp = px = 0
    Dx  = np.diag(nu / omega)
    Dp  = np.diag(nu * omega)
    Cx  = np.diag(alpha / omega)
    Cp  = np.diag(-alpha * omega)

    xx_nm = np.block([[Dx, Cx],
                      [Cx, Dx]])
    pp_nm = np.block([[Dp, Cp],
                      [Cp, Dp]])

    Gamma_nm = np.block([[xx_nm, np.zeros((2*N, 2*N))],
                         [np.zeros((2*N, 2*N)), pp_nm]])

    # Transform back to site basis on BOTH L and R, for x and p:
    # x_L = O x'_L, x_R = O x'_R, p_L = O p'_L, p_R = O p'_R
    O2 = np.block([[O, np.zeros((N, N))],
                   [np.zeros((N, N)), O]])   # acts on (L,R) index within x-block or p-block
    S = np.block([[O2, np.zeros((2*N, 2*N))],
                  [np.zeros((2*N, 2*N)), O2]])

    Gamma_site = S @ Gamma_nm @ S.T
    Gamma_site = 0.5 * (Gamma_site + Gamma_site.T)

    # Also return the one-side thermal covariance (useful sanity check)
    Gamma_th, nu_check = thermal_cov_one_side_from_modes(O, omega, beta)

    # purity check
    nu_tfd = symplectic_eigenvalues(Gamma_site)   # should be ~0.5 for all 2N modes
    return Gamma_site

#define coupling Hamiltonian

# Global oscillator indices of left and right boundaries
#bdy_len = 2**(L - 1)         # e.g. 128
#bdy_1 = np.arange(N - bdy_len, N)               # left boundary: physical indices
#bdy_2 = np.arange(N_tot - bdy_len, N_tot)       # right boundary: physical indices

# Map these physical indices into the post-measurement (Gamma_TFD) indexing
# You need to find where each bdy_1 and bdy_2 element lies in un_set
#lookup = {node: i for i, node in enumerate(un_set)}
#bdy_1_idx = np.array([lookup[x] for x in bdy_1])
#bdy_2_idx = np.array([lookup[x] for x in bdy_2])

N=3
bdy_len = N
bdy_1_idx = np.arange(bdy_len)
bdy_2_idx = np.arange(bdy_len,2*bdy_len)

#carrier_indices = np.arange(0, bdy_len)  # skip teleportation qubit

insert_idx = 1
carrier_indices1 = np.arange(0,insert_idx)
carrier_indices2 = np.arange(insert_idx+1,bdy_len)
carrier_indices = np.concatenate((carrier_indices1,carrier_indices2))

def idx_x(j): return j
def idx_p(j): return j + n_total

n_total = 2*bdy_len
H_coupling_OG = np.zeros((2*n_total, 2*n_total))
mu = 1
k = 6
m_squared=13
omega0 = np.sqrt(m_squared + 2*k)
#omega0=1

for j in carrier_indices:
    x_L = bdy_1_idx[j]
    x_R = bdy_2_idx[j]
    # x coupling
    H_coupling_OG[x_L, x_R] = H_coupling_OG[x_R, x_L] = mu*omega0 / 2
    # p coupling
    H_coupling_OG[x_L + n_total, x_R + n_total] = H_coupling_OG[x_R + n_total, x_L + n_total] = mu / (2*omega0)
"""
L = 4
Lh = 3
n_tube = 0
g_tube = 1
mu_A = 1
mu_B = 1
mu_s = 1
t = 10

# Build the graph
N = 2**(Lh - 1) * (2**(L - Lh + 1) - 1)
bdy_len = 2**(L - 1)
bdy_1 = np.arange(N - bdy_len, N)
N_tot = 2 * N + n_tube * 2**(Lh - 1)
bdy_2 = np.arange(N_tot - bdy_len, N_tot)

# Build base adjacency matrix A
A = np.zeros((N, N), dtype=np.float64)
for l1 in range(Lh, L + 1):
    for s1 in range(1, 2**(l1 - 1) + 1):
        for l2 in range(Lh, L + 1):
            for s2 in range(1, 2**(l2 - 1) + 1):
                prev1 = sum(2**(k - 1) for k in range(Lh, l1))
                prev2 = sum(2**(k - 1) for k in range(Lh, l2))
                ind1 = prev1 + s1 - 1
                ind2 = prev2 + s2 - 1
                if l1 == l2 and (abs(s1 - s2) == 1 or abs(s1 - s2) == 2**(l1 - 1) - 1):
                    A[ind1, ind2] = mu_s
                if l2 == l1 + 1 and s2 in [2*s1, 2*s1 - 1]:
                    A[ind1, ind2] = mu_s
                if l1 == l2 + 1 and s1 in [2*s2, 2*s2 - 1]:
                    A[ind1, ind2] = mu_s

# Full adjacency with duplicated regions and tube
A_tot = np.zeros((N_tot, N_tot),dtype=np.float64)
A_tot[:N, :N] = A
A_tot[N_tot - N:, N_tot - N:] = A
hor_1 = np.arange(2**(Lh - 1))
for ell in range(n_tube + 1):
    offset = N + (ell - 1) * 2**(Lh - 1)
    if ell == 0:
        for i in hor_1:
            A_tot[i, i + N] = A_tot[i + N, i] = g_tube
    elif ell > 0:
        for i in hor_1:
            A_tot[i + offset, i + offset + 2**(Lh - 1)] = g_tube
            A_tot[i + offset + 2**(Lh - 1), i + offset] = g_tube
            # Horizontal connections
            if i < 2**(Lh - 1) - 1:
                A_tot[i + offset, i + offset + 1] = A_tot[i + offset + 1, i + offset] = g_tube
            else:
                A_tot[i + offset, i + offset - (2**(Lh - 1) - 1)] = A_tot[i + offset - (2**(Lh - 1) - 1), i + offset] = g_tube
 
# Index sets
un_set = np.concatenate([bdy_1, bdy_2])
meas_set = np.setdiff1d(np.arange(N_tot), un_set)


Gamma_0 =.5 * np.eye(2*N_tot,dtype=np.complex128)




# Number of total modes
n = N_tot

# Default: mass = 1, so kinetic term is identity
M = 1* np.eye(n)
D = np.zeros((n,n))
for i in range(n):
    D[i,i]=sum(A_tot[i,:])

# Potential term = adjacency + onsite mass term
mu_squared = 0  # Choose this to control oscillator frequency
K = D - A_tot + mu_squared * np.eye(n)

# Construct full Hamiltonian H (2n x 2n) in (x1..xn, p1..pn) basis
H = np.block([
    [K,         np.zeros((n, n))],
    [np.zeros((n, n)),   M     ]
])


n = Gamma_0.shape[0] // 2
Omega = symplectic_form(n)
S_t = expm(Omega @ H * t)
Gamma_q = S_t @ Gamma_0 @ S_t.T


Gamma_TFD = momentum_measured_1(Gamma_q,un_set,meas_set)


b = bdy_len
keep = np.arange(b)  # keep left boundary
Gamma_reduced = trace_out_subsystem(Gamma_TFD, keep)

#HL = covmat_to_hamil(Gamma_reduced)
HL = construct_modular_hamiltonian_with_pinning(Gamma_reduced)
"""
HL = np.zeros((2*N,2*N))
for i in range(2*N):
    if i < N-1:
        HL[i, i] = m_squared + 2 * k  # on-site + two neighbors
        HL[i,i+1] = -k
        HL[i+1, i] = -k    
    if i == N-1:
        HL[i,0] = -k
        HL[0,i] = -k 
        HL[i,i] = m_squared + 2 * k 
    if i > N-1:
        HL[i,i] = 1

#N = Gamma_TFD.shape[0]//4
HL_full = np.zeros((4*N, 4*N))
HL_full[np.ix_(range(N), range(N))] = HL[:N, :N]                     # x-x
HL_full[np.ix_(range(N), range(2*N, 3*N))] = HL[:N, N:]             # x-p
HL_full[np.ix_(range(2*N, 3*N), range(N))] = HL[N:, :N]             # p-x
HL_full[np.ix_(range(2*N, 3*N), range(2*N, 3*N))] = HL[N:, N:]      # p-p


HR_full = np.zeros((4*N, 4*N))
HR_full[np.ix_(range(N, 2*N), range(N, 2*N))] = HL[:N, :N]
HR_full[np.ix_(range(N, 2*N), range(3*N, 4*N))] = HL[:N, N:]
HR_full[np.ix_(range(3*N, 4*N), range(N, 2*N))] = HL[N:, :N]
HR_full[np.ix_(range(3*N, 4*N), range(3*N, 4*N))] = HL[N:, N:]

H_LR = HL_full+HR_full

H_coupling_OG += H_LR


def teleportation_protocol(s,theta,insert_idx,wormhole,n_one_side,H_coupling,coupling):
    q = insert_idx
    if wormhole == False:
        N = 2*n_one_side
        k = 6
        m_squared = 13
        HL = np.zeros((N,N))
        
        for i in range(N):
            if i < N//2-1:
                HL[i, i] = m_squared + 2 * k  # on-site + two neighbors
                HL[i,i+1] = -k
                HL[i+1, i] = -k 
                 
            if i == N//2-1:
                HL[i,0] = -k
                HL[0,i] = -k 
                HL[i,i] = m_squared + 2 * k 
            if i > N//2-1:
                HL[i,i] = 1
        """
        HL_rand=np.zeros((N,N))
        for i in range(N):
            a = np.random.uniform(.2,2)
            if i < N//2-1:
                HL_rand[i, i] += m_squared + a   # on-site + two neighbors
                HL_rand[i+1, i+1] += a
                HL_rand[i,i+1] = -a
                HL_rand[i+1, i] = -a
      
            if i == N//2-1:
                HL_rand[i,0] = -a
                HL_rand[0,i] = -a
                HL_rand[i,i] += m_squared + a
                HL_rand[0,0] += a
            if i > N//2-1:
                HL_rand[i,i] = 2.5


        HL_rand_all_A = np.zeros((N//2,N//2))
        HL_rand_all_mom = 2.5*np.eye(N//2)

        for i in range(N//2):
            for j in range(N//2):
                HL_rand_all_A[i,j] = np.random.uniform(.1,2)
                HL_rand_all_A[j,i] = HL_rand_all_A[i,j]
        D = np.zeros((N//2,N//2))
        for i in range(N//2):
            D[i,i]=sum(HL_rand_all_A[i,:]) + m_squared

        HL_rand_all_pos = D - HL_rand_all_A
        HL_rand_all = np.block([[HL_rand_all_pos,np.zeros((N//2,N//2))],
                       [np.zeros((N//2,N//2)),HL_rand_all_mom]])

        HL = HL
        """
        #Gamma_reconstructed, nu, eps_reconstructed = build_thermal_state_from_modular_hamiltonian(HL)

        #Gamma_TFD = gaussian_purification(Gamma_reconstructed)
        V = build_ring_potential(N//2, k, m_squared)
        
        Gamma_TFD = tfd_cov_ring_from_normal_modes(N//2, k, m_squared, V, beta=1, eps_omega=1e-15)

        t0 = 2


    else:
        # Parameters
        L = 4
        Lh = 3
        n_tube = 0
        g_tube = 1
        mu_A = 1
        mu_B = 1
        mu_s = 1
        t = 10

        # Build the graph
        N = 2**(Lh - 1) * (2**(L - Lh + 1) - 1)
        bdy_len = 2**(L - 1)
        bdy_1 = np.arange(N - bdy_len, N)
        N_tot = 2 * N + n_tube * 2**(Lh - 1)
        bdy_2 = np.arange(N_tot - bdy_len, N_tot)

        # Build base adjacency matrix A
        A = np.zeros((N, N), dtype=np.float64)
        for l1 in range(Lh, L + 1):
            for s1 in range(1, 2**(l1 - 1) + 1):
                for l2 in range(Lh, L + 1):
                    for s2 in range(1, 2**(l2 - 1) + 1):
                        prev1 = sum(2**(k - 1) for k in range(Lh, l1))
                        prev2 = sum(2**(k - 1) for k in range(Lh, l2))
                        ind1 = prev1 + s1 - 1
                        ind2 = prev2 + s2 - 1
                        if l1 == l2 and (abs(s1 - s2) == 1 or abs(s1 - s2) == 2**(l1 - 1) - 1):
                            A[ind1, ind2] = mu_s
                        if l2 == l1 + 1 and s2 in [2*s1, 2*s1 - 1]:
                            A[ind1, ind2] = mu_s
                        if l1 == l2 + 1 and s1 in [2*s2, 2*s2 - 1]:
                            A[ind1, ind2] = mu_s

        # Full adjacency with duplicated regions and tube
        A_tot = np.zeros((N_tot, N_tot),dtype=np.float64)
        A_tot[:N, :N] = A
        A_tot[N_tot - N:, N_tot - N:] = A
        hor_1 = np.arange(2**(Lh - 1))
        for ell in range(n_tube + 1):
            offset = N + (ell - 1) * 2**(Lh - 1)
            if ell == 0:
                for i in hor_1:
                    A_tot[i, i + N] = A_tot[i + N, i] = g_tube
            elif ell > 0:
                for i in hor_1:
                    A_tot[i + offset, i + offset + 2**(Lh - 1)] = g_tube
                    A_tot[i + offset + 2**(Lh - 1), i + offset] = g_tube
                    # Horizontal connections
                    if i < 2**(Lh - 1) - 1:
                        A_tot[i + offset, i + offset + 1] = A_tot[i + offset + 1, i + offset] = g_tube
                    else:
                        A_tot[i + offset, i + offset - (2**(Lh - 1) - 1)] = A_tot[i + offset - (2**(Lh - 1) - 1), i + offset] = g_tube
 
        # Index sets
        un_set = np.concatenate([bdy_1, bdy_2])
        meas_set = np.setdiff1d(np.arange(N_tot), un_set)


        Gamma_0 =.5 * np.eye(2*N_tot,dtype=np.complex128)




        # Number of total modes
        n = N_tot

        # Default: mass = 1, so kinetic term is identity
        M = 1* np.eye(n)
        D = np.zeros((n,n))
        for i in range(n):
            D[i,i]=sum(A_tot[i,:])

        # Potential term = adjacency + onsite mass term
        mu_squared = 0  # Choose this to control oscillator frequency
        K = D - A_tot + mu_squared * np.eye(n)

        # Construct full Hamiltonian H (2n x 2n) in (x1..xn, p1..pn) basis
        H = np.block([
            [K,         np.zeros((n, n))],
            [np.zeros((n, n)),   M     ]
        ])


        n = Gamma_0.shape[0] // 2
        Omega = symplectic_form(n)
        S_t = expm(Omega @ H * t)
        Gamma_q = S_t @ Gamma_0 @ S_t.T


        Gamma_TFD = momentum_measured_1(Gamma_q,un_set,meas_set)


        b = bdy_len
        keep = np.arange(b)  # keep left boundary
        Gamma_reduced = trace_out_subsystem(Gamma_TFD, keep)

        #HL = covmat_to_hamil(Gamma_reduced)
        HL = construct_modular_hamiltonian_with_pinning(Gamma_reduced)
        t0 = 2


    ############



    n = Gamma_TFD.shape[0] // 2
    bdy_len = Gamma_TFD.shape[0] // 4
    b = bdy_len


    HL_full = np.zeros((2*n, 2*n))
    HL_full[np.ix_(range(b), range(b))] = HL[:b, :b]                     # x-x
    HL_full[np.ix_(range(b), range(n, n + b))] = HL[:b, b:]             # x-p
    HL_full[np.ix_(range(n, n + b), range(b))] = HL[b:, :b]             # p-x
    HL_full[np.ix_(range(n, n + b), range(n, n + b))] = HL[b:, b:]      # p-p




    # Symplectic form
    Omega = symplectic_form(n)

    # Evolve backward in time

    S_back = expm(-1 * Omega @ HL_full * t0)
    Gamma_back = S_back @ Gamma_TFD @ S_back.T


    ###########
    # insert quantum information on one side
    ###########


    

    #teleported_idx = bdy_len + q # index 0 on right side starts here


    n_total = Gamma_TFD.shape[0] // 2



    Rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    Squeeze = 0.5 * np.array([[np.exp(-2*s), 0],
                          [0, np.exp(2*s)]])

    Gamma_rot_squeezed = Rot @ Squeeze @ Rot.T

    Gamma_insert = insert_unentangled_mode(Gamma_back, insert_idx, Gamma_rot_squeezed)

    Gamma_2mode = two_mode_squeezed_state(r=1)

    Gamma_with_observer = insert_two_mode_state_direct_sum(Gamma_back, insert_idx, Gamma_2mode)

    HL_full_padded = pad_matrix_for_observer(HL_full)

    #######
    # evolve forwards in time
    #######
    S_forward_no_insert = expm(Omega @ HL_full * t0)
    Gamma_forward = S_forward_no_insert @ Gamma_insert @ S_forward_no_insert.T

    n_total = (Gamma_with_observer.shape[0]) // 2  # now n+1
    Omega_padded = symplectic_form(n_total)
    S_forward_observer = expm(Omega_padded @ HL_full_padded * t0)
    Gamma_forward_observer = S_forward_observer @ Gamma_with_observer @ S_forward_observer.T

    #######
    # couple the two sides
    #######
    if coupling==True:
        t_couple = 3
        S_coupling = expm(Omega @ H_coupling * t_couple)
        Gamma_coupled = S_coupling @ Gamma_forward @ S_coupling.T

        H_coupling_padded = pad_matrix_for_observer(H_coupling)
        S_coupling_observer = expm(Omega_padded @ H_coupling_padded * t_couple)
        Gamma_coupled_observer = S_coupling_observer @ Gamma_forward_observer @ S_coupling_observer.T
    else:
        Gamma_coupled = Gamma_forward
        Gamma_coupled_observer = Gamma_forward_observer

    ######
    # evolve state forwards in time with KR
    ######


    HR_full = np.zeros((2*n, 2*n))
    HR_full[np.ix_(range(b, 2*b), range(b, 2*b))] = HL[:b, :b]
    HR_full[np.ix_(range(b, 2*b), range(n + b, n + 2*b))] = HL[:b, b:]
    HR_full[np.ix_(range(n + b, n + 2*b), range(b, 2*b))] = HL[b:, :b]
    HR_full[np.ix_(range(n + b, n + 2*b), range(n + b, n + 2*b))] = HL[b:, b:]

    HR_full_padded = pad_matrix_for_observer(HR_full)



    S_final = expm(Omega @ HR_full * t0)
    Gamma_final = S_final @ Gamma_coupled @ S_final.T

    S_final_observer = expm(Omega_padded @ HR_full_padded * t0)
    Gamma_final_observer = S_final_observer @ Gamma_coupled_observer @ S_final_observer.T

    teleported_idx = bdy_len + q # index 0 on right side starts here



    Gamma_teleported = extract_mode_block(Gamma_final, teleported_idx)


    Gamma_out_real = 0.5 * (Gamma_teleported + Gamma_teleported.conj().T)
    return Gamma_final_observer, Gamma_final, Gamma_forward_observer, Gamma_forward

def orthogonal_with_first_col(v, eps=1e-12):
    """
    Return an orthogonal matrix Q such that Q[:,0] = v (unit-norm).
    Deterministic via Householder.
    """
    v = np.asarray(v, float)
    v = v / (np.linalg.norm(v) + eps)
    m = v.size

    e1 = np.zeros(m); e1[0] = 1.0
    # If v already equals e1, Q = I
    if np.linalg.norm(v - e1) < 1e-10:
        return np.eye(m)

    # Householder that maps e1 -> v (or v -> e1; both work up to transpose)
    u = e1 - v
    u = u / (np.linalg.norm(u) + eps)
    H = np.eye(m) - 2.0 * np.outer(u, u)

    # H @ e1 = v
    return H


def sym(A): 
    return 0.5*(A + A.T)

def invsqrt_psd(M, eps=1e-10):
    M = sym(M)
    w, U = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return U @ np.diag(1/np.sqrt(w)) @ U.T

def build_passive_decoder_from_observer(V_OR_xxpp, m, eps=1e-10):
    V = sym(V_OR_xxpp)

    xO = 0
    xR = np.arange(1, m+1)
    pO = m+1
    pR = np.arange(m+2, 2*m+2)

    A = V[np.ix_([xO, pO], [xO, pO])]

    idxR = np.concatenate([xR, pR])
    B = V[np.ix_(idxR, idxR)]
    B_xx = B[:m, :m]
    B_pp = B[m:, m:]

    C = V[np.ix_([xO, pO], idxR)]
    Cx = C[:, :m]
    Cp = C[:, m:]

    Ainv = np.linalg.inv(sym(A))
    Bxx_invsqrt = invsqrt_psd(B_xx, eps=eps)
    Bpp_invsqrt = invsqrt_psd(B_pp, eps=eps)

    Mx = Bxx_invsqrt @ (Cx.T @ Ainv @ Cx) @ Bxx_invsqrt
    Mp = Bpp_invsqrt @ (Cp.T @ Ainv @ Cp) @ Bpp_invsqrt
    M = sym(Mx + Mp)

    w, U = np.linalg.eigh(M)
    u = U[:, np.argmax(w)]

    v = Bxx_invsqrt @ u
    v = v / np.linalg.norm(v)
    
    #Q, _ = np.linalg.qr(np.column_stack([v, np.random.randn(m, m-1)]))
    Q = orthogonal_with_first_col(v)
    #if np.dot(Q[:,0], v) < 0:
    #    Q[:,0] *= -1
    O = Q.T
    return O, v

def passive_decode_right_block(B_xxpp, O):
    m = O.shape[0]
    S = np.block([
        [O, np.zeros((m,m))],
        [np.zeros((m,m)), O]
    ])
    Bout = S @ sym(B_xxpp) @ S.T
    return sym(Bout)

def first_mode_from_block(B_xxpp):
    m = B_xxpp.shape[0] // 2
    x1 = 0
    p1 = m + 0
    V1 = B_xxpp[np.ix_([x1, p1], [x1, p1])]
    return sym(V1)

def extract_block_xxpp(Gamma, modes):
    n = Gamma.shape[0]//2
    x = np.array(modes)
    p = x + n
    idx = np.concatenate([x, p])
    return sym(Gamma[np.ix_(idx, idx)])

def right_segment_ids(teleported_id, n, m):
    # right ring ids are n..2n-1
    start = teleported_id - (m//2)
    start = max(start, n)
    start = min(start, 2*n - m)
    return np.arange(start, start + m)

def left_segment_ids(insert_id, n, m):
    # right ring ids are n..2n-1
    start = insert_id - (m//2)
    start = max(start, 0)
    start = min(start, n - m)
    return np.arange(start, start + m)

def right_segment_ids_centered(center_idx,n,m):
    i = m//2
    if m == 1:
        segment_telep=np.array([center_idx+n])
    elif center_idx  - i >= 0  and center_idx  + i < n:
        segment_telep = np.arange(center_idx + n - i, center_idx + n + i)
    elif center_idx - i < 0 :
        diff = np.abs(center_idx - i)
        segment_telep_1 = np.arange(2*n-diff,2*n)
        segment_telep_2 = np.arange(n ,center_idx + n + i)
        segment_telep = np.concatenate((segment_telep_1,segment_telep_2))
    elif center_idx + i >= n:
        diff = center_idx + i - n
        segment_telep_1 = np.arange(center_idx + n - i,2*n)
        segment_telep_2 = np.arange(n,n + diff)
        segment_telep = np.concatenate((segment_telep_1,segment_telep_2))
    return(segment_telep)


def left_segment_ids_centered(center_idx,n,m):
    i = m//2
    if m == 1:
        segment_telep=np.array([center_idx])
    elif center_idx  - i >= 0  and center_idx  + i < n:
        segment_telep = np.arange(center_idx - i, center_idx + i)
    elif center_idx - i < 0 :
        diff = np.abs(center_idx - i)
        segment_telep_1 = np.arange(n-diff,n)
        segment_telep_2 = np.arange(0 ,center_idx + i)
        segment_telep = np.concatenate((segment_telep_1,segment_telep_2))
    elif center_idx + i >= n:
        diff = center_idx + i - n
        segment_telep_1 = np.arange(center_idx - i,n)
        segment_telep_2 = np.arange(0,diff)
        segment_telep = np.concatenate((segment_telep_1,segment_telep_2))
    return(segment_telep)

def extract_block_xxpp_LRO(Gamma, mode_ids, Ntot):
    """
    Gamma ordering: [x_L (n), x_R (n), x_O (1), p_L (n), p_R (n), p_O (1)]
    mode_ids: list/array of mode IDs in 0..Ntot-1, where:
        left j -> j
        right j -> n + j
        obs -> 2n
    Returns block covariance in xxpp ordering for those modes:
        [x_modes..., p_modes...]
    """
    mode_ids = np.array(mode_ids, dtype=int)
    x_idx = mode_ids
    p_idx = mode_ids + Ntot
    idx = np.concatenate([x_idx, p_idx])
    return sym(Gamma[np.ix_(idx, idx)])




def build_V_OR_xxpp(Gamma_global, obs_idx, right_seg, Ntot):
    modes = np.concatenate([[obs_idx], right_seg])
    return extract_block_xxpp_LRO(Gamma_global, modes, Ntot)

def make_input_covariance(s, theta):
    Rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    Squeeze = 0.5 * np.array([[np.exp(-2*s), 0],
                              [0, np.exp( 2*s)]])
    return sym(Rot @ Squeeze @ Rot.T)

def make_rotation(M):
    """Force a 2x2 orthogonal matrix to have det=+1 (proper rotation)."""
    M = M.copy()
    if np.linalg.det(M) < 0:
        M[:, 1] *= -1
    return M


def decoder_from_X_symplectic_old(X, mode="rotation+squeeze", tol=1e-10):
    U, s, Vt = np.linalg.svd(X)
    U = make_rotation(U)
    if mode == "rotation":
        return U.T

    s1, s2 = s
    s1 = max(s1, tol)
    s2 = max(s2, tol)

    r = 0.5*np.log(s1/s2)
    S = np.diag([np.exp(-r/2), np.exp(r/2)])  # det=1

    return S @ U.T


def get_nearest_orthogonal_symplectic(M):
    """
    Forces a matrix into the nearest orthogonal symplectic matrix.
    An orthogonal symplectic matrix must satisfy the block form:
    [[A, -B], [B, A]] where A + iB is unitary.
    """
    N = M.shape[0] // 2
    # 1. Project onto the block-circulant structure (Symplectic symmetry)
    # Extract blocks
    M11, M12 = M[:N, :N], M[:N, N:]
    M21, M22 = M[N:, :N], M[N:, N:]
    
    # Average the blocks to enforce [[A, -B], [B, A]]
    A = 0.5 * (M11 + M22)
    B = 0.5 * (M21 - M12)
    
    # 2. Project onto the Unitary group (Orthogonality)
    # Form the complex matrix A + iB
    complex_mat = A + 1j * B
    # Use polar decomposition to find the nearest unitary matrix
    U_complex, _ = polar(complex_mat)
    
    # 3. Reconstruct the 2N x 2N real matrix
    O = np.block([
        [U_complex.real, -U_complex.imag],
        [U_complex.imag,  U_complex.real]
    ])
    return O


def decoder_from_X_symplectic(X):
    U, s, Vt = np.linalg.svd(X)
    O1= U.copy()
    O2 = Vt.copy()

    if det(U)<0:
        O1[:,1]*=-1

    s1, s2 = s
    D = np.diag((s2,s1))
    r = 0.5*np.log(s2/s1)
    squeeze = np.diag([np.exp(-r), np.exp(r)])

    eta = np.sqrt(s1 * s2)

    if det(U)<0:
        loss = np.diag((eta,-eta))
    else:
       loss = np.diag((eta,eta))  


    return O1 @ squeeze @ O2



def decompose_X(X):
    U, s, Vt = np.linalg.svd(X)
    O1= U.copy()
    O2 = Vt.copy()

    if det(U)<0:
        O1[:,1]*=-1

    s1, s2 = s
    D = np.diag((s2,s1))
    r = 0.5*np.log(s2/s1)
    squeeze = np.diag([np.exp(-r), np.exp(r)])

    eta = np.sqrt(s1 * s2)

    if det(U)<0:
        loss = np.diag((eta,-eta))
    else:
       loss = np.diag((eta,eta))  


    return O1, loss, squeeze, O2


def decoder_from_X_flip(X):
    U, s, Vt = np.linalg.svd(X)

    s1, s2 = s
    D = np.diag((s2,s1))
    r = 0.5*np.log(s2/s1)
    squeeze = np.diag([np.exp(-r), np.exp(r)])

    eta = np.sqrt(s1 * s2)
    loss = np.diag((s1,s2))


    return U @ squeeze @ Vt




def sym(A): 
    return 0.5*(A + A.T)


def pack_params(X, Y):
    # Y symmetric
    return np.array([X[0,0], X[0,1], X[1,0], X[1,1], Y[0,0], Y[0,1], Y[1,1]], dtype=float)

def unpack_params(p):
    a,b,c,d,y11,y12,y22 = p
    X = np.array([[a,b],[c,d]], dtype=float)
    Y = np.array([[y11,y12],[y12,y22]], dtype=float)
    return X, Y

def residuals(p, Vins, Vouts):
    X, Y = unpack_params(p)
    r = []
    for Vin, Vout in zip(Vins, Vouts):
        E = sym(Vout - (X @ Vin @ X.T + Y))
        r.extend([E[0,0], E[0,1], E[1,1]])  # 3 independent comps
    return np.array(r, dtype=float)


def fit_gaussian_channel(Vins, Vouts, X0=None, Y0=None, lam=1e-3, iters=200):
    Vins  = [sym(V) for V in Vins]
    Vouts = [sym(V) for V in Vouts]

    if X0 is None:
        X0 = np.eye(2)
    if Y0 is None:
        # crude initial Y as average difference
        Y0 = sym(np.mean([Vout - X0@Vin@X0.T for Vin,Vout in zip(Vins,Vouts)], axis=0))

    p = pack_params(X0, Y0)

    for _ in range(iters):
        r = residuals(p, Vins, Vouts)
        cost = r @ r

        # numerical Jacobian (7 params)
        J = np.zeros((len(r), len(p)))
        eps = 1e-6
        for j in range(len(p)):
            dp = np.zeros_like(p); dp[j] = eps
            r2 = residuals(p + dp, Vins, Vouts)
            J[:,j] = (r2 - r) / eps

        # LM step: (J^T J + lam I) delta = J^T r
        A = J.T @ J + lam*np.eye(len(p))
        g = J.T @ r
        delta = np.linalg.solve(A, g)

        p_new = p - delta
        r_new = residuals(p_new, Vins, Vouts)
        cost_new = r_new @ r_new

        # accept/reject, update damping
        if cost_new < cost:
            p = p_new
            lam *= 0.7
        else:
            lam *= 2.0

        if np.linalg.norm(delta) < 1e-10:
            break

    X, Y = unpack_params(p)
    return X, sym(Y)

def reorder_to_block_form(Gamma):
    """
    Reorders 2-mode covariance matrix from [x0,p0,x1,p1] to [x0,x1,p0,p1]
    """
    perm = [0, 2, 1, 3]
    return Gamma[np.ix_(perm, perm)]

def tmsv_cov(r):
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

def fidelity_stable(V1, V2):
    V1 = 0.5*(V1 + V1.T)
    V2 = 0.5*(V2 + V2.T)
    n = V1.shape[0] // 2
    omega = symplectic_form(n)

    Vsum = V1 + V2
    V_aux = omega.T @ np.linalg.inv(Vsum) @ (0.25 * omega + V2 @ omega @ V1)

    I = np.eye(2*n)
    A = V_aux @ omega

    # A^{-2} = solve(A, solve(A, I))
    Ainv2 = np.linalg.solve(A, np.linalg.solve(A, I))
    inside = I + 0.25 * Ainv2

    F_tot4 = np.linalg.det(2 * (sqrtm(inside) + I) @ V_aux)
    F_tot = np.real_if_close(F_tot4)**0.25
    F0 = F_tot / (np.linalg.det(Vsum)**0.25)

    return float(np.real(F0))


def decode_on_B_xxpp(V_RB_xxpp, S_dec,Y,subtract_Y):
    I2 = np.eye(2)
    # xxpp ordering: (xR, xB, pR, pB)
    # decoding acts on (xB,pB) => indices [1,3], not contiguous.
    V = 0.5*(V_RB_xxpp + V_RB_xxpp.T)
    idx_R = [0, 2]
    idx_B = [1, 3]

    Vout = V.copy()

    # transform blocks: B -> S_dec B S_dec^T, C -> C S_dec^T
    A = V[np.ix_(idx_R, idx_R)]
    B = V[np.ix_(idx_B, idx_B)]
    C = V[np.ix_(idx_R, idx_B)]

    if subtract_Y == True:
        B-=Y

    B2 = S_dec @ B @ S_dec.T
    C2 = C @ S_dec.T

    Vout[np.ix_(idx_R, idx_R)] = A
    Vout[np.ix_(idx_B, idx_B)] = B2
    Vout[np.ix_(idx_R, idx_B)] = C2
    Vout[np.ix_(idx_B, idx_R)] = C2.T

    return 0.5*(Vout + Vout.T)


def entanglement_fidelity_gaussian(X, Y, S,subtract_Y,r=1.0):
    V0 = tmsv_cov(r)
    V1 = apply_channel_to_second_mode_xxpp(V0, X, Y)
    V1_dec = decode_on_B_xxpp(V1,inv(S),Y,subtract_Y)
    # zero means:
    #mu0 = np.zeros(4)
    #mu1 = np.zeros(4)  
    return fidelity_stable(V0,V1_dec)

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

def apply_channel_to_second_mode_xxpp(V_RB_xxpp, X, Y):
    """
    Apply a 1-mode Gaussian channel (X,Y) to mode B of a 2-mode covariance matrix
    given in xxpp ordering: (xR, xB, pR, pB).

    V_RB_xxpp: 4x4 covariance in order [xR, xB, pR, pB]
    X, Y: 2x2 with respect to (xB, pB)
    """
    V = sym(V_RB_xxpp)
    X = np.asarray(X, float)
    Y = sym(np.asarray(Y, float))

    # Indices for the R and B modes in xxpp ordering
    idx_R = [0, 2]  # (xR, pR)
    idx_B = [1, 3]  # (xB, pB)

    # Extract 2x2 blocks in (x,p) ordering for each mode
    A = V[np.ix_(idx_R, idx_R)]   # Cov of R
    B = V[np.ix_(idx_B, idx_B)]   # Cov of B
    C = V[np.ix_(idx_R, idx_B)]   # Cross-cov R-B

    # Transform blocks under channel on B
    A_out = A
    C_out = C @ X.T
    B_out = X @ B @ X.T + Y

    # Reassemble full 4x4 in xxpp ordering
    V_out = V.copy()
    V_out[np.ix_(idx_R, idx_R)] = A_out
    V_out[np.ix_(idx_R, idx_B)] = C_out
    V_out[np.ix_(idx_B, idx_R)] = C_out.T
    V_out[np.ix_(idx_B, idx_B)] = B_out

    return sym(V_out)

def sym(A): return 0.5*(A + A.T)

def noise_metrics(X, Y):
    Y = sym(Y)
    detX = np.linalg.det(X)
    y_eff = 0.5*np.trace(Y)  # average added noise
    y_det = np.sqrt(max(np.linalg.det(Y), 0.0))
    y_iso_min = abs(1 - detX)/2  # phase-insensitive quantum-limited scale
    ratio = y_eff / (y_iso_min + 1e-12)
    return detX, y_eff, y_det, y_iso_min, ratio


def build_rankK_coupling_LRO(
    N_boundary,        # n in your L/R (without observer): left has n, right has n
    left_seg,          # length m, values in [0..n-1]
    right_seg,         # length m, values in [n..2n-1]  (GLOBAL mode ids in LRO convention)
    O_L,               # K x m
    O_R,               # K x m
    g=None,            # None or length-K array of coupling strengths
    include_observer=False
):
    """
    Returns H_coup for ordering:
      [x_L(n), x_R(n), x_O, p_L(n), p_R(n), p_O]  if include_observer
      [x_L(n), x_R(n),       p_L(n), p_R(n)]      if not include_observer

    Notes:
      - left_seg must be left mode IDs (0..n-1)
      - right_seg must be right mode IDs (n..2n-1)
      - O_L, O_R are K×m weights defining collective modes on those segments
    """
    n = N_boundary
    m = len(left_seg)
    assert len(right_seg) == m
    K = O_L.shape[0]
    assert O_L.shape == (K, m)
    assert O_R.shape == (K, m)

    if g is None:
        g = np.ones(K)
    g = np.asarray(g, float)
    assert g.shape == (K,)

    G = np.diag(g)  # K×K
    # physical segment coupling J = O_L^T G O_R  (m×m)
    J = O_L.T @ G @ O_R

    if include_observer:
        Ntot = 2*n + 1   # total modes including observer
        dim = 2*Ntot
        obs = 2*n
    else:
        Ntot = 2*n
        dim = 2*Ntot

    H = np.zeros((dim, dim), dtype=float)

    # ---- x-x block coupling between physical modes in left_seg and right_seg ----
    # In LRO ordering, x indices are just mode ids themselves.
    xL = np.array(left_seg, dtype=int)
    xR = np.array(right_seg, dtype=int)

    # Place 1/2 * J into H[xL, xR] using segment-local indexing
    # We need to map segment-local (0..m-1) pairs to global indices.
    for a in range(m):
        for b in range(m):
            H[xL[a], xR[b]] += 0.5 * J[a, b]
            H[xR[b], xL[a]] += 0.5 * J[a, b]  # symmetric (since we used same J)

    # ---- p-p block coupling ----
    # p index = mode_id + Ntot
    pL = xL + Ntot
    pR = xR + Ntot
    for a in range(m):
        for b in range(m):
            H[pL[a], pR[b]] += 0.5 * J[a, b]
            H[pR[b], pL[a]] += 0.5 * J[a, b]

    # Symmetrize to be safe
    H = 0.5 * (H + H.T)
    return H


def build_coupling_LRO(
    N_boundary,        # n in your L/R (without observer): left has n, right has n
    left_seg,          # length m, values in [0..n-1]
    right_seg,         # length m, values in [n..2n-1]  (GLOBAL mode ids in LRO convention)
    O_L,               # K x m
    O_R,               # K x m
    g=None,            # None or length-K array of coupling strengths
    include_observer=False
):
    """
    Returns H_coup for ordering:
      [x_L(n), x_R(n), x_O, p_L(n), p_R(n), p_O]  if include_observer
      [x_L(n), x_R(n),       p_L(n), p_R(n)]      if not include_observer

    Notes:
      - left_seg must be left mode IDs (0..n-1)
      - right_seg must be right mode IDs (n..2n-1)
      - O_L, O_R are K×m weights defining collective modes on those segments
    """
    n = N_boundary
    m = len(left_seg)
    assert len(right_seg) == m
    KO = O_L.shape[0]
    if g is None:
        g = np.ones(KO)
    g = np.asarray(np.ones(KO), float)
    
    G = np.diag(g)  # K×K
    # physical segment coupling J = O_L^T G O_R  (m×m)
    J = O_L.T @ G @ O_R

    if include_observer:
        Ntot = 2*n + 1   # total modes including observer
        dim = 2*Ntot
        obs = 2*n
    else:
        Ntot = 2*n
        dim = 2*Ntot

    H = np.zeros((dim, dim), dtype=float)

    # ---- x-x block coupling between physical modes in left_seg and right_seg ----
    # In LRO ordering, x indices are just mode ids themselves.
    xL = np.array(left_seg, dtype=int)
    xR = np.array(right_seg, dtype=int)

    # Place 1/2 * J into H[xL, xR] using segment-local indexing
    # We need to map segment-local (0..m-1) pairs to global indices.
    for a in range(m):
        for b in range(m):
            H[xL[a], xR[b]] += 0.5 * J[a, b]
            H[xR[b], xL[a]] += 0.5 * J[a, b]  # symmetric (since we used same J)

    # ---- p-p block coupling ----
    # p index = mode_id + Ntot
    pL = xL + Ntot
    pR = xR + Ntot
    for a in range(m):
        for b in range(m):
            H[pL[a], pR[b]] += 0.5 * J[a, b]
            H[pR[b], pL[a]] += 0.5 * J[a, b]

    # Symmetrize to be safe
    H = 0.5 * (H + H.T)
    return H

def right_segment_ids(teleported_id, n, m):
    # right ring ids are n..2n-1
    start = teleported_id - (m//2)
    start = max(start, n)
    start = min(start, 2*n - m)
    return np.arange(start, start + m)

def left_segment_ids(insert_id, n, m):
    # right ring ids are n..2n-1
    start = insert_id - (m//2)
    start = max(start, 0)
    start = min(start, n - m)
    return np.arange(start, start + m)

def X_metrics(X):
    s = np.linalg.svd(X, compute_uv=False)     # singular values
    s = np.sort(s)[::-1]
    spec = s[0]
    fro  = np.linalg.norm(X, 'fro')
    det  = abs(np.linalg.det(X))
    return {"s1": s[0], "s2": s[1], "spec": spec, "fro": fro, "det": det}



############
# Begin Active Decoder
############
import numpy as np

def sym(A): 
    return 0.5*(A + A.T)

def Omega_xxpp(m):
    I = np.eye(m); Z = np.zeros((m,m))
    return np.block([[Z, I], [-I, Z]])

def block_diag_2(A, B):
    return np.block([[A, np.zeros((A.shape[0], B.shape[1]))],
                     [np.zeros((B.shape[0], A.shape[1])), B]])

def mode_permutation_symplectic_xxpp_old(m, perm):
    """
    perm is a permutation of modes [0..m-1] applied to both x and p parts.
    Returns 2m×2m symplectic permutation matrix P in xxpp ordering.
    """
    perm = np.asarray(perm, dtype=int)
    Px = np.eye(m)[:, perm]
    Pp = np.eye(m)[:, perm]
    return np.block([[Px, np.zeros((m,m))],
                     [np.zeros((m,m)), Pp]])

def partition_V_OR_xxpp(V_OR_xxpp, m):
    """
    V_OR_xxpp is 2(1+m)×2(1+m) in ordering:
      [xO, xR1..xRm, pO, pR1..pRm]
    Returns A (2×2), B (2m×2m), C (2×2m).
    """
    V = sym(V_OR_xxpp)
    xO = 0
    xR = np.arange(1, m+1)
    pO = m+1
    pR = np.arange(m+2, 2*m+2)

    idxO = np.array([xO, pO])
    idxR = np.concatenate([xR, pR])

    A = V[np.ix_(idxO, idxO)]
    B = V[np.ix_(idxR, idxR)]
    C = V[np.ix_(idxO, idxR)]
    return sym(A), sym(B), C

def conditional_covariance(B, C, A, eps=1e-12):
    """
    V_{R|O} = B - C^T A^{-1} C, stabilized inverse for A.
    """
    w, U = np.linalg.eigh(sym(A))
    w = np.clip(w, eps, None)
    Ainv = U @ np.diag(1.0/w) @ U.T
    return sym(B - C.T @ Ainv @ C)

def williamson_S_from_strawberry(V):
    """
    Your williamson_strawberry returns SinvT, Db, nus.
    Recover the symplectic S such that V = S^T Db S.
    """
    SinvT, Db, nus = williamson_strawberry(sym(V))
    S = np.linalg.inv(SinvT).T
    return S, Db, nus

def active_decoder_williamson_from_VOR(V_OR_xxpp, m):
    """
    Build an active (symplectic) decoder on the right block using Williamson on V_{R|O}.

    Returns:
      S_dec (2m×2m) : symplectic acting on right block
      perm (m,)     : mode permutation applied so mode 0 is best
      nus_c (m,)    : conditional symplectic eigenvalues (unsorted)
      Vc            : conditional covariance V_{R|O}
    """
    A, B, C = partition_V_OR_xxpp(V_OR_xxpp, m)
    Vc = conditional_covariance(B, C, A)

    S_cT, Db_c, nus_c = williamson_strawberry(sym(Vc))
    S_c = S_cT.T

    # sort modes so smallest conditional nu is first
    perm = np.argsort(nus_c)
    P = mode_permutation_symplectic_xxpp(m, perm)

    # decoder acts on R: R' = (P S_c) R
    S_dec = P @ S_c

    # sanity check symplecticity
    Om = Omega_xxpp(m)
    err = np.linalg.norm(S_dec @ Om @ S_dec.T - Om)
    if err > 1e-7:
        raise ValueError(f"S_dec not symplectic: ||SΩS^T-Ω||={err:.2e}")

    return S_dec, perm, nus_c, Vc


import numpy as np

def sym(A): return 0.5*(A + A.T)

def Omega_xxpp(m):
    Z = np.zeros((m,m))
    I = np.eye(m)
    return np.block([[Z, I],[-I, Z]])

def clip_symplectic_squeezing_from_williamson_old(S, rmax=1.2, eps=1e-12):
    """
    Minimal 'gain cap': project S onto something with bounded singular values.
    Not perfect mathematically, but very effective numerically for preventing Y blow-up.
    """
    # SVD of S (not symplectic-canonical, but bounds Euclidean gain)
    U, s, Vt = np.linalg.svd(S)
    s = np.clip(s, np.exp(-rmax), np.exp(rmax))
    return U @ np.diag(s) @ Vt

def active_decoder_williamson_from_VOR_SNR(
    V_OR_xxpp,
    m,
    lam=1e-4,
    ridge=1e-8,
    rmax=None,
):
    """
    Active decoder built from conditional Williamson, but with variance-penalized mode choice.

    - Still computes Vc = V_{R|O}
    - Still does Williamson: Vc = S_c^T D S_c
    - BUT: choose which decoded mode is "first" using an SNR objective instead of min(nu)
    - Optional: cap squeezing of S_dec via rmax to prevent huge gain

    Returns:
      S_dec (2m×2m), perm, nus_c, Vc
    """
    A, B, C = partition_V_OR_xxpp(V_OR_xxpp, m)      # you already have this
    Vc = conditional_covariance(B, C, A)             # you already have this
    Vc = sym(Vc)

    # Williamson on conditional covariance
    S_cT, Db_c, nus_c = williamson_strawberry(Vc)
    S_c = S_cT.T

    # ---- SNR-based mode scoring in the Williamson basis ----
    # Transform conditional block into Williamson coordinates:
    # R_w = S_c R_phys  (since Vc = S_c^T D S_c)
    # Cov in w-basis is D = Db_c.
    # We also need observer correlations expressed in w-basis.
    # The conditional covariance formula used Vc = B - C^T A^{-1} C;
    # correlations of O with R are still in C. We map C into w-basis:
    # C_w = C * S_c^{-1,T}  if you treat cov blocks carefully.
    #
    # In xxpp, C is (2 × 2m): rows [xO,pO], cols [xR..., pR...].
    # If R_w = S_c R_phys, then R_phys = S_c^{-1} R_w, so
    # Cov(O, R_w) = Cov(O, R_phys) (S_c^{-1})^T = C @ (S_c^{-1})^T
    SinvT = np.linalg.inv(S_c).T
    C_w = C @ SinvT                                # shape (2, 2m)

    # Split x/p parts in the w-basis (still xxpp layout)
    Cx_w = C_w[:, :m]                               # (2×m)
    Cp_w = C_w[:, m:]                               # (2×m)

    # Build a simple SNR score per Williamson mode k:
    # signal_k = ||Cov(O, x_k)||^2 + ||Cov(O, p_k)||^2
    # var_k = D_xx(k,k) + D_pp(k,k) (+ lam)
    Dx = np.diag(Db_c)[:m]                          # nu_k
    Dp = np.diag(Db_c)[m:]                          # nu_k again
    var = Dx + Dp + lam

    signal = np.sum(Cx_w**2, axis=0) + np.sum(Cp_w**2, axis=0)   # length m
    score = signal / var

    k_best = int(np.argmax(score))

    # Permute Williamson modes to bring k_best -> 0
    perm = np.arange(m)
    perm[0], perm[k_best] = perm[k_best], perm[0]
    P = mode_permutation_symplectic_xxpp(m, perm)

    S_dec = P @ S_c

    # Optional: cap gain (prevents huge Y for large m)
    if rmax is not None:
        S_dec = clip_symplectic_squeezing_from_williamson(S_dec, rmax=rmax)

    # Symplectic sanity check (only valid if we did not clip via SVD)
    if rmax is None:
        Om = Omega_xxpp(m)
        err = np.linalg.norm(S_dec @ Om @ S_dec.T - Om)
        if err > 1e-7:
            raise ValueError(f"S_dec not symplectic: ||SΩS^T-Ω||={err:.2e}")

    return S_dec, perm, nus_c, Vc, score

def apply_right_decoder_to_VOR(V_OR_xxpp, S_dec):
    """
    Apply (I_2 ⊕ S_dec) to the OR covariance (xxpp ordering).
    """
    m = S_dec.shape[0] // 2
    S_tot = block_diag_2(np.eye(2), S_dec)
    Vp = S_tot @ sym(V_OR_xxpp) @ S_tot.T
    return sym(Vp)

def first_decoded_right_mode_block(V_OR_dec_xxpp, m):
    """
    After decoding, extract the 2×2 (x,p) block of the *first right mode*.
    In OR xxpp ordering, the first right mode has:
      x index = 1
      p index = m+2
    """
    x1 = 1
    p1 = m + 2
    return sym(V_OR_dec_xxpp[np.ix_([x1, p1], [x1, p1])])

def decode_block_first_mode(B_xxpp, S_dec):
    """
    Apply symplectic decoder S_dec (ROW map) to block covariance B_xxpp:
      V_dec = S_dec B S_dec^T
    Return first decoded mode covariance (2×2) in (x,p) ordering.
    For xxpp ordering: x1 index=0, p1 index=m.
    """
    B = sym(B_xxpp)
    Vdec = sym(S_dec @ B @ S_dec.T)
    m = B.shape[0] // 2
    idx = [0, m]  # x1, p1
    return sym(Vdec[np.ix_(idx, idx)]), Vdec


############
#End Active Decoder
############


###########
# Begin Active + Passive Decoder
###########

import numpy as np

def sym(A): 
    return 0.5*(A + A.T)

def Omega_xxpp(m):
    Z = np.zeros((m, m))
    I = np.eye(m)
    return np.block([[Z, I], [-I, Z]])

def mode_permutation_symplectic_xxpp(m, perm):
    """
    perm: permutation of modes [0..m-1] applied to both x and p in xxpp ordering.
    Returns 2m×2m symplectic permutation P.
    """
    perm = np.asarray(perm, dtype=int)
    Px = np.eye(m)[:, perm]
    Pp = np.eye(m)[:, perm]
    return np.block([[Px, np.zeros((m, m))],
                     [np.zeros((m, m)), Pp]])

def clip_symplectic_squeezing_from_williamson(S, rmax=1.2):
    """
    Gain cap (Euclidean): clip singular values to [exp(-rmax), exp(+rmax)].
    NOTE: clipping breaks exact symplecticity; use only to stabilize fits.
    """
    U, s, Vt = np.linalg.svd(S)
    s = np.clip(s, np.exp(-rmax), np.exp(rmax))
    return U @ np.diag(s) @ Vt

def _normalize(v, eps=1e-12):
    nrm = np.linalg.norm(v)
    if nrm < eps:
        return v*0.0
    return v / nrm

def _topk_indices(score, K):
    K = int(min(max(K, 1), len(score)))
    return np.argsort(score)[::-1][:K]

def _embed_prev_w_to_m(prev_w, prev_m, m):
    """
    Embed previous length-prev_m weight vector into length-m by zero-padding.
    Assumes your segments are nested/centered so the previous decoded mode
    should live mostly in the old support.
    """
    out = np.zeros(m, dtype=float)
    L = min(prev_m, m)
    out[:L] = prev_w[:L]
    return out

def _decoded_firstmode_weights_in_phys(S_c, perm, m):
    """
    For a symplectic S_dec = P @ S_c acting on the right block,
    the decoded quadratures are:
        x' = (S_dec_xx) x + (S_dec_xp) p
        p' = (S_dec_px) x + (S_dec_pp) p
    In xxpp ordering, x part is first m, p part is last m.
    Return the weight vectors (length m) mapping phys x->x'_0 and phys p->p'_0.
    """
    P = mode_permutation_symplectic_xxpp(m, perm)
    S_dec = P @ S_c
    Sxx = S_dec[:m, :m]
    Sxp = S_dec[:m, m:]
    Spx = S_dec[m:, :m]
    Spp = S_dec[m:, m:]
    # first decoded mode (row 0 for x'_0, row 0 for p'_0)
    wx = Sxx[0, :].copy()
    wp = Spp[0, :].copy()
    # (If you want to include x<-p mixing too, you can also look at Sxp, Spx.
    # For overlap tracking, wx/wp are usually enough.)
    return wx, wp, S_dec

def _overlap_score(wx, wp, prev_wx, prev_wp):
    wx = _normalize(wx); wp = _normalize(wp)
    prev_wx = _normalize(prev_wx); prev_wp = _normalize(prev_wp)
    # sign-invariant overlap
    ox = abs(np.dot(wx, prev_wx))
    op = abs(np.dot(wp, prev_wp))
    return 0.5*(ox + op)

def active_decoder_williamson_tracked(
    V_OR_xxpp,
    m,
    lam=1e-4,
    ridge=1e-8,
    rmax=1.2,
    Kcand=6,
    prev_state=None,
    prefer_score_weight=0.3,
):
    """
    Stable active decoder sequence with:
      1) candidate set of modes (top-K by SNR score in Williamson basis),
      2) overlap-based tracking against previous chosen decoded mode,
      3) gain cap via rmax (optional; set None to skip).

    Inputs:
      V_OR_xxpp : 2(1+m)×2(1+m) covariance in [xO, xR(1..m), pO, pR(1..m)] ordering
      m         : block size
      lam       : variance regularizer in score
      ridge     : numerical ridge for inversions (inside your conditional_covariance / etc.)
      rmax      : cap on singular values of S_dec (set None to skip)
      Kcand     : number of candidate modes to consider per m
      prev_state: dict returned from previous call; if None, picks best-by-score
      prefer_score_weight: in [0,1], tradeoff between overlap tracking and raw score.
                           0 => pure overlap, 1 => pure score.

    Returns:
      S_dec   : 2m×2m decoder on right block (xxpp)
      info    : dict with details, including updated prev_state for next m
    """

    # --- partition and conditional covariance ---
    A, B, C = partition_V_OR_xxpp(V_OR_xxpp, m)
    Vc = conditional_covariance(B, C, A)   # uses your stabilized inverse
    Vc = sym(Vc)

    # --- Williamson on conditional covariance ---
    # you have: S_cT, Db_c, nus_c = williamson_strawberry(Vc)
    S_cT, Db_c, nus_c = williamson_strawberry(Vc)
    S_c = S_cT.T  # so Vc = S_c^T Db_c S_c

    # --- map observer correlations into Williamson basis ---
    SinvT = np.linalg.inv(S_c).T
    C_w = C @ SinvT                # (2 × 2m)
    Cx_w = C_w[:, :m]
    Cp_w = C_w[:, m:]

    Dx = np.diag(Db_c)[:m]
    Dp = np.diag(Db_c)[m:]
    var = Dx + Dp + lam

    signal = np.sum(Cx_w**2, axis=0) + np.sum(Cp_w**2, axis=0)
    score = signal / (var + 1e-18)

    # --- candidate set: top-K by score ---
    cand = _topk_indices(score, Kcand)

    # --- choose among candidates using overlap tracking ---
    chosen_k = int(cand[0])
    best_obj = -np.inf
    best_perm = None
    best_S_dec = None
    best_wx = None
    best_wp = None
    best_overlap = None

    if prev_state is None:
        # no tracking: just choose best-by-score
        chosen_k = int(np.argmax(score))
        perm = np.arange(m)
        perm[0], perm[chosen_k] = perm[chosen_k], perm[0]
        P = mode_permutation_symplectic_xxpp(m, perm)
        S_dec = P @ S_c
        wx, wp, _ = _decoded_firstmode_weights_in_phys(S_c, perm, m)
        # optional gain cap
        if rmax is not None:
            S_dec = clip_symplectic_squeezing_from_williamson(S_dec, rmax=rmax)
        # package info
        new_state = {
            "m": m,
            "wx": _normalize(wx),
            "wp": _normalize(wp),
        }
        info = {
            "chosen_k": chosen_k,
            "candidates": cand,
            "score": score,
            "score_chosen": float(score[chosen_k]),
            "overlap_chosen": None,
            "perm": perm,
            "nus_c": nus_c,
            "Vc": Vc,
            "prev_state": new_state,
        }
        return S_dec, info

    # tracking case
    prev_m = int(prev_state["m"])
    prev_wx = np.asarray(prev_state["wx"], float)
    prev_wp = np.asarray(prev_state["wp"], float)

    # embed previous weights to current length if needed (assumes nested/consistent ordering)
    prev_wx_m = _embed_prev_w_to_m(prev_wx, prev_m, m)
    prev_wp_m = _embed_prev_w_to_m(prev_wp, prev_m, m)

    # evaluate each candidate k
    for k in cand:
        k = int(k)
        perm = np.arange(m)
        perm[0], perm[k] = perm[k], perm[0]

        wx, wp, S_dec_uncapped = _decoded_firstmode_weights_in_phys(S_c, perm, m)
        ov = _overlap_score(wx, wp, prev_wx_m, prev_wp_m)

        # combine overlap + (normalized) score
        # normalize score to [0,1] over candidates to avoid huge scale dependence
        sc = float(score[k])
        sc_norm = (sc - float(np.min(score[cand]))) / (float(np.ptp(score[cand])) + 1e-18)

        obj = (1.0 - prefer_score_weight)*ov + prefer_score_weight*sc_norm

        if obj > best_obj:
            best_obj = obj
            chosen_k = k
            best_perm = perm
            best_overlap = ov
            best_wx = wx.copy()
            best_wp = wp.copy()
            best_S_dec = S_dec_uncapped.copy()

    S_dec = best_S_dec
    if rmax is not None:
        S_dec = clip_symplectic_squeezing_from_williamson(S_dec, rmax=rmax)

    # optional symplectic sanity check only if not capped
    if rmax is None:
        Om = Omega_xxpp(m)
        err = np.linalg.norm(S_dec @ Om @ S_dec.T - Om)
        if err > 1e-6:
            raise ValueError(f"S_dec not symplectic: ||SΩS^T-Ω||={err:.2e}")

    new_state = {
        "m": m,
        "wx": _normalize(best_wx),
        "wp": _normalize(best_wp),
    }

    info = {
        "chosen_k": int(chosen_k),
        "candidates": cand,
        "score": score,
        "score_chosen": float(score[chosen_k]),
        "overlap_chosen": float(best_overlap),
        "perm": best_perm,
        "nus_c": nus_c,
        "Vc": Vc,
        "prev_state": new_state,
        "objective": float(best_obj),
    }
    return S_dec, info


# --------- Convenience: build a stable sequence over block_sizes ---------

def build_stable_Sdec_sequence(
    VOR_builder,          # callable: VOR_builder(m) -> V_OR_xxpp for that m
    block_sizes,
    lam=1e-4,
    ridge=1e-8,
    rmax=1.2,
    Kcand=6,
    prefer_score_weight=0.3,
):
    """
    Returns:
      S_list   : list of 2m×2m decoders (aligned with block_sizes)
      info_list: list of info dicts (contains tracking diagnostics)
    """
    prev = None
    S_list = []
    info_list = []
    for m in block_sizes:
        V_OR_xxpp = VOR_builder(m)
        S_dec, info = active_decoder_williamson_tracked(
            V_OR_xxpp, m,
            lam=lam, ridge=ridge, rmax=rmax,
            Kcand=Kcand,
            prev_state=prev,
            prefer_score_weight=prefer_score_weight,
        )
        prev = info["prev_state"]
        S_list.append(S_dec)
        info_list.append(info)
    return S_list, info_list

###########
# End Active + Passive Decoder
###########

########
# Begin Test
########
import numpy as np

def sym(A): return 0.5*(A + A.T)


def build_collective_picker_symplectic(w):
    """
    Build S_dec (2m×2m) in xxpp ordering that makes first decoded mode
    be x_c = w·x, p_c = w·p.
    """
    m = len(w)
    O = orthogonal_with_first_row(w)
    S_dec = np.block([[O, np.zeros((m,m))],
                      [np.zeros((m,m)), O]])
    return S_dec

def extract_block_xxpp_idx(Gamma, modes):
    n = Gamma.shape[0]//2
    modes = np.asarray(modes, int)
    idx = np.concatenate([modes, modes+n])
    return sym(Gamma[np.ix_(idx, idx)]), idx

def insert_collective_mode_into_segment(Gamma, segment_modes, V_ins_2x2, w, decouple=True):
    """
    Gamma: full covariance in (x_all, p_all) ordering.
    segment_modes: list of m physical mode indices (in the full system) you target.
    V_ins_2x2: desired covariance of the collective mode (x_c, p_c).
    w: weights over the m modes (if None, uniform).
    decouple: if True, zero correlations between inserted collective mode and the rest
              within the segment (in decoded basis).

    Returns: Gamma_new
    """
    Gamma = sym(Gamma)
    V_ins_2x2 = sym(V_ins_2x2)

    m = len(segment_modes)
    if w is None:
        w = np.ones(m)/np.sqrt(m)
    else:
        w = np.asarray(w, float)
        w = w/np.linalg.norm(w)

    # 1) Extract segment covariance B (xxpp for the segment)
    B, full_idx = extract_block_xxpp_idx(Gamma, segment_modes)  # 2m×2m


    # 2) Decode to collective-mode basis
    S_dec = build_collective_picker_symplectic(w)
    B_dec = sym(S_dec @ B @ S_dec.T)

    

    # 3) Overwrite first mode block (x1,p1) in decoded basis
    # In xxpp, first mode x index=0, p index=m
    idx_mode1 = [0, m]
    if decouple:
        B_dec[idx_mode1, :] = 0.0
        B_dec[:, idx_mode1] = 0.0
    B_dec[np.ix_(idx_mode1, idx_mode1)] = V_ins_2x2



    # 4) Transform back
    S_inv = np.linalg.inv(S_dec)
    B_new = sym(S_inv @ B_dec @ S_inv.T)

    # 5) Put back into the full Gamma (only replacing the segment block)
    Gamma_new = Gamma.copy()
    Gamma_new[np.ix_(full_idx, full_idx)] = B_new
    return sym(Gamma_new)

import numpy as np



def orthogonal_with_first_row(w, eps=1e-12):
    """
    Returns O (m×m) orthogonal with O[0,:] = w^T (unit norm).
    Deterministic via Householder.
    """
    w = np.asarray(w, float)
    w = w / (np.linalg.norm(w) + eps)
    m = w.size

    e1 = np.zeros(m); e1[0] = 1.0
    if np.linalg.norm(w - e1) < 1e-10:
        return np.eye(m)

    # Householder mapping e1 -> w
    u = e1 - w
    u = u / (np.linalg.norm(u) + eps)
    H = np.eye(m) - 2.0 * np.outer(u, u)
    # H @ e1 = w, so first column is w; we want first ROW = w^T.
    # Take transpose so first row is w^T.
    return H.T



def sym(A): return 0.5*(A + A.T)

def add_observer_LRO(Gamma_LR, V_obs):
    """
    Gamma_LR: (4n×4n) in ordering [xL(n), xR(n), pL(n), pR(n)]
    Returns: (2Ntot×2Ntot) in ordering [xL,xR,xO,pL,pR,pO], Ntot=2n+1
    with observer initially uncorrelated from system.
    """
    Gamma_LR = sym(Gamma_LR)
    n2 = Gamma_LR.shape[0] // 2            # = 2n
    n = n2 // 2                             # one-side size
    Ntot = 2*n + 1
    dim = 2*Ntot

    if V_obs is None:
        V_obs = 0.5*np.eye(2)
    V_obs = sym(V_obs)

    G = np.zeros((dim, dim))
    # System part mapping:
    # Gamma_LR x-block is size 2n: [xL,xR]
    # Gamma_LR p-block is size 2n: [pL,pR]
    # Place xL,xR into [0:2n] and pL,pR into [Ntot:Ntot+2n]
    G[0:2*n, 0:2*n] = Gamma_LR[0:2*n, 0:2*n]
    G[0:2*n, Ntot:Ntot+2*n] = Gamma_LR[0:2*n, 2*n:4*n]
    G[Ntot:Ntot+2*n, 0:2*n] = Gamma_LR[2*n:4*n, 0:2*n]
    G[Ntot:Ntot+2*n, Ntot:Ntot+2*n] = Gamma_LR[2*n:4*n, 2*n:4*n]

    # Observer indices
    xO = 2*n
    pO = Ntot + 2*n
    G[np.ix_([xO,pO],[xO,pO])] = V_obs

    return sym(G)



def insert_RB_collective_into_right_segment_LRO(
    Gamma_LR,              # (4n×4n) ordering [xL,xR,pL,pR] (no observer)
    right_seg,             # length m, GLOBAL mode ids in right: n..2n-1
    V_RB_xxpp,             # 4×4 in order [xR, xB, pR, pB] (R=observer, B=collective)
    w,                # length m weights, ||w||=1; default uniform
    decouple_within_segment=True,
    decouple_obs_from_rest=True,
):
    """
    Returns Gamma_full in ordering [xL,xR,xO,pL,pR,pO].
    The injected correlations guarantee that (observer, collective decoded mode) marginal = V_RB_xxpp,
    where collective mode is w^T over the chosen right_seg.
    """
    V_RB_xxpp = sym(V_RB_xxpp)

    # Extract target blocks in (x,p) per mode:
    # indices in xxpp: [xR, xB, pR, pB]
    V_RR = V_RB_xxpp[np.ix_([0,2],[0,2])]   # observer self (xR,pR)
    V_BB = V_RB_xxpp[np.ix_([1,3],[1,3])]   # collective self (xB,pB)
    V_RB = V_RB_xxpp[np.ix_([0,2],[1,3])]   # (xR,pR) vs (xB,pB)

    # Add observer uncorrelated first, but set its self-cov to V_RR
    Gamma_full = add_observer_LRO(Gamma_LR, V_obs=V_RR)
    Gamma_full = sym(Gamma_full)

    # Sizes / indices under LRO
    # one-side size n inferred from Gamma_LR: (4n×4n)
    n2 = Gamma_LR.shape[0]//2   # = 2n
    n  = n2//2
    Ntot = 2*n + 1

    xO = 2*n
    pO = Ntot + 2*n

    right_seg = np.asarray(right_seg, int)
    m = len(right_seg)

    if w is None:
        w = np.ones(m)/np.sqrt(m)
    else:
        w = np.asarray(w, float)
        w = w/np.linalg.norm(w)

    # Build segment indices in FULL matrix
    x_seg = right_seg
    p_seg = right_seg + Ntot
    seg_idx = np.r_[x_seg, p_seg]  # length 2m, in xxpp-style order for this subblock

    # --- Step A: enforce collective covariance V_BB on the segment ---
    # Extract current segment covariance (xxpp ordering: x_seg..., p_seg...)
    B = sym(Gamma_full[np.ix_(seg_idx, seg_idx)])

    # Passive “collective picker” on the segment: first decoded mode is w^T
    O = orthogonal_with_first_row(w)                 # first row = w^T
    S_dec = np.block([[O, np.zeros((m,m))],
                      [np.zeros((m,m)), O]])         # symplectic passive

    B_dec = sym(S_dec @ B @ S_dec.T)

    # Overwrite first decoded mode (x1,p1) to V_BB
    idx_c = [0, m]  # x1, p1 in this decoded-segment basis
    if decouple_within_segment:
        B_dec[idx_c, :] = 0.0
        B_dec[:, idx_c] = 0.0
    B_dec[np.ix_(idx_c, idx_c)] = V_BB

    # Back to physical segment basis
    S_inv = np.linalg.inv(S_dec)
    B_new = sym(S_inv @ B_dec @ S_inv.T)
    Gamma_full[np.ix_(seg_idx, seg_idx)] = B_new

    # --- Step B: set observer <-> segment correlations to match V_RB for the collective mode ---
    # V_RB is 2×2: rows [xO,pO], cols [x_c,p_c]
    cov_O_xc = V_RB[:, 0]  # shape (2,)
    cov_O_pc = V_RB[:, 1]  # shape (2,)

    # Build cross block C: (2 × 2m) with columns [x_seg..., p_seg...]
    C = np.zeros((2, 2*m))
    C[:, 0:m]   = np.outer(cov_O_xc, w)   # obs with each x_seg_j
    C[:, m:2*m] = np.outer(cov_O_pc, w)   # obs with each p_seg_j

    if decouple_obs_from_rest:
        # wipe observer correlations to system, keep V_RR
        Gamma_full[[xO,pO], :] = 0.0
        Gamma_full[:, [xO,pO]] = 0.0
        Gamma_full[np.ix_([xO,pO],[xO,pO])] = V_RR

    # write observer-segment cross correlations
    Gamma_full[np.ix_([xO,pO], seg_idx)] = C
    Gamma_full[np.ix_(seg_idx, [xO,pO])] = C.T

    return sym(Gamma_full)

def extract_OR_collective_check(Gamma_full, right_seg, w, n_one_side):
    n = n_one_side
    Ntot = 2*n + 1
    xO = 2*n
    pO = Ntot + 2*n

    right_seg = np.asarray(right_seg, int)
    m = len(right_seg)
    w = np.asarray(w, float); w = w/np.linalg.norm(w)

    x_seg = right_seg
    p_seg = right_seg + Ntot
    seg_idx = np.r_[x_seg, p_seg]
    B = sym(Gamma_full[np.ix_(seg_idx, seg_idx)])

    O = orthogonal_with_first_row(w)
    S_dec = np.block([[O, np.zeros((m,m))],
                      [np.zeros((m,m)), O]])
    B_dec = sym(S_dec @ B @ S_dec.T)

    # first decoded mode covariance:
    Vcc = B_dec[np.ix_([0,m],[0,m])]

    # observer-segment cross, then map to observer-collective
    C = Gamma_full[np.ix_([xO,pO], seg_idx)]          # 2×2m
    # collective columns correspond to x_c = w^T x_seg, p_c = w^T p_seg
    cov_O_xc = C[:, 0:m] @ w
    cov_O_pc = C[:, m:2*m] @ w
    Voc = np.column_stack([cov_O_xc, cov_O_pc])       # 2×2

    Voo = Gamma_full[np.ix_([xO,pO],[xO,pO])]

    # Build 4×4 in xxpp order [xO, x_c, pO, p_c]
    V = np.zeros((4,4))
    V[np.ix_([0,2],[0,2])] = Voo
    V[np.ix_([1,3],[1,3])] = Vcc
    V[np.ix_([0,2],[1,3])] = Voc
    V[np.ix_([1,3],[0,2])] = Voc.T
    return sym(V)

def extract_collective_mode_cov_from_segment(Gamma, segment_modes, w):
    # Extract segment block
    B, _ = extract_block_xxpp_idx(Gamma, segment_modes)
    m = len(segment_modes)
    w = np.asarray(w, float); w = w/np.linalg.norm(w)
    O = orthogonal_with_first_row(w)
    S_dec = np.block([[O, np.zeros((m,m))],[np.zeros((m,m)), O]])
    B_dec = sym(S_dec @ B @ S_dec.T)
    return sym(B_dec[np.ix_([0,m],[0,m])])  # (x_c,p_c)

import numpy as np

def sym(A): 
    return 0.5*(A + A.T)


def extract_collective_mode_cov_from_segment_2(Gamma, segment_modes, w):
    """
    Directly compute Vcc for x_c = w^T x_seg, p_c = w^T p_seg
    from the segment covariance (xxpp).
    """
    B, _ = extract_block_xxpp_idx(Gamma, segment_modes)
    m = len(segment_modes)
    w = np.asarray(w, float)
    w = w / np.linalg.norm(w)

    B_xx = B[:m, :m]
    B_xp = B[:m, m:]
    B_px = B[m:, :m]
    B_pp = B[m:, m:]

    v_xx = float(w.T @ B_xx @ w)
    v_xp = float(w.T @ B_xp @ w)
    v_px = float(w.T @ B_px @ w)
    v_pp = float(w.T @ B_pp @ w)

    Vcc = np.array([[v_xx, v_xp],
                    [v_px, v_pp]], dtype=float)
    return sym(Vcc)


def teleport_test(s,theta,n_one_side,center_idx,m):
    N = n_one_side
    k = 1
    m_squared = 29
    HL = np.zeros((2*N,2*N))
    for i in range(2*N):
        if i < N-1:
            HL[i, i] = m_squared + 2 * k  # on-site + two neighbors
            HL[i,i+1] = -k
            HL[i+1, i] = -k    
        if i == N-1:
            HL[i,0] = -k
            HL[0,i] = -k 
            HL[i,i] = m_squared + 2 * k 
        if i > N-1:
            HL[i,i] = 2.5
    Gamma_reconstructed, nu, eps_reconstructed = build_thermal_state_from_modular_hamiltonian(HL)

    Gamma_TFD = gaussian_purification(Gamma_reconstructed)

    t0 = 2

    n = Gamma_TFD.shape[0] // 2
    bdy_len = Gamma_TFD.shape[0] // 4
    b = bdy_len


    HL_full = np.zeros((2*n, 2*n))
    HL_full[np.ix_(range(b), range(b))] = HL[:b, :b]                     # x-x
    HL_full[np.ix_(range(b), range(n, n + b))] = HL[:b, b:]             # x-p
    HL_full[np.ix_(range(n, n + b), range(b))] = HL[b:, :b]             # p-x
    HL_full[np.ix_(range(n, n + b), range(n, n + b))] = HL[b:, b:]      # p-p




    # Symplectic form
    Omega = symplectic_form(n)

    # Evolve forward in time

    S_forward = expm(1 * Omega @ HL_full * t0)
    Gamma_forward = S_forward @ Gamma_TFD @ S_forward.T


    Rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    Squeeze = 0.5 * np.array([[np.exp(-2*s), 0],
                          [0, np.exp(2*s)]])

    Gamma_insert= Rot @ Squeeze @ Rot.T

    Gamma_2mode = two_mode_squeezed_state(r=1)

    s0 = np.log(.09/11.72)
    squeeze_channel = np.array([[np.exp(-.5*s0), 0],
                          [0, np.exp(.5*s0)]])
    phi1 = -13*np.pi/28
    rot_channel_1 = np.array([[np.cos(phi1), -np.sin(phi1)],
                [np.sin(phi1), np.cos(phi1)]])
    phi2 = 27*np.pi/28
    rot_channel_2 = np.array([[np.cos(phi2), -np.sin(phi2)],
                [np.sin(phi2), np.cos(phi2)]])
    
    eta = .4

    loss = np.diag((eta,eta))

    X =  rot_channel_1 @ loss @ squeeze_channel @ rot_channel_2
    Y = np.array([[.24,-.82],[-.82,10.5]])

    V_ins = X @ Gamma_insert @ X.T + Y

    V_ins_observer = apply_channel_to_second_mode_xxpp(Gamma_2mode,X,Y)

    Gamma_final = Gamma_forward.copy()

    segment = right_segment_ids_centered(center_idx, N, m)

    w0 = np.ones(m)/np.sqrt(m)
    w1 = [i for i in range(1,N+1)]

    w2 = [i**2 for i in range(1,N+1)]

    w3 = [i**3 for i in range(1,N+1)]

    w4 = [i**4 for i in range(1,N+1)]

    w4f = np.flip(w4) 

    w12 = [1.2**i for i in range(1,N+1)]

    w11 = [446-1.1**i for i in range(1,N+1)]

    #w_m4 = [.001*i**2 for i in range(1,N+1)]
    #w_m4=np.array(w_m4)
    w_m4 = .001*np.ones(N)

    #w_m4[61:64] = 20
    #w_m4[0:2] = 20
    #w_m4[2:4] = 5
    #w_m4[59:61] = 5

    w_m4[30:33] = 20


    Y1 = np.array([[2,-1],[-1,11]])
    segment_1 = right_segment_ids_centered(center_idx, N, 64)
    #Gamma_final_out_1 = insert_collective_mode_into_segment_1(Gamma_final_1, segment_1, V_ins,w0)



    Gamma_final_out= insert_collective_mode_into_segment(Gamma_final, segment, V_ins,w0)
    #Gamma_final_out_2= insert_collective_mode_into_segment_2(Gamma_final, segment1,segment2,V_ins,w0)


    Gamma_final_out_observer = insert_RB_collective_into_right_segment_LRO(
    Gamma_LR=Gamma_forward,
    right_seg=segment,
    V_RB_xxpp=V_ins_observer,
    w=w0,      # uniform collective mode
    decouple_within_segment=True,
    decouple_obs_from_rest=True
    )

    V_check = extract_OR_collective_check(Gamma_final_out_observer, segment, w=np.ones(m)/np.sqrt(m), n_one_side=n_one_side)
    err_obs = np.linalg.norm(V_check - V_ins_observer)
    Vcc = extract_collective_mode_cov_from_segment(Gamma_final_out, segment, np.ones(m)/np.sqrt(m))
    err_no_obs=np.linalg.norm(Vcc - V_ins)

    #S_symp = decoder_from_X_symplectic(X)
    #S_flip = decoder_from_X_flip(X)
    #Fs = entanglement_fidelity_gaussian(X,Y,S_symp)
    #Ff = entanglement_fidelity_gaussian(X,Y,S_flip)


    return Gamma_final_out, Gamma_final_out_observer,X,Y




########
# End Test
########





def cp_check(X, Y, tol=1e-9):
    Omega=symplectic_form(X.shape[0]//2)
    M = Y + 1j*(Omega - X@Omega@X.T)
    eigs = np.linalg.eigvalsh(M)
    return eigs.min(), eigs


def fidelity_vs_block_size_old(
    block_sizes,
    obs_idx,
    teleported_idx,
    bdy_len,
    input_ensemble,   # list of (s, theta) you use for fitting
    H_coupling,
    N,
    center_idx,
    wormhole,
):
    insert_idx = teleported_idx - N
    Fms = []
    Fmf = []
    #Fm0 = []
    #Fma = []
    #F_active_test = []
    F_passive_symp_test = []
    F_passive_flip_test = []
    ## dXp = []
    ## dYp = []

    #Fm1 = []
    #Fm2 = []
    #Fm_null = []
    #dF = []
    #s1_on = []
    #s1_off = []

    #det_on = []
    #det_off = []
    s1_Y_on = []
    s1_Y_off = []
    #det_Y_on = []
    #det_Y_off = []

    s2_Y_on = []
    s2_Y_off = []

    #s10 = []
    #s20 = []

    #s1_active = []
    #s2_active = []
    #y_eff_list = []
    #y_eff_list_0 = []
    #y_eff_list_1 = []
    #y_eff_list_2 = []
    #y_eff_list_null = []

    # --- 1) Run ONE observer simulation ONCE (choose a representative input) ---
    # Use, e.g., s=0, theta=0, but inserted as half of TMSV with observer.
    Gamma_final_obs,Gamma_final, Gamma_forward_obs, Gamma_forward = teleportation_protocol(s=0.0, theta=0.0, insert_idx =insert_idx, wormhole=wormhole,n_one_side=N,H_coupling = H_coupling,coupling = True)
    #Gamma_final_test_1, Gamma_final_obs_test,X_true_test,Y_true_test = teleport_test(s=0,theta=0,n_one_side=N,center_idx=32,m=8)  
    #Gamma_final_obs_null,Gamma_final_null, Gamma_forward_obs_null, Gamma_forward_null = teleportation_protocol(s=0.0, theta=0.0, insert_idx =32, wormhole=wormhole,H_coupling = H_coupling,coupling = False)   
   
    # Gamma_final_obs must be in global [x_all,p_all] ordering.

    #prev = None
    #prev_test = None

    for m in block_sizes:
        # --- 2) Define right segment of length m ---
        right_seg = right_segment_ids_centered(center_idx, bdy_len, m)
        #left_seg = left_segment_ids_centered(center_idx, bdy_len, m)

        # --- 3) Build V_OR_xxpp and compute passive decoder O ---
        N_tot = 2*N+1
        obs_idx = 2*N
        V_OR_xxpp = build_V_OR_xxpp(Gamma_final_obs, obs_idx, right_seg,N_tot)

        #V_OR_xxpp_test = build_V_OR_xxpp(Gamma_final_obs_test, obs_idx, right_seg,N_tot)
        #V_OR_xxpp_0 = build_V_OR_xxpp(Gamma_forward_obs, obs_idx, left_seg,N_tot)
        #V_OR_xxpp_1 = build_V_OR_xxpp(Gamma_final_obs, obs_idx, left_seg,N_tot)
        #V_OR_xxpp_2 = build_V_OR_xxpp(Gamma_forward_obs, obs_idx, right_seg,N_tot)
        #V_OR_xxpp_null = build_V_OR_xxpp(Gamma_final_obs_null, obs_idx, right_seg,N_tot)
   
      
        O, v = build_passive_decoder_from_observer(V_OR_xxpp, m)
        #O_test, v_test = build_passive_decoder_from_observer(V_OR_xxpp_test, m)
        
        #O0, v0 = build_passive_decoder_from_observer(V_OR_xxpp_0, m)
        #O1, v1 = build_passive_decoder_from_observer(V_OR_xxpp_1, m)
        #O2, v2 = build_passive_decoder_from_observer(V_OR_xxpp_2, m)

        # --- 4) Build training pairs (Vin, Vout_decoded_firstmode) ---
        Vins, Vouts = [], []
        #Vouts_0 = []
        #Vouts_1 = []
        #Vouts_2 = []
        #Vouts_null = []

        #Vouts_active = []

        #Vouts_active_test = []
        #Vouts_passive_test = []
       

        #S_dec_active_build_X,info= active_decoder_williamson_tracked(V_OR_xxpp, m,lam=1e-4, rmax=1.2, Kcand=6, prev_state=prev)
        #prev = info["prev_state"]
        
        #S_dec_active_build_X_test,info_test= active_decoder_williamson_tracked(V_OR_xxpp_test, m,lam=1e-4, rmax=1.2, Kcand=6, prev_state=prev_test)
        #prev_test = info_test["prev_state"]
              
        #V_OR_dec = apply_right_decoder_to_VOR(V_OR_xxpp, S_dec_active_build_X)
    
        #S_dec_build_X_0 = build_full_symplectic_decoder_from_observer(V_OR_xxpp_0, m)
        #S_dec_build_X_null = build_full_symplectic_decoder_from_observer(V_OR_xxpp_null, m)

        for s, theta in input_ensemble:
            # Run your usual protocol (NO observer) to get global Gamma_final
            Gamma_final_obs_1, Gamma_final, Gamma_forward_obs_1,Gamma_forward = teleportation_protocol(
                s=s, theta=theta, insert_idx =insert_idx,wormhole=wormhole,n_one_side=N,H_coupling=H_coupling,coupling = True
            )
            #Gamma_final_obs_null, Gamma_final_null, Gamma_forward_obs_null,Gamma_forward_null = teleportation_protocol(
                #s=s, theta=theta, insert_idx =insert_idx,wormhole=wormhole,n_one_side=N,H_coupling=H_coupling,coupling = False
            #)

            #Gamma_final_test, Gamma_final_obs_test_1,_,_ = teleport_test(s,theta,n_one_side=N,center_idx=32,m=8)  

            #single-site, no decoder blocks:



            # Extract right segment block covariance in xxpp
            B_xxpp = extract_block_xxpp(Gamma_final, right_seg)   # 2m x 2m
            #B_xxpp_test = extract_block_xxpp(Gamma_final_test, right_seg)   # 2m x 2m
            
            #B_xxpp_0 = extract_block_xxpp(Gamma_forward, left_seg)   # 2m x 2m
            #B_xxpp_1 = extract_block_xxpp(Gamma_final, left_seg)   # 2m x 2m
            #B_xxpp_2 = extract_block_xxpp(Gamma_forward, right_seg)   # 2m x 2m
            #B_xxpp_null = extract_block_xxpp(Gamma_final_null, right_seg)   # 2m x 2m



            # Passive decode
            B_dec = passive_decode_right_block(B_xxpp, O)
            #B_dec_passive_test = passive_decode_right_block(B_xxpp_test, O_test)
            #B_dec_0 = passive_decode_right_block(B_xxpp_0, O0)
            #B_dec_1 = passive_decode_right_block(B_xxpp_1, O1)
            #B_dec_2 = passive_decode_right_block(B_xxpp_2, O2)
            #B_dec_null = passive_decode_right_block(B_xxpp_null, O)

            # active decode
            #Vout_active, B_dec_full = decode_block_first_mode(B_xxpp, S_dec_active_build_X)
            #Vout_active_test, B_dec_full_test = decode_block_first_mode(B_xxpp_test, S_dec_active_build_X_test)
            
            
            #Vout_0, B_dec_full_0 = decode_block_and_take_mode(B_xxpp_0, S_dec_build_X_0, mode_k=0)
            #Vout_null, B_dec_full_null = decode_block_and_take_mode(B_xxpp_null, S_dec_build_X, mode_k=0)


            # Take decoded first mode as Vout (2x2)
            Vout = first_mode_from_block(B_dec)
            #Vout_passive_test = first_mode_from_block(B_dec_passive_test)
            
            #Vout_0 = first_mode_from_block(B_dec_0)
            #Vout_1 = first_mode_from_block(B_dec_1)
            #Vout_2 = first_mode_from_block(B_dec_2)
            #Vout_null = first_mode_from_block(B_dec_null)

            if len(block_sizes)==1:
                Vout = extract_subsystem_covariance(Gamma_final,[center_idx+N])


            # Your input Vin is the inserted mode covariance (2x2)
            Vin = make_input_covariance(s, theta)  # returns 2x2 in (x,p)

            Vins.append(Vin)
            Vouts.append(Vout)
            #Vouts_0.append(Vout_0)
            #Vouts_1.append(Vout_1)
            #Vouts_2.append(Vout_2)
            #Vouts_null.append(Vout_null)

            #Vouts_active.append(Vout_active)
            #Vouts_active_test.append(Vout_active_test)
            #Vouts_passive_test.append(Vout_passive_test)


        # --- 5) Fit a single-mode Gaussian channel for this decoded mode ---
        X, Y = fit_gaussian_channel(Vins, Vouts)
        #print("Y_passive=",Y)
        #X0, Y0 = fit_gaussian_channel(Vins, Vouts_0)
        #X1, Y1 = fit_gaussian_channel(Vins, Vouts_1)
        #X2, Y2 = fit_gaussian_channel(Vins, Vouts_2)
        #X_null, Y_null = fit_gaussian_channel(Vins, Vouts_null)
        #X_active,Y_active = fit_gaussian_channel(Vins,Vouts_active)
        #print("Y_active =",Y_active)
        #print("active cp=",cp_check(X_active,Y_active))
        #X_active_test,Y_active_test = fit_gaussian_channel(Vins,Vouts_active_test)
        #X_passive_test,Y_passive_test = fit_gaussian_channel(Vins,Vouts_passive_test)


        #dX_active = np.linalg.norm(X_true_test - X_active_test)
        #dX_passive = np.linalg.norm(X_true_test - X_passive_test)
        #dXp.append(dX_passive)
        #dY_passive = np.linalg.norm(Y_true_test - Y_passive_test)
        #dYp.append(dY_passive)
        #print(m,"X diff =",np.linalg.norm(X_active_test-X_true_test))

        rot1,loss,squeeze,rot2=decompose_X(X)
        print(m,center_idx,"rot1=",rot1)
        print(m,"rot2=",rot2)
        print(m,"loss=",loss)
        print(m,"squeeze=",squeeze)
        print(m, "Y=",Y)

        
        #met_on  = X_metrics(X)
        #met_off = X_metrics(X_null)
        #s1_on.append(met_on["s1"])   
        #s1_off.append(met_off["s1"])

        #met_0 = X_metrics(X0)


        #met_active = X_metrics(X_active)

        #det_on.append(met_on("det"))
        #det_off.append(met_off("det"))


        #y_eff_list.append(noise_metrics(X,Y)[1])
        #y_eff_list_0.append(noise_metrics(X0,Y0)[1])
        #y_eff_list_1.append(noise_metrics(X1,Y1)[1])
        #y_eff_list_2.append(noise_metrics(X2,Y2)[1])
        #y_eff_list_null.append(noise_metrics(X_null,Y_null)[1])

        #Y_norm_on = noise_metrics(X,Y)[1]
        #Y_norm_off = noise_metrics(X_null,Y_null)[1]
        #Y_norm_0 = noise_metrics(X0,Y0)[1]
        #Y_norm_active = noise_metrics(X_active,Y_active)[1]

        #s1_Y_on.append(met_on["s1"]/Y_norm_on)
        #s1_Y_off.append(met_off["s1"]/Y_norm_off)
        #s1_active.append(met_active["s1"]/Y_norm_active)

        #s2_Y_on.append(met_on["s2"]/Y_norm_on)
        #s2_Y_off.append(met_off["s2"]/Y_norm_off)
        #s2_active.append(met_active["s2"]/Y_norm_active)

        #det_Y_on.append(met_on["det"]/Y_norm_on)
        #det_Y_off.append(met_off["det"]/Y_norm_off)

        #s10.append(met_0["s1"]/Y_norm_0)
        #s20.append(met_0["s2"]/Y_norm_0)
        # Optional: check CP
        # assert cp_check(X,Y) > -1e-8

        # --- 6) Compute entanglement fidelity ---
        # If you already have a symplectic single-mode decoder from X, apply it
        S_dec_symp = decoder_from_X_symplectic(X)  # your preferred
        S_dec_flip = decoder_from_X_flip(X)  # your preferred
        #S_dec_flip = X

        #S_dec_0 = decoder_from_X_symplectic(X0, mode="rotation+squeeze")  # your preferred
        #S_dec_1 = decoder_from_X_symplectic(X1, mode="rotation+squeeze")  # your preferred
        #S_dec_2 = decoder_from_X_symplectic(X2, mode="rotation+squeeze")  # your preferred
        #S_dec_null = decoder_from_X_symplectic(X_null, mode="rotation+squeeze")  # your preferred
        #S_dec_active = decoder_from_X_symplectic(X_active) 
        #S_dec_active_test = decoder_from_X_symplectic(X_active_test)
        #S_dec_passive_symp_test = decoder_from_X_symplectic(X_passive_test)
        #S_dec_passive_flip_test = decoder_from_X_flip(X_passive_test)
  


        Fs = entanglement_fidelity_gaussian(X, Y, S_dec_symp, subtract_Y=False,r=1.0)
        Ff = entanglement_fidelity_gaussian(X, Y, S_dec_flip, subtract_Y=False,r=1.0)      
        
        #F0 = entanglement_fidelity_gaussian(X0, Y0, S_dec_0, r=1.0)
        #F1 = entanglement_fidelity_gaussian(X1, Y1, S_dec_1, r=1.0)
        #F2 = entanglement_fidelity_gaussian(X2, Y2, S_dec_2, r=1.0)
        #F_null = entanglement_fidelity_gaussian(X_null, Y_null, S_dec_null, r=1.0)
        #Fa = entanglement_fidelity_gaussian(X_active,Y_active,S_dec_active,r=1.0)
        #Fat = entanglement_fidelity_gaussian(X_active_test,Y_active_test,S_dec_active_test,r=1.0)
        #Fpst = entanglement_fidelity_gaussian(X_passive_test,Y_passive_test,S_dec_passive_symp_test,r=1.0)
        #Fpft = entanglement_fidelity_gaussian(X_passive_test,Y_passive_test,S_dec_passive_flip_test,r=1.0)

        Fms.append(Fs)
        Fmf.append(Ff)
        #Fm0.append(F0)
        #Fm1.append(F1)
        #Fm2.append(F2)
        #Fm_null.append(F_null)
        #dF.append(F-F_null)
        #Fma.append(Fa)
        #F_active_test.append(Fat)
        #F_passive_symp_test.append(Fpst)
        #F_passive_flip_test.append(Fpft)



    return np.array(Fms),np.array(Fmf)#,np.array(F_passive_symp_test),F_passive_flip_test,s1_Y_on,s2_Y_on#,dXp,dYp #np.array(Fm0),np.array(Fm1),np.array(Fm2),np.array(y_eff_list),np.array(y_eff_list_0),np.array(y_eff_list_1),np.array(y_eff_list_2)



def fidelity_vs_site(
    insert_idx,
    input_ensemble,   # list of (s, theta) you use for fitting
    H_coupling,
    N,
    wormhole):


    Vins = []

    Vouts = [[] for i in range(N)]


    for s, theta in input_ensemble:
        # Run your usual protocol (NO observer) to get global Gamma_final
        Gamma_final_obs_1, Gamma_final, Gamma_forward_obs_1,Gamma_forward = teleportation_protocol(
                s=s, theta=theta, insert_idx =insert_idx,wormhole=wormhole,n_one_side=N,H_coupling=H_coupling,coupling = True
            )        
            
        Vins.append(make_input_covariance(s,theta))
        for i in range(N):
            Vouts[i].append(extract_subsystem_covariance(Gamma_final,[i+N]))
        

        # --- 5) Fit a single-mode Gaussian channel for this decoded mode ---

    fid_symp = []
    fid_flip = []


    for i in range(N):
        X, Y = fit_gaussian_channel(Vins, Vouts[i])
        rot1,loss,squeeze,rot2 = decompose_X(X)
        print(i+N)
        print(f"rot1={rot1}")
        print(f"rot2={rot2}")
        print(f"loss={loss}")
        print(f"squeeze={squeeze}")
        print(f"Y={Y}")

        S_dec_symp = decoder_from_X_symplectic(X)  # your preferred
        S_dec_flip = decoder_from_X_flip(X)  # your preferred

        Fs = entanglement_fidelity_gaussian(X, Y, S_dec_symp, subtract_Y=False, r=1.0)
        Ff = entanglement_fidelity_gaussian(X, Y, S_dec_flip, subtract_Y=False, r=1.0)

        fid_symp.append(Fs)
        fid_flip.append(Ff)

        print(f"fid_flip_3={Ff}")
        print(f"fid_symp_3={Fs}")

    return fid_symp,fid_flip



site_fidelities_symp=[]
site_fidelities_flip=[]
block_sizes = [1]
#block_sizes = [1,2,4,6,8,10]

N = 3
obs_idx = 2*N
insert_idx = 1
teleported_idx = insert_idx+N
bdy_len = N

Ss = np.linspace(-1, 1, 4)
Thetas = np.linspace(0, 2*np.pi, 3, endpoint=False)
input_ensemble = [(s, th) for s in Ss for th in Thetas]  # 120 points, deterministic

sites=np.arange(N,2*N)

#for f in range(len(sites)):

Fs,Ff= fidelity_vs_site(insert_idx,input_ensemble,H_coupling_OG,N=N,wormhole=False) 
"""
for f in range(len(sites)):
    #Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    #plt.plot(block_sizes,Fs,label=sites[f])
Fs,Ff= fidelity_vs_site(insert_idx,input_ensemble,H_coupling_OG,N=N,wormhole=False)   
    #site_fidelities_symp.append(Fs)
    #site_fidelities_flip.append(Ff)

"""
plt.xlabel("decoder block size")
plt.ylabel("fidelity")
plt.legend()
plt.show()
"""    

plt.plot(sites,Fs,label="symplectic")
plt.plot(sites,Ff,label="allow flip")
plt.xlabel("site")
plt.ylabel("fidelity")
plt.legend()
plt.show()

print("done")


"""
Gamma_final_obs, Gamma_final, Gamma_forward_obs, Gamma_forward = teleportation_protocol(
    s=0.0, theta=0.0, insert_idx=32, wormhole=True,H_coupling = H_coupling_OG,coupling = True
)


m0 = 24
K0 = 7
mu = 1
teleported_idx = 32 + 64
left_seg  = left_segment_ids(insert_id=32, n=bdy_len, m=m0)                 # 0..n-1
right_seg = right_segment_ids(teleported_idx, n=bdy_len, m=m0)               # n..2n-1  (make sure!)

V_OL_xxpp = build_V_OR_xxpp(Gamma_forward_obs, obs_idx=128, right_seg=left_seg,  Ntot=129)
V_OR_xxpp = build_V_OR_xxpp(Gamma_forward_obs, obs_idx=128, right_seg=right_seg, Ntot=129)

#O_L, lamL = build_passive_decoder_from_observer_K(V_OL_xxpp, m0, K=K0)
#O_R, lamR = build_passive_decoder_from_observer_K(V_OR_xxpp, m0, K=K0)

O_L, lamL = build_passive_decoder_from_observer(V_OL_xxpp, m0)
O_R, lamR = build_passive_decoder_from_observer(V_OR_xxpp, m0)




H_coupling_codeaware = build_rankK_coupling_LRO(
    N_boundary=bdy_len,
    left_seg=left_seg,
    right_seg=right_seg,
    O_L=O_L, O_R=O_R,
    g=np.ones(K0)*mu,
    include_observer=False   # if you’re coupling the no-observer Gamma
)


H_coupling_codeaware_2 = build_coupling_LRO(
    N_boundary=bdy_len,
    left_seg=left_seg,
    right_seg=right_seg,
    O_L=O_L, O_R=O_R,
    g=np.ones(K0)*mu,
    include_observer=False   # if you’re coupling the no-observer Gamma
)
"""


#####
# begin mutual info test
#####



def mut_info_vs_segments(s,theta,n_one_side,center_idx_telep,m,insert_idx,wormhole,H_coupling,coupling,center_idx,test):
    if test == True:
        Gamma_TFD, Gamma_LR,_,_ = teleport_test(s,theta,n_one_side,center_idx_telep,m)
    else: 
        Gamma_LR, Gamma_TFD,_,_ = teleportation_protocol(s,theta,insert_idx,wormhole,n_one_side,H_coupling,coupling)
    
    mut_info_insert_regions = []
    mut_info_telep_regions = []

    lengths_array = np.linspace(1,Gamma_TFD.shape[0] // 8,Gamma_TFD.shape[0] // 8)



    observer_idx = 2*n_one_side
    n_L = n_one_side
    n = 2*n_one_side


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
    full_lengths_array[-1] = 64    
    plt.plot(full_lengths_array,mut_info_insert_regions,color='k',label = "insert side")
    plt.plot(full_lengths_array,mut_info_telep_regions,color='red', label ="teleport side")
    plt.axhline(compute_MI_with_observer(Gamma_LR,observer_idx,list(range(n))), color = "blue", label = "total mutual info with observer")
    plt.xlabel("length of segment")
    plt.ylabel("mutual info with observer")
    plt.title("mutual information of segments with observer")
    plt.legend()
    plt.show()

mut_info_vs_segments(s=0,theta=0,n_one_side=64,center_idx_telep=32,m=8,insert_idx=32,wormhole=False,H_coupling=H_coupling_OG,coupling=True,center_idx=32,test=True)
mut_info_vs_segments(s=0,theta=0,n_one_side=64,center_idx_telep=32,m=8,insert_idx=32,wormhole=False,H_coupling=H_coupling_OG,coupling=True,center_idx=30,test=False)

#######
# end mutual info test
#######




#block_sizes = [1,2,4,8,12,16,20,24,28,30,31,32,33,34,36,40,44,48,52,56,60,64]
#block_sizes = [1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]
block_sizes = [1]
obs_idx = 128
insert_idx = 1
teleported_idx = 1 + 3
bdy_len = 1
"""
input_ensemble = [(0.0,0), (.5,0), (-.5,np.pi/4),
                 (1,np.pi/4), (-1,np.pi/2), (.4,np.pi/2)]
"""
"""
input_ensemble = []

for i in range(60):
    input_ensemble.append((np.random.uniform(-1.5,1.5),np.random.uniform(0,2*np.pi)))
"""



Fs,Ff,F_passive_symp_test,F_passive_flip_test,s1_Y_on,s2_Y_on= fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=3,center_idx=10,wormhole=False)


#plt.rc('font', size=25) 
plt.plot(block_sizes, s1_Y_on, marker='o',label="s1 passive")
plt.plot(block_sizes, s2_Y_on, marker='o',label="s2 passive")
#plt.plot(block_sizes, s1_active, marker='o',label="s1 active")
#plt.plot(block_sizes, s2_active, marker='o',label="s2 active")

#plt.plot(block_sizes, s10, marker='o',label="s1 before coupling")
#plt.plot(block_sizes, s20, marker='o',label="s2 before coupling")
#plt.plot(block_sizes, det_Y_on, marker='o',label="det(X) coupling")
#plt.plot(block_sizes, s1_Y_off, marker='o',label="det(X) no coupling")

#plt.plot(block_sizes, F0, marker='o',label="left before coupling")
#plt.plot(block_sizes, F1, marker='o',label="left after coupling")
#plt.plot(block_sizes, F2, marker='o',label="right before coupling")
#plt.plot(block_sizes, dF, marker='o',label="coupling - no coupling")

plt.xlabel("decoder block size")
plt.ylabel("singular value/noise")
plt.title("Singular Values vs. Decoder Block Size")
plt.legend()
plt.show()

"""
plt.plot(block_sizes, s2_Y_on, marker='o',label="coupling")
plt.plot(block_sizes, s2_Y_off, marker='o',label="no coupling")

plt.xlabel("decoder block size m")
plt.ylabel("(second largest singular value of X)/noise")
plt.legend()
plt.show()


plt.plot(block_sizes, det_Y_on, marker='o',label="coupling")
plt.plot(block_sizes, det_Y_off, marker='o',label="no coupling")

plt.xlabel("decoder block size m")
plt.ylabel("det(X)/noise")
plt.legend()
plt.show()

"""

#plt.rc('font', size=25) 

plt.plot(block_sizes, Fs, marker='o',label="symplectic")
plt.plot(block_sizes, Ff, marker='o',label="allow flip")
#plt.plot(block_sizes, Fa, marker='o',label="active")

#plt.plot(block_sizes, F0, marker='o',label="before coupling")
#plt.plot(block_sizes, F_null, marker='o',label="no coupling")

#plt.plot(block_sizes, y_eff_list_0, marker='o',label="left before coupling")
#plt.plot(block_sizes, y_eff_list_1, marker='o',label="left after coupling")
#plt.plot(block_sizes, y_eff_list_2, marker='o',label="right before coupling")
#plt.plot(block_sizes, y_eff_list_null, marker='o',label="right after coupling")
plt.xlabel("decoder block size")
plt.ylabel("fidelity")
plt.title("Fidelity vs. Decoder Block Size")
plt.legend()
plt.show()

#plt.plot(block_sizes, F_active_test, marker='o',label="active")
plt.plot(block_sizes, F_passive_symp_test, marker='o',label="symplectic")
plt.plot(block_sizes, F_passive_flip_test, marker='o',label="allow flip")
plt.xlabel("decoder block size")
plt.ylabel("fidelity")
plt.title("Fidelity vs. Decoder Block Size Test")
plt.legend()
plt.show()

"""
plt.plot(block_sizes, dXp, marker='o',label="dX")
plt.plot(block_sizes, dYp, marker='o',label="dY")
plt.xlabel("decoder block size")
plt.ylabel("log(||true-calc||)")
plt.yscale("log")
plt.title("dX vs. Decoder Block Size Test")
plt.legend()
plt.show()
"""




print("stop")

