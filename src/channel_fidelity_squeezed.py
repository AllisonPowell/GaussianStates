import numpy as np
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
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


def H_coupling(N):
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
    k = 5
    m_squared = 13
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
    return H_coupling_OG


def generate_interacting_tfd(n, omega_0, J, beta,periodic):
    """
    Generates the coupled modular Hamiltonian and covariance matrix
    for a continuous-variable tight-binding chain.
    """
    # 1. Construct the spatial hopping matrix (Hamiltonian h)
    h = omega_0 * np.eye(n)
    for i in range(n - 1):
        h[i, i+1] = -J
        h[i+1, i] = -J
    if periodic==True:
        h[0,n-1] = -J
        h[n-1,0] = -J

    # 2. Diagonalize to find collective normal modes
    eigenvalues, V = np.linalg.eigh(h)

    # 3. Calculate mode-dependent squeezing parameters from temperature
    # cosh(2r) = coth(beta * omega / 2)
    cosh_r = np.zeros(n)
    sinh_r = np.zeros(n)
    lambda_diagonal = np.zeros(n)

    for i, omega_i in enumerate(eigenvalues):
        # Prevent division by zero for unphysical modes
        omega_i = max(omega_i, 1e-5)
        # Physical relation: tanh(r) = exp(-beta * omega / 2)
        tanh_ri = np.exp(-beta * omega_i / 2)
        # Avoid pure saturation limits
        tanh_ri = min(tanh_ri, 0.999) 
        
        # Recover squeezing parameter r
        r_i = np.arctanh(tanh_ri)
        
        cosh_r[i] = np.cosh(2 * r_i)
        sinh_r[i] = np.sinh(2 * r_i)
        lambda_diagonal[i] = np.log(1.0 / tanh_ri)

    # 4. Assemble matrices in the normal-mode basis
    C_mat = np.diag(cosh_r)
    S_mat = np.diag(sinh_r)
    zeros_n = np.zeros((n, n))

    Gamma_NM = np.block([
        [C_mat, S_mat, zeros_n, zeros_n],
        [S_mat, C_mat, zeros_n, zeros_n],
        [zeros_n, zeros_n, C_mat, -S_mat],
        [zeros_n, zeros_n, -S_mat, C_mat]
    ])

    KL_NM_block = np.diag(lambda_diagonal)
    KL_NM = np.block([
        [KL_NM_block, zeros_n],
        [zeros_n, KL_NM_block]
    ])

    # 5. Transform back to the physical local spatial basis using V
    V_4n = block_diag(V, V, V, V)
    V_2n = block_diag(V, V)

    Gamma_physical = V_4n @ Gamma_NM @ V_4n.T
    KL_physical = V_2n @ KL_NM @ V_2n.T

    return Gamma_physical, KL_physical


def make_boundary_coupling(n, insert_idx, g):

    coupling_sites_1 = np.arange(0,insert_idx)
    coupling_sites_2 = np.arange(insert_idx+1,n)
    coupling_sites= np.concatenate((coupling_sites_1,coupling_sites_2))


    N = 4*n
    G = np.zeros((N,N))

    for j in coupling_sites:

        # x_L x_R
        G[j, n+j] = g
        G[n+j, j] = g

        # p_L p_R
        G[2*n+j, 3*n+j] = g
        G[3*n+j, 2*n+j] = g

    return G

def teleportation_protocol(s,theta,n,insert_idx, omega_0, J, beta,H_coupling,t_evolve,t_couple,periodic):
    Gamma_TFD, HL = generate_interacting_tfd(n, omega_0, J, beta,periodic)

    HL_full = np.zeros((4*n, 4*n))
    HL_full[np.ix_(range(n), range(n))] = HL[:n, :n]                     # x-x
    HL_full[np.ix_(range(n), range(2*n, 3*n))] = HL[:n, n:]             # x-p
    HL_full[np.ix_(range(2*n, 3*n), range(n))] = HL[n:, :n]             # p-x
    HL_full[np.ix_(range(2*n, 3*n), range(2*n, 3*n))] = HL[n:, n:]      # p-p



    # Symplectic form
    Omega = symplectic_form(2*n)

    # Evolve backward in time

    S_back = expm(-1 * Omega @ HL_full * t_evolve)
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
    S_forward_no_insert = expm(Omega @ HL_full * t_evolve)
    Gamma_forward = S_forward_no_insert @ Gamma_insert @ S_forward_no_insert.T

    n_total = (Gamma_with_observer.shape[0]) // 2  # now 2n+1
    Omega_padded = symplectic_form(n_total)
    S_forward_observer = expm(Omega_padded @ HL_full_padded * t_evolve)
    Gamma_forward_observer = S_forward_observer @ Gamma_with_observer @ S_forward_observer.T

    #######
    # couple the two sides
    #######

  
    S_coupling = expm(Omega @ H_coupling * t_couple)
    Gamma_coupled = S_coupling @ Gamma_forward @ S_coupling.T

    H_coupling_padded = pad_matrix_for_observer(H_coupling)
    S_coupling_observer = expm(Omega_padded @ H_coupling_padded * t_couple)
    Gamma_coupled_observer = S_coupling_observer @ Gamma_forward_observer @ S_coupling_observer.T

    #Gamma_coupled_observer = Gamma_forward_observer
    #Gamma_coupled = Gamma_forward

    ######
    # evolve state forwards in time with KR
    ######


    HR_full = np.zeros((4*n, 4*n))
    HR_full[np.ix_(range(n, 2*n), range(n, 2*n))] = HL[:n, :n]
    HR_full[np.ix_(range(n, 2*n), range(3*n, 4*n))] = HL[:n, n:]
    HR_full[np.ix_(range(3*n, 4*n), range(n, 2*n))] = HL[n:, :n]
    HR_full[np.ix_(range(3*n, 4*n), range(3*n, 4*n))] = HL[n:, n:]

    HR_full_padded = pad_matrix_for_observer(HR_full)



    S_final = expm(Omega @ HR_full * t_evolve)
    Gamma_final = S_final @ Gamma_coupled @ S_final.T

    S_final_observer = expm(Omega_padded @ HR_full_padded * t_evolve)
    Gamma_final_observer = S_final_observer @ Gamma_coupled_observer @ S_final_observer.T


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






def fidelity_vs_site(
    insert_idx,
    input_ensemble,   # list of (s, theta) you use for fitting
    H_coupling,
    n,
    t_evolve,
    t_couple,
    omega_0,
    J,
    beta,
    periodic):


    Vins = []

    Vouts = [[] for i in range(2*n)]


    for s, theta in input_ensemble:
        # Run your usual protocol (NO observer) to get global Gamma_final
        Gamma_final_obs_1, Gamma_final, Gamma_forward_obs_1,Gamma_forward = teleportation_protocol(
                s,theta,n,insert_idx, omega_0, J, beta,H_coupling,t_evolve,t_couple,periodic
            )        
            
        Vins.append(make_input_covariance(s,theta))
        for i in range(2*n):
            Vouts[i].append(extract_subsystem_covariance(Gamma_final,[i]))
            #Vouts[i].append(extract_subsystem_covariance(Gamma_final,[i]))
        

        # --- 5) Fit a single-mode Gaussian channel for this decoded mode ---

    fid_symp = []
    fid_flip = []


    for i in range(2*n):
        X, Y = fit_gaussian_channel(Vins, Vouts[i])
        rot1,loss,squeeze,rot2 = decompose_X(X)
        #print(i)
        #print(f"rot1={rot1}")
        #print(f"rot2={rot2}")
        #print(f"loss={loss}")
        #print(f"squeeze={squeeze}")
        #print(f"Y={Y}")

        S_dec_symp = decoder_from_X_symplectic(X)  # your preferred
        S_dec_flip = decoder_from_X_flip(X)  # your preferred

        Fs = entanglement_fidelity_gaussian(X, Y, S_dec_symp, subtract_Y=False, r=1.0)
        Ff = entanglement_fidelity_gaussian(X, Y, S_dec_flip, subtract_Y=False, r=1.0)

        fid_symp.append(Fs)
        fid_flip.append(Ff)

        #print(f"fid_flip_3={Ff}")
        #print(f"fid_symp_3={Fs}")

    return fid_symp,fid_flip


n = 10
omega_0 = 1
J = .4
beta = 1
insert_idx = 1
t_evolve = 4.55
t_couple = 1.6



Ss = np.linspace(-1, 1, 4)
Thetas = np.linspace(0, 2*np.pi, 3, endpoint=False)
input_ensemble = [(s, th) for s in Ss for th in Thetas]  # 120 points, deterministic

sites=np.arange(0,2*n)

#for f in range(len(sites)):

H_coupling = make_boundary_coupling(n, insert_idx, g=1)

Fs,Ff= fidelity_vs_site(
    insert_idx,
    input_ensemble,
    H_coupling,
    n,
    t_evolve,
    t_couple,
    omega_0,
    J,
    beta,
    periodic=True)



#plt.plot(sites,Fs,label="symplectic")
plt.rc('font', size=14)
plt.plot(sites,Ff,color="k",linewidth=2)
plt.axvline(insert_idx,color="blue",linestyle="dashed",linewidth=2,label="insert site")
plt.axvline(insert_idx+n,color="red",linestyle="dashed",linewidth=2,label="teleport site")
plt.xlabel("site")
plt.ylabel("fidelity")
plt.title("Channel Fidelity")
plt.legend()
plt.show()



t_evolve_fid = np.linspace(.1,20,80)
fidelity_evolve_list = []
for t in range(len(t_evolve_fid)):
    Fs,Ff= fidelity_vs_site(
        insert_idx,
        input_ensemble,
        H_coupling,
        n,
        t_evolve_fid[t],
        t_couple,
        omega_0,
        J,
        beta,
        periodic=False)
    fidelity_evolve_list.append(Ff[insert_idx+n])
    #print(fidelity_evolve_list[-1],t_evolve_list[t])



t_couple_fid = np.linspace(.1,12.5,80)
fidelity_couple_list = []
for t in range(len(t_couple_fid)):
    Fs,Ff= fidelity_vs_site(
        insert_idx,
        input_ensemble,
        H_coupling,
        n,
        t_evolve,
        t_couple_fid[t],
        omega_0,
        J,
        beta,
        periodic=False)
    fidelity_couple_list.append(Ff[insert_idx+n])
    #print(fidelity_couple_list[-1],t_couple_list[t])




t_evolve_mi = np.linspace(.1,20,80)
t_couple_mi = np.linspace(.1,12.5,80)

mi_evolve_list = []
mi_couple_list = []

s=1
theta = np.pi/2

for t in range(len(t_evolve_mi)):
    Gamma_obs_evolve,_,_,_ = teleportation_protocol(s,theta,n,insert_idx, omega_0, J, beta,H_coupling,t_evolve_mi[t],t_couple,periodic=False)
    mi_evolve = mutual_information(Gamma_obs_evolve,[2*n],list(range(n,2*n)))
    mi_evolve*= 1/(mutual_information(Gamma_obs_evolve,[2*n],list(range(0,2*n))))
    mi_evolve_list.append(mi_evolve)
    
    Gamma_obs_couple,_,_,_ = teleportation_protocol(s,theta,n,insert_idx, omega_0, J, beta,H_coupling,t_evolve,t_couple_mi[t],periodic=False)
    mi_couple = mutual_information(Gamma_obs_couple,[2*n],list(range(n,2*n)))
    mi_couple*= 1/(mutual_information(Gamma_obs_couple,[2*n],list(range(0,2*n))))    
    mi_couple_list.append(mi_couple)


plt.rc('font', size=14)
plt.plot(t_evolve_mi,mi_evolve_list,'k-',label="mutual information")
plt.plot(t_evolve_fid,fidelity_evolve_list,'r-',label="fidelity")
plt.axhline(.2225,color="blue",linestyle="dashed",label="no coupling")
plt.xlabel("evolution time")
plt.ylabel("metric")
plt.legend()
plt.show()

plt.rc('font', size=14)
plt.plot(t_couple_mi,mi_couple_list,'k-',label="mutual information")
plt.plot(t_couple_fid,fidelity_couple_list,'r-',label="fidelity")
plt.axhline(.2225,color="blue",linestyle="dashed",label="no coupling")
plt.xlabel("coupling time")
plt.ylabel("metric")
plt.legend()
plt.show()

print("stop")

