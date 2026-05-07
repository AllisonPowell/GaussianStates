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
    H_coupling = np.zeros((2*n_total, 2*n_total))
    mu = 1
    k = 5
    m_squared = 13
    omega0 = np.sqrt(m_squared + 2*k)
    #omega0=1

    for j in carrier_indices:
        x_L = bdy_1_idx[j]
        x_R = bdy_2_idx[j]
        # x coupling
        H_coupling[x_L, x_R] = H_coupling[x_R, x_L] = mu*omega0 / 2
        # p coupling
        H_coupling[x_L + n_total, x_R + n_total] = H_coupling[x_R + n_total, x_L + n_total] = mu / (2*omega0)
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

    #H_LR = HL_full+HR_full

    #H_coupling_OG += H_LR
    """
    return H_coupling






def sym(A): 
    return 0.5*(A + A.T)



def measure_left_side(Gamma,bdy_len):
    n = Gamma.shape[0]//2
    na = bdy_len
    Gamma_AA = np.zeros((2*na,2*na))

    Gamma_AA = np.zeros((2*na,2*na))
    for i in range(1,3):
        for j in range(1,3):

            Gamma_AA[i*na-na:i*na,j*na-na:j*na]=Gamma[i*n-na:i*n,j*n-na:j*n]
       
    Gamma_BB = np.zeros((2*na,2*na))
    for i in range(2):
        for j in range(2):
            Gamma_BB[i*na:i*na+na,j*na:j*na+na]=Gamma[i*n:i*n+na,j*n:j*n+na]

    Gamma_AB = np.zeros((2*na,2*na))

    for i in range(1,3):
        for j in range(2):
            Gamma_AB[i*na-na:i*na,j*na:j*na+na]=Gamma[i*n-na:i*n,j*n:j*n+na]

    m = Gamma_BB.shape[0]//2
    P = momentum_projection_matrix(m)
    V_bdy = Gamma_AA - Gamma_AB @ np.linalg.pinv(P @ Gamma_BB @ P) @ Gamma_AB.T

    return V_bdy



def teleportation_protocol(s,theta,insert_idx,wormhole,n_one_side,n_tube,H_coupling,coupling):
    q = insert_idx
    if wormhole == False:
        N = 2*n_one_side
        k = 5
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

        #Gamma_reconstructed, nu, eps_reconstructed = build_thermal_state_from_modular_hamiltonian(HL)

        #Gamma_TFD = gaussian_purification(Gamma_reconstructed)
        V = build_ring_potential(N//2, k, m_squared)
        
        Gamma_TFD = tfd_cov_ring_from_normal_modes(N//2, k, m_squared, V, beta=1, eps_omega=1e-15)

        t0 = 1.4


    else:
        # Parameters
        L = 4
        Lh = 3
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
        t0=4


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

    #Gamma_final = measure_left_side(Gamma_final,n_one_side)


    Gamma_out_real = 0.5 * (Gamma_teleported + Gamma_teleported.conj().T)
    return Gamma_final_observer, Gamma_final, Gamma_forward_observer, Gamma_forward


tube_lengths = np.arange(60)




print("stop")

