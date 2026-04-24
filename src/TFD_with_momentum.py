import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, expm, sqrtm, schur, block_diag, eigh, det
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




def construct_modular_hamiltonian_with_pinning(Gamma, epsilon_max=15, tol=1e-8):
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
            #eps = np.arctanh(1/(2*v))
            eps = 2 * np.arctanh(1/(2*v))

            epsilons.append(eps)
    
    E_diag = np.diag(np.repeat(epsilons, 2))
    # Modular Hamiltonian: K = S^{-T} E S^{-1}
    S_inv = inv(S)
    K = S_inv.T @ E_diag @ S_inv
    return K, epsilons, V




def left_side(Gamma):
    n =Gamma.shape[0]
    Gamma_left = np.zeros((n//2,n//2))
    Gamma_left[0:n//4,0:n//4]=Gamma[0:n//4,0:n//4]
    Gamma_left[n//4:n//2,n//4:n//2]=Gamma[n//2:3*n//4,n//2:3*n//4]
    Gamma_left[0:n//4,n//4:n//2]=Gamma[0:n//4,n//2:3*n//4]
    Gamma_left[n//4:n//2,0:n//4]=Gamma[n//2:3*n//4,0:n//4]
    return Gamma_left

def mutual_information(Gamma, idx_L, idx_R):
    S_L = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_L))
    S_R = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_R))
    S_LR = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_L + idx_R))
    return S_L + S_R - S_LR


def left_side(Gamma):
    n =Gamma.shape[0]
    Gamma_left = np.zeros((n//2,n//2))
    Gamma_left[0:n//4,0:n//4]=Gamma[0:n//4,0:n//4]
    Gamma_left[n//4:n//2,n//4:n//2]=Gamma[n//2:3*n//4,n//2:3*n//4]
    Gamma_left[0:n//4,n//4:n//2]=Gamma[0:n//4,n//2:3*n//4]
    Gamma_left[n//4:n//2,0:n//4]=Gamma[n//2:3*n//4,0:n//4]
    return Gamma_left

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

def pad_matrix_for_observer2(H_sys, observer_modes=1):
    """
    Pad a (2n x 2n) Hamiltonian matrix H_sys with observer_modes modes that evolve trivially.
    
    Parameters:
        H_sys : (2n x 2n) np.array
            Original system Hamiltonian (e.g. KL_full, KR_full, H)
        observer_modes : int
            Number of extra modes to add (default = 1)
    
    Returns:
        H_padded : (2(n + m) x 2(n + m)) np.array
            Extended Hamiltonian with zero blocks for observer modes
    """
    n_full = H_sys.shape[0]
    pad = 2 * observer_modes
    H_padded = np.zeros((n_full + pad, n_full + pad))
    H_padded[:n_full, :n_full] = H_sys
    return H_padded

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

def harmonic_chain_covariance(n, omega=1.0, g=1.0):
    """
    Construct the 2n x 2n ground state covariance matrix for an
    n-site 1D harmonic oscillator chain with nearest-neighbor coupling.
    """
    # Potential matrix V
    V = (omega**2 + 2*g) * np.eye(n)
    for i in range(n-1):
        V[i, i+1] = V[i+1, i] = -g

    # Covariance blocks
    cov_xx = 0.5 * sqrtm(np.linalg.inv(V))
    cov_pp = 0.5 * sqrtm(V)
    cov_xp = np.zeros((n, n))  # ⟨xp⟩ = 0 in ground state

    # Full covariance matrix in phase space ordering
    Gamma = np.block([
        [cov_xx, cov_xp],
        [cov_xp.T, cov_pp]
    ])

    return Gamma
def heisenberg_operator_evolution(K, t):
    """
    Evolves phase space operators under modular Hamiltonian K in Heisenberg picture.

    Parameters:
        K : (2n x 2n) modular Hamiltonian
        t : time

    Returns:
        S_t : (2n x 2n) Heisenberg evolution matrix such that r(t) = S_t @ r(0)
    """
    n = K.shape[0] // 2
    Omega = symplectic_form(n)
    S_t = expm(Omega @ K * t)
    return S_t

def describe_operator_spread(S_t, mode_index):
    """
    Prints how x_i(t) and p_i(t) spread over initial r(0) basis.

    Parameters:
        S_t : evolution matrix
        mode_index : index of mode (0 ≤ i < n)
    """
    n = S_t.shape[0] // 2
    labels = [f"x_{i}" for i in range(n)] + [f"p_{i}" for i in range(n)]

    x_vec = S_t[mode_index]
    p_vec = S_t[mode_index + n]

    print(f"\nHeisenberg evolution for x_{mode_index}(t):")
    for i, coeff in enumerate(x_vec):
        if np.abs(coeff) > 1e-4:
            print(f"  + {coeff:.4f} × {labels[i]}")

    print(f"\nHeisenberg evolution for p_{mode_index}(t):")
    for i, coeff in enumerate(p_vec):
        if np.abs(coeff) > 1e-4:
            print(f"  + {coeff:.4f} × {labels[i]}")

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
    im1 = axs[0].imshow(np.abs(coeffs_t[:, :n]),  aspect='auto', cmap='inferno', origin='lower')
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
    H = (1j * omega @ (v @ np.diag(np.arctanh(1.0 /(l.real))) @ np.linalg.inv(v))).real

    return H

def fidelity(V1,V2):
    n = V1.shape[0] // 2
    omega = symplectic_form(n)
    V_aux = omega.T @ inv(V1 + V2) @ (1/4 * omega + V2 @ omega @ V1)
    F_tot4 = det(2 * (sqrtm(np.eye(2*n)+1/4 * np.linalg.matrix_power(V_aux @ omega,-2))+np.eye(2*n)) @ V_aux)
    F_tot = F_tot4**.25
    F0 = F_tot/(det(V1+V2))**.25
    return F0



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

# Initial covariance matrix
#Gamma_0 = np.zeros((2*N_tot, 2*N_tot), dtype=np.complex128)
Gamma_0 =.5 * np.eye(2*N_tot,dtype=np.complex128)
#Gamma_0[np.ix_(un_set, un_set)] = mu_A * np.eye(len(un_set))
#Gamma_0[np.ix_(meas_set, meas_set)] = mu_B * np.eye(len(meas_set))


#################
""""
n = N_tot  # total number of modes
Gamma_1 = np.zeros((2*n, 2*n))

# Mode labels
un_set = np.sort(un_set)
meas_set = np.sort(meas_set)

# x and p indices
x_idx_un = un_set
p_idx_un = x_idx_un + n
x_idx_meas = meas_set
p_idx_meas = x_idx_meas + n

# Set variances for unmeasured modes (μ_A)
for xi, pi in zip(x_idx_un, p_idx_un):
    Gamma_1[xi, xi] = mu_A  # ⟨x²⟩
    Gamma_1[pi, pi] = mu_A  # ⟨p²⟩

# Set variances for measured modes (μ_B)
for xi, pi in zip(x_idx_meas, p_idx_meas):
    Gamma_1[xi, xi] = mu_B
    Gamma_1[pi, pi] = mu_B

# Optional: add small noise for numerical stability
Gamma_1 += 1e-12 * np.eye(2*n)

print(Gamma_1)
"""

############


# Number of total modes
n = N_tot

# Default: mass = 1, so kinetic term is identity
M = 3* np.eye(n)
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


"""
# Post-measurement via Schur complement
GammaAA = Gamma[np.ix_(un_set, un_set)]
GammaAB = Gamma[np.ix_(un_set, meas_set)]
GammaBA = Gamma[np.ix_(meas_set, un_set)]
GammaBB = Gamma[np.ix_(meas_set, meas_set)]

Gamma_post = GammaAA - GammaAB @ inv(GammaBB) @ GammaBA
Gamma_post = .5*(Gamma_post + Gamma_post.T)

cov_xx = 0.5 * inv(np.real(Gamma_post).astype(dtype=np.float64))
cov_pp = 0.5 * inv(np.real(inv(Gamma_post)).astype(dtype=np.float64))
cov_xp_sym = 1j * cov_xx @ Gamma_post.T - 0.5j * np.eye(len(un_set),dtype=np.complex128)
cov_growth_quench = np.block([
    [cov_xx, cov_xp_sym],
    [cov_xp_sym.conj().T, cov_pp]
])
"""

"""
# Quench
J = A_tot 
Gamma = Gamma_0 + 1j * J * t
"""

# Compute entropies
S_1, S_2, S_12 = [], [], []
for r in range(1, bdy_len):
    idx_1 = np.arange(r)
    idx_2 = np.arange(bdy_len, bdy_len + r)
    idx_both = np.concatenate([idx_1, idx_2])
    S_1.append(von_neumann_entropy_alt(extract_subsystem_covariance(Gamma_TFD, idx_1)))
    S_2.append(von_neumann_entropy_alt(extract_subsystem_covariance(Gamma_TFD, idx_2)))
    S_12.append(von_neumann_entropy_alt(extract_subsystem_covariance(Gamma_TFD, idx_both)))


# Plot
x_bdy = np.arange(1, bdy_len)
plt.figure(figsize=(6, 4))
plt.plot(x_bdy, S_1, 'k-', label='One Side')
plt.plot(x_bdy, S_12, 'r-', linewidth=2, label='Both Sides')
plt.xlabel("Segment Length")
plt.ylabel("Von Neumann Entropy")
plt.title("Entropy vs Segment Size (1-side and 2-side)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


b = bdy_len
keep = np.arange(b)  # keep left boundary
Gamma_reduced = trace_out_subsystem(Gamma_TFD, keep)

#KL = covmat_to_hamil(Gamma_reduced)
KL,epsilon,v = construct_modular_hamiltonian_with_pinning(Gamma_reduced)


############

"""
N = Gamma_TFD.shape[0]//2
HL = np.zeros((N,N))
for i in range(N):
    if i < N//4-1:
        HL[i,i+1] = 1
        HL[i+1, i] = 1
    if i == N//4-1:
        HL[i,0] = 1
        HL[0,i] = 1
    if i >N//4-1 and i <N//2-1:
        HL[i,i+1] = 1
        HL[i+1, i] = 1
    if i == N//2-1:
        HL[i,N//4] = 1
        HL[N//4,i] = 1   

    if i > N//2-1:
        HL[i,i] = 1
     
"""




HL = KL

"""
plt.imshow(np.abs(KL))
plt.colorbar()
plt.title("Wormhole Modular Hamiltonian")
plt.show()
"""
mode_indices = np.linspace(0,HL.shape[0],HL.shape[0])
plt.plot(mode_indices, eigh(HL)[0])
plt.title("Quench + Measure Modular Hamiltonian Eigenvalues")
plt.xlabel("quadrature")
plt.ylabel("eigenvalue")
plt.show()






###########
# investigate spreading
###########


S_t = heisenberg_operator_evolution(HL, t=5.0)
describe_operator_spread(S_t, mode_index=0)

t_list = np.linspace(0, 10, 100)  # 100 time steps from t=0 to t=10
coeffs_t = operator_spread_over_time(HL, t_list, op_index=0)  # evolve x_0(t)
#plot_light_cone(coeffs_t, title="Light Cone of $x_0(t)$")





t0 = 5



Gamma_left = left_side(Gamma_TFD)



n_left = Gamma_TFD.shape[0] // 4
Omega_left = symplectic_form(n_left)

S_left = expm(Omega_left @ HL * t0)
Gamma_evolved = S_left @ Gamma_left @ S_left.T

"""
plt.imshow(np.abs(Gamma_left))
plt.colorbar()
plt.title("Gamma_left")
plt.show()

plt.imshow(np.abs(Gamma_evolved))
plt.colorbar()
plt.title("Gamma_evolved")
plt.show()
"""




T = 60
times = np.linspace(0,t0,T)
dt = t0/T
S_dt = expm(Omega_left @ HL * dt)
k = 1

corr_xk_xj  = [] 
corr_pk_pj  = [] 

mutual_info_1 = []
mutual_info_2 = []
mutual_info_3 = []
mutual_info_4 = []

i_0 = 0
i_1 = 1
i_2 = 2
i_3 = 3
i_4 = 4

n_L = Gamma_TFD.shape[0]//4

Gamma_left = left_side(Gamma_TFD)
Gamma = Gamma_left

#Gamma_harmonic = harmonic_chain_covariance(64, omega=1.0, g=1.0)

#Gamma = Gamma_harmonic
"""
for t in enumerate(times):
    # equal-time covariance  |<x_k x_j>|  for light-cone picture
    Cxx = 0.5 * inv(np.real(Gamma))[:n_L, :n_L]      # only xx block
    Cpp = 0.5 * inv(np.real(Gamma))[n_L:2*n_L, n_L:2*n_L]      # only pp block
    corr_xk_xj.append( np.abs(Cxx[k, :]) )
    corr_pk_pj.append( np.abs(Cpp[k, :]) )
    mutual_info_1.append(mutual_information(Gamma, idx_L=[i_0], idx_R=[i_1]))
    mutual_info_2.append(mutual_information(Gamma, idx_L=[i_0], idx_R=[i_2]))
    mutual_info_3.append(mutual_information(Gamma, idx_L=[i_0], idx_R=[i_3]))
    mutual_info_4.append(mutual_information(Gamma, idx_L=[i_0], idx_R=[i_4]))

    # ---- propagate one step ------------------------------------------
    Gamma = S_dt @ Gamma @ S_dt.T


plt.plot(times,mutual_info_1, label = f"{i_0} and {i_1}")
plt.plot(times,mutual_info_2, label = f"{i_0} and {i_2}")
plt.plot(times,mutual_info_3, label = f"{i_0} and {i_3}")
plt.plot(times,mutual_info_4, label = f"{i_0} and {i_4}")

plt.xlabel("times")
plt.ylabel("mutual information")
plt.title(f"mutual information between sites")
plt.legend()
plt.show()
"""

"""
x_corr_matrix = np.array(corr_xk_xj)     # shape (t_steps, N)
plt.figure(figsize=(6,3.5))
plt.imshow(x_corr_matrix, aspect='auto', origin='lower',
           extent=(0, n_L-1, 0, t0))
plt.colorbar(label=r'$|\langle \hat x_k \hat x_j\rangle|$')
plt.xlabel('site $j$')
plt.ylabel('time')
plt.title(f'propagation of position correlations with site {k}')
plt.tight_layout()
plt.show()
"""
"""
p_corr_matrix= np.array(corr_pk_pj)     # shape (t_steps, N)
plt.figure(figsize=(6,3.5))
plt.imshow(p_corr_matrix,  aspect='auto', origin='lower',
           extent=(0, n_L-1, 0, t0))
plt.colorbar(label=r'$|\langle \hat x_k \hat x_j\rangle|$')
plt.xlabel('site $j$')
plt.ylabel('time')
plt.title(f'propagation of momentum correlations with site {k}')
plt.tight_layout()
plt.show()
"""


n = Gamma_TFD.shape[0] // 2


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

HR_full_padded = pad_matrix_for_observer(HR_full)
HL_full_padded = pad_matrix_for_observer(HL_full)

H_LR_padded = HL_full_padded+HR_full_padded


# Symplectic form
Omega = symplectic_form(n)

# Evolve backward in time

S_back = expm(-1 * Omega @ HL_full * t0)
Gamma_back = S_back @ Gamma_TFD @ S_back.T

T = 80
dt = t0/T


I_0 = mutual_information(Gamma_TFD, idx_L=list(range(n_L)), idx_R=list(range(n_L, n)))
I_mut = []

Gamma_LR = Gamma_TFD


times_back=np.linspace(0,t0,T)
S_back_dt = expm(-1 * Omega @ HL_full * dt)

for t in times_back:
    I = mutual_information(Gamma_LR, idx_L=list(range(n_L)), idx_R=list(range(n_L, n)))
    I_mut.append(I)
    Gamma_LR = S_back_dt @ Gamma_LR @ S_back_dt.T



###########
# insert quantum information on one side
###########




# define a squeezed input state
insert_idx = Gamma_LR.shape[0] // 8

teleported_idx = bdy_len + insert_idx # index 0 on right side starts here



Gamma_2mode = two_mode_squeezed_state(r=0.2)

Gamma_with_observer = insert_two_mode_state_direct_sum(Gamma_back, insert_idx, Gamma_2mode)


observer_idx = Gamma_with_observer.shape[0] // 2 - 1 # new mode is last


###
# set squeezing
###
s = -.5
theta = np.pi/2

Rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), -np.cos(theta)]])
Squeeze = 0.5 * np.array([[np.exp(-2*s), 0],
                          [0, np.exp(2*s)]])

Gamma_squeezed = Rot @ Squeeze @ Rot.T

Gamma_insert_wigner = insert_unentangled_mode(Gamma_back, insert_idx, Gamma_insert=Gamma_squeezed)




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
times_obs_forward = np.linspace(0,t0,T)

Gamma_LR = Gamma_with_observer
Gamma_LR_no_insert = Gamma_back
Gamma_LR_wigner = Gamma_insert_wigner

S_forward_no_insert = expm(Omega @ HL_full * dt)
S_forward_dt = expm(1 * Omega_padded @ HL_full_padded * dt)

for t in times_obs_forward:
    I_L = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L)))
    I_R = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L, n)))
    I_obs_L.append(I_L)
    I_obs_R.append(I_R)
    I = mutual_information(Gamma_LR, idx_L=list(range(n_L)), idx_R=list(range(n_L, n)))
    I_mut.append(I)
    I_tot.append(total_mutual_information_with_observer(Gamma_LR,n_total,observer_idx))
    I_insert.append(compute_MI_with_observer(Gamma_LR, observer_idx, [insert_idx]))
    Gamma_LR = S_forward_dt @ Gamma_LR @ S_forward_dt.T
    Gamma_LR_no_insert = S_forward_no_insert @ Gamma_LR_no_insert @ S_forward_no_insert.T
    Gamma_LR_wigner = S_forward_no_insert @ Gamma_LR_wigner @ S_forward_no_insert.T


initial_mut_info= []
for i in range(Gamma_LR.shape[0]//2-1):
    initial_mut_info.append(compute_MI_with_observer(Gamma_LR, observer_idx, [i]))

plt.plot(np.arange(Gamma_LR.shape[0]//2-1),initial_mut_info,color='k')
plt.axvline(insert_idx,color="blue",linestyle = "dashed")
plt.axvline(teleported_idx,color="red",linestyle="dashed")
plt.xlabel("site")
plt.ylabel("mutual info with observer")
plt.title("mutual information with observer before coupling")
plt.legend()
plt.show()



#######
# couple the two sides
#######

n_total = Gamma_TFD.shape[0] // 2

#N = n_total
# Global oscillator indices of left and right boundaries
bdy_len = 2**(L - 1)         # e.g. 128
bdy_1 = np.arange(N - bdy_len, N)               # left boundary: physical indices
bdy_2 = np.arange(N_tot - bdy_len, N_tot)       # right boundary: physical indices

# Map these physical indices into the post-measurement (Gamma_TFD) indexing
# You need to find where each bdy_1 and bdy_2 element lies in un_set
#lookup = {node: i for i, node in enumerate(un_set)}
#bdy_1_idx = np.array([lookup[x] for x in bdy_1])
#bdy_2_idx = np.array([lookup[x] for x in bdy_2])

bdy_1_idx = np.arange(bdy_len)
bdy_2_idx = np.arange(bdy_len,2*bdy_len)

#carrier_indices = np.arange(0, bdy_len)  # skip teleportation qubit

carrier_indices1 = np.arange(0,insert_idx)
carrier_indices2 = np.arange(insert_idx+1,bdy_len)
carrier_indices = np.concatenate((carrier_indices1,carrier_indices2))

def idx_x(j): return j
def idx_p(j): return j + n_total


H = np.zeros((2*n_total, 2*n_total))
mu = 1
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
    H[x_L, x_R] = H[x_R, x_L] = mu / 2
    # p coupling
    H[x_L + n_total, x_R + n_total] = H[x_R + n_total, x_L + n_total] = mu / 2





H_padded = pad_matrix_for_observer(H)
#H_padded+=H_LR_padded  




t_couple = 3.2
dt_couple = t_couple/T
S_coupling = expm(Omega_padded @ H_padded * t_couple)

Gamma_coupled = S_coupling @ Gamma_forward @ S_coupling.T



S_couple_no_insert = expm(Omega @ H * dt_couple)
S_couple_dt = expm(1 * Omega_padded @ H_padded * dt_couple)

times_obs_coupling = np.linspace(t0,t0+t_couple,T)
for t in times_obs_coupling:
    I_L = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L)))
    I_R = compute_MI_with_observer(Gamma_LR, observer_idx, list(range(n_L, n)))
    I_obs_L.append(I_L)
    I_obs_R.append(I_R)
    I = mutual_information(Gamma_LR, idx_L=list(range(n_L)), idx_R=list(range(n_L, n)))
    I_mut.append(I)
    I_tot.append(total_mutual_information_with_observer(Gamma_LR,n_total,observer_idx))
    I_insert.append(compute_MI_with_observer(Gamma_LR, observer_idx, [insert_idx]))
    Gamma_LR = S_couple_dt @ Gamma_LR @ S_couple_dt.T
    Gamma_LR_no_insert = S_couple_no_insert @ Gamma_LR_no_insert @ S_couple_no_insert.T
    Gamma_LR_wigner = S_couple_no_insert @ Gamma_LR_wigner @ S_couple_no_insert.T







######
# evolve state forwards in time with KR
######




S_final = expm(Omega_padded @ HR_full_padded * t0)
Gamma_final = S_final @ Gamma_coupled @ S_final.T



S_final_no_insert = expm(1 * Omega @ HR_full* dt)
times_obs_final = np.linspace(t0+t_couple,2*t0+t_couple,T)
for t in times_obs_forward:
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


times_bdy_info = np.concatenate((times_back,times_obs_forward+times_back[-1],times_obs_coupling+times_back[-1],times_obs_final+times_back[-1]))
#times = np.concatenate((times1,times2,times3))

plt.plot(times_bdy_info,I_mut,"k",label = "I_mut")
#plt.axvline(times_back[-1],color="green",label = "back")
#plt.axvline(times_obs_forward[-1]+times_back[-1],color="red",label = "forward")
#plt.axvline(times_obs_coupling[-1]+times_back[-1],color = "blue",label = "couple")
#plt.axvline(times_obs_final[-1]+times_back[-1],color = "orange",label = "final")

plt.xlabel("time")
plt.ylabel("mutual information")
plt.legend()
plt.show()




times = np.concatenate((times_obs_forward,times_obs_coupling,times_obs_final))

plt.plot(times,I_obs_L,"k",label = "mutual info with left")
plt.plot(times,I_obs_R,"r",label = "mutual info with right")
plt.plot(times,I_tot,"blue",label = "total mutual info")
plt.plot(times,I_insert,"green",label = "mutual info with insert")
plt.axvline(t0)


#plt.axvline(t0*2,label = "kick")
plt.xlabel("time")
plt.ylabel("mutual information")
plt.legend()
plt.show()


teleported_idx = bdy_len + insert_idx # index 0 on right side starts here



final_mut_info = []
for i in range(Gamma_LR.shape[0]//2-1):
    final_mut_info.append(compute_MI_with_observer(Gamma_LR, observer_idx, [i]))


plt.plot(np.arange(Gamma_LR.shape[0]//2-1),final_mut_info,color='k')
plt.axvline(insert_idx,color="blue",linestyle = "dashed")
plt.axvline(teleported_idx,color="red",linestyle ="dashed")
plt.xlabel("site")
plt.ylabel("mutual info with observer")
plt.title("mutual information with observer after coupling")
plt.legend()
plt.show()

mut_info_insert_regions = []
mut_info_telep_regions = []

lengths_array = np.linspace(1,Gamma_TFD.shape[0] // 8,Gamma_TFD.shape[0] // 8)
"""
for i in range(1,lengths_array.shape[0]):
    #if i == 0:
    #segment_telep = [teleported_idx]
    segment_insert = np.arange(insert_idx - i, insert_idx + i+1)
    segment_insert = np.ndarray.tolist(segment_insert)
    mut_info_insert_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, segment_insert))
    segment_telep = np.arange(teleported_idx - i, teleported_idx + i+1)
    segment_telep = np.ndarray.tolist(segment_telep)
    mut_info_telep_regions.append(compute_MI_with_observer(Gamma_LR, observer_idx, segment_telep))
"""

center_idx = 32

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
plt.axhline(compute_MI_with_observer(Gamma_LR,observer_idx,list(range(n_total))), color = "blue", label = "total mutual info with observer")
plt.xlabel("length of segment")
plt.ylabel("mutual info with observer")
plt.title("mutual information of segments centered around teleported site")
plt.legend()
plt.show()




Gamma_teleported = extract_mode_block(Gamma_LR_wigner, teleported_idx)
Gamma_teleported2 = extract_mode_block(Gamma_LR_no_insert, teleported_idx)




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
plot_wigner_ellipse(extract_mode_block(Gamma_squeezed,0), ax, label='Input', color='green')
plot_wigner_ellipse(Gamma_out_real, ax, label='Output', color='red')
plot_wigner_ellipse(Gamma_out_real2, ax, label='No Input', color='k')
#plot_wigner_ellipse(Gamma_out_shift, ax, label='No Input', color='orange')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("Position Quadrature")
ax.set_ylabel("Momentum Quadrature")
ax.set_aspect('equal')
ax.legend()
plt.title("Input vs Output Wigner Ellipses")
plt.grid(True)
plt.show()



f = fidelity(Gamma_squeezed,Gamma_out_real)
print("f=",f)



Gamma_input = extract_mode_block(Gamma_2mode,0)
Gamma_output = Gamma_out_real



fidelity = gaussian_fidelity_mixed(Gamma_input, Gamma_output)
print(f"Fidelity of teleportation: {fidelity:.4f}")

print("stop")

if np.sqrt(np.linalg.det(Gamma_input)) ==.5 and np.sqrt(np.linalg.det(Gamma_out_real)) == .5:
    S_rec = compute_recovery_transformation(Gamma_input,Gamma_out_real)
    Gamma_recovered = S_rec @ Gamma_out_real @ S_rec.T
    theta1, r, theta2, R1, S, R2 = decompose_symplectic_2x2(S_rec)
else:
    S_rec_approx, Gamma_recovered = closest_symplectic_approximation(Gamma_input, Gamma_out_real)
    theta1, r, theta2, R1, S, R2 = decompose_symplectic_2x2(S_rec_approx)

F = gaussian_fidelity_pure_input(Gamma_input, Gamma_recovered)

print("fidelity =",F)




print("stop")
