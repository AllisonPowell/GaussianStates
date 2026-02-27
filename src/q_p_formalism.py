import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, expm, sqrtm, schur, block_diag, eigh, det, polar
from thewalrus.symplectic import xpxp_to_xxpp, sympmat
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse


def symplectic_form(n):
    """Returns the 2n × 2n symplectic form Omega"""
    return np.block([
        [np.zeros((n, n),dtype=np.float64), np.eye(n,dtype=np.float64)],
        [-np.eye(n,dtype=np.float64), np.zeros((n, n),dtype=np.float64)]
    ])


def symplectic_eigenvalues(Gamma):
    """
    Compute the symplectic eigenvalues ν_i of a covariance matrix Γ.
    """
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * Omega @ Gamma)
    ν = np.sort(np.abs(eigvals))[::2]  # Take only one of each ν_i pair
    return ν


def get_derivatives(state, params):
    """
    Computes time derivatives for the system and the linearized deviations.
    state: Tuple (q, p, dq, dp)
           q, p   : Shape (M, N) -> M trajectories, N sites
           dq, dp : Shape (M, N) -> Deviations (tangent vectors)
    params: Dictionary containing physics parameters (omega, lam, J)
    """
    q, p, dq, dp = state
    omega = params['omega']
    lam = params['lam']  # The quartic interaction strength
    J = params['J']      # Coupling strength
    
    # 1. Classical Equations of Motion (Non-linear)
    # ---------------------------------------------
    # dq/dt = p
    q_dot = p
    
    # dp/dt = -dV/dq - Coupling
    # Neighbors (using np.roll for periodic/ring boundary conditions)
    coupling_force = J * (np.roll(q, 1, axis=1) + np.roll(q, -1, axis=1))
    
    # Force = -omega^2*q - 4*lambda*q^3 + coupling
    p_dot = -(omega**2) * q - 4 * lam * (q**3) + coupling_force
    
    # 2. Linearized Equations for Deviations (Jacobian evolution)
    # -----------------------------------------------------------
    # d(dq)/dt = dp
    dq_dot = dp
    
    # d(dp)/dt = -V''(q)*dq - Coupling*dq
    # The effective 'spring constant' depends on the current position q(t)!
    # V''(q) = omega^2 + 12*lambda*q^2
    curvature = omega**2 + 12 * lam * (q**2)
    
    coupling_dq = J * (np.roll(dq, 1, axis=1) + np.roll(dq, -1, axis=1))
    
    dp_dot = -curvature * dq + coupling_dq
    
    return (q_dot, p_dot, dq_dot, dp_dot)

def rk4_step(state, dt, params):
    """Standard Runge-Kutta 4 integrator."""
    k1 = get_derivatives(state, params)
    
    # Helper to add k*factor to state
    def add(s, k, factor):
        return tuple(s[i] + k[i] * factor for i in range(4))
    
    state_k1 = add(state, k1, 0.5 * dt)
    k2 = get_derivatives(state_k1, params)
    
    state_k2 = add(state, k2, 0.5 * dt)
    k3 = get_derivatives(state_k2, params)
    
    state_k3 = add(state, k3, dt)
    k4 = get_derivatives(state_k3, params)
    
    # Combine
    new_state = tuple(
        state[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        for i in range(4)
    )
    return new_state

def get_derivatives_masked_old(state, params, mask):
    """
    state: (q, p) tuple. 
           Deviations (dq, dp) are omitted for clarity, but logic is identical.
    mask:  (2*N,) array. 1.0 for active sites, 0.0 for frozen sites.
    """
    q, p = state
    m2 = params['m_squared']
    k = params['k_coupling']
    lam = params['lam']
    
    # --- Forces (Physics) ---
    # 1. Harmonic term
    force_harmonic = -(omega**2) * q
    
    # 2. Quartic term (The Source of Scrambling)
    force_quartic = -4 * lam * (q**3)
    
    # 3. Nearest Neighbor Coupling (intra-ring)
    # We must handle boundaries carefully. We assume q is [q_L1...q_LN, q_R1...q_RN]
    # We can compute coupling globally, but the mask will kill it for frozen sites.
    # Note: This assumes L and R are not coupled by J, only internally.
    
    # Reshape to (M, 2, N_site) to handle L/R neighbors separately easily
    q_reshaped = q.reshape(q.shape[0], 2, N_site)
    
    # Roll along the site axis (axis 2)
    neighbors = np.roll(q_reshaped, 1, axis=2) + np.roll(q_reshaped, -1, axis=2)
    coupling_force = J * neighbors.reshape(q.shape) # Flatten back
    
    # Total Force
    p_dot_raw = force_harmonic + force_quartic + coupling_force
    q_dot_raw = p
    
    # --- APPLY MASK ---
    # If mask is 0, derivatives are 0 -> variables don't change
    q_dot = q_dot_raw * mask
    p_dot = p_dot_raw * mask
    
    return (q_dot, p_dot)


def get_derivatives_masked_observer_old(state, params, mask):
    q, p = state
    N = params['N_site']
    M = q.shape[0]
    
    # 1. Separate the Rings from the Observer
    # Rings are indices 0 to 2N-1. Observer is at index 2N.
    q_rings = q[:, :2*N]
    p_rings = p[:, :2*N]
    
    # 2. Reshape only the rings for the roll operation
    q_reshaped = q_rings.reshape(M, 2, N)
    
    # 3. Calculate Neighbor Forces for the rings only
    # axis=2 ensures we only roll within each individual ring
    neighbors = np.roll(q_reshaped, 1, axis=2) + np.roll(q_reshaped, -1, axis=2)
    ring_coupling_force = params['J'] * neighbors.reshape(M, 2*N)
    
    # 4. Reconstruct the full p_dot (Forces)
    # We add 0.0 force for the observer at the end
    p_dot_rings = -(params['omega']**2) * q_rings - 4 * params['lam'] * (q_rings**3) + ring_coupling_force
    p_dot_observer = np.zeros((M, 1)) # The observer feels no physical forces
    
    p_dot_raw = np.hstack([p_dot_rings, p_dot_observer])
    q_dot_raw = p # q_dot is always p (velocity)
    
    # 5. Apply Mask (The safety lock)
    q_dot = q_dot_raw * mask
    p_dot = p_dot_raw * mask
    
    return (q_dot, p_dot)

def get_derivatives_masked_observer(state, params, mask):
    q, p = state
    N = params['N_site']
    M = q.shape[0]
    
    # Slice rings and observer
    q_rings = q[:, :2*N]
    q_reshaped = q_rings.reshape(M, 2, N)
    
    m2 = params['m_squared']
    k = params['k_coupling']
    lam = params['lam']

    # New Coupling Force: k * (q_{j-1} + q_{j+1} - 2*q_j)
    # The -2*q_j term is the "renormalization" from the springs
    sum_neighbors = np.roll(q_reshaped, 1, axis=2) + np.roll(q_reshaped, -1, axis=2)
    coupling_force = k * (sum_neighbors - 2 * q_reshaped)
    coupling_force = coupling_force.reshape(M, 2*N)
    
    # On-site Force: -m^2*q - 4*lambda*q^3
    p_dot_rings = -(m2 * q_rings) - 4 * lam * (q_rings**3) + coupling_force
    
    # Pad for observer
    p_dot_raw = np.hstack([p_dot_rings, np.zeros((M, 1))])
    q_dot_raw = params["momentum"]*p
    
    return (q_dot_raw * mask, p_dot_raw * mask)

def get_derivatives_masked(state, params, mask):
    q, p = state
    N = params['N_site']
    M = q.shape[0]
    
    # Slice rings and observer
    q_rings = q[:, :2*N]
    q_reshaped = q_rings.reshape(M, 2, N)
  
    m2 = params['m_squared']
    k = params['k_coupling']
    lam = params['lam']

    # New Coupling Force: k * (q_{j-1} + q_{j+1} - 2*q_j)
    # The -2*q_j term is the "renormalization" from the springs
    sum_neighbors = np.roll(q_reshaped, 1, axis=2) + np.roll(q_reshaped, -1, axis=2)
    coupling_force = k * (sum_neighbors - 2 * q_reshaped)
    coupling_force = coupling_force.reshape(M, 2*N)
    
    # On-site Force: -m^2*q - 4*lambda*q^3
    p_dot_rings = -(m2 * q_rings) - 4 * lam * (q_rings**3) + coupling_force

    p_dot_raw = p_dot_rings
    q_dot_raw = params["momentum"]*p

    return (q_dot_raw * mask, p_dot_raw * mask)


# --- 2. RK4 Step handling the mask ---
def rk4_step_masked(state, dt, params, mask):
    # Standard RK4, just passing the mask down
    k1 = get_derivatives_masked(state, params, mask)
    
    def add(s, k, factor):
        return tuple(s[i] + k[i] * factor for i in range(len(s)))
    
    state_k1 = add(state, k1, 0.5 * dt)
    k2 = get_derivatives_masked(state_k1, params, mask)
    
    state_k2 = add(state, k2, 0.5 * dt)
    k3 = get_derivatives_masked(state_k2, params, mask)
    
    state_k3 = add(state, k3, dt)
    k4 = get_derivatives_masked(state_k3, params, mask)
    
    new_state = tuple(
        state[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        for i in range(2)
    )
    return new_state

def rk4_step_masked_observer(state, dt, params, mask):
    # Standard RK4, just passing the mask down
    k1 = get_derivatives_masked_observer(state, params, mask)
    
    def add(s, k, factor):
        return tuple(s[i] + k[i] * factor for i in range(len(s)))
    
    state_k1 = add(state, k1, 0.5 * dt)
    k2 = get_derivatives_masked_observer(state_k1, params, mask)
    
    state_k2 = add(state, k2, 0.5 * dt)
    k3 = get_derivatives_masked_observer(state_k2, params, mask)
    
    state_k3 = add(state, k3, dt)
    k4 = get_derivatives_masked_observer(state_k3, params, mask)
    
    new_state = tuple(
        state[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        for i in range(2)
    )
    return new_state





def apply_coupling(q, p, g, N):
    """
    Apply instantaneous coupling U = exp(i * g * Sum(qL * qR))
    This shifts momentum: p -> p + g * q_partner
    """
    # q and p shape: (M, 2*N) where first N are Left, last N are Right
    q_L = q[:, :N]
    q_R = q[:, N:]
    
    # Update momenta
    # Note: The sign depends on your definition of H_int. 
    # Usually strictly attractive interaction is required for teleportation (g > 0).
    p[:, :N] += g * q_R  # Kick to Left momenta
    p[:, N:] += g * q_L  # Kick to Right momenta
    
    return q, p

def apply_LR_coupling_kick_obs_safe(state, g, N):
    """
    Instantaneous coupling U = exp(i g Σ_i qLi qRi)  (up to sign conventions).
    This induces:
      pL <- pL + g*qR
      pR <- pR + g*qL
    Observer unchanged.
    """
    q, p = state
    L = slice(0, N)
    R = slice(N, 2*N)

    qL = q[:, L]
    qR = q[:, R]

    p_new = p.copy()
    p_new[:, L] += g * qR
    p_new[:, R] += g * qL

    return (q, p_new)

def compute_entropy_from_samples_old(q_samples, p_samples):
    """
    Computes Von Neumann entropy of the Gaussian approximation 
    of the sample cloud.
    """
    if len(q_samples.shape)==1:
        N_sub = 1
    else:
        N_sub = q_samples.shape[1]

  
    # 1. Construct the Covariance Matrix (2N x 2N)
    # Vector X = [q1, ..., qN, p1, ..., pN]
    # We want Cov_ij = < {X_i - <X_i>, X_j - <X_j>} >
    
    # Stack q and p: Shape (M, 2*N_sub)
    X = np.hstack([q_samples, p_samples])
    
    # Compute covariance using numpy (rowvar=False means columns are variables)
    sigma = np.cov(X, rowvar=False) 
    
    # 2. Symplectic Eigenvalues
    # We need the eigenvalues of i * Omega * Sigma
    # Omega = [[0, I], [-I, 0]]
    N = N_sub
    Omega = np.block([
        [np.zeros((N, N)), np.eye(N)],
        [-np.eye(N), np.zeros((N, N))]
    ])
    
    # Matrix to diagonalize: i * Omega * Sigma
    mat = 1j * np.dot(Omega, sigma)
    
    # The eigenvalues will come in pairs +/- nu. We take the positive ones.
    eigvals = np.linalg.eigvals(mat)
    nus = np.abs(eigvals.imag)[::2] # Take every second one (pairs)
    
    # 3. Entropy Formula
    # Handle numerical instability if nu is exactly 0.5 (pure state)
    nus = np.maximum(nus, 0.5 + 1e-9) 
    
    term1 = (nus + 0.5) * np.log(nus + 0.5)
    term2 = (nus - 0.5) * np.log(nus - 0.5)
    entropy = np.sum(term1 - term2)
    
    return entropy



def compute_entropy_from_samples(q_samples, p_samples,indices):
    """
    Computes Von Neumann entropy of the Gaussian approximation 
    of the sample cloud.
    """
  
    # 1. Construct the Covariance Matrix (2N x 2N)
    # Vector X = [q1, ..., qN, p1, ..., pN]
    # We want Cov_ij = < {X_i - <X_i>, X_j - <X_j>} >
    
    # Stack q and p: Shape (M, 2*N_sub)
    X = np.hstack([q_samples, p_samples])
    
    # Compute covariance using numpy (rowvar=False means columns are variables)
    sigma = np.cov(X, rowvar=False) 

    return(von_neumann_entropy_alt(extract_subsystem_covariance(sigma,indices)))

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
    eigvals = np.linalg.eigvals(1j * Omega @ Gamma)
    nu = np.sort(np.abs(eigvals))[::2]
    nu = np.clip(nu, 0.500001, None)
    return sum((nu + 0.5)*np.log(nu + 0.5) - (nu - 0.5)*np.log(nu - 0.5))


def get_mutual_info_LR(q, p, N):
    # Split into L and R
    q_L, p_L = q[:, :N], p[:, :N]
    q_R, p_R = q[:, N:], p[:, N:]
    
    S_L = compute_entropy_from_samples(q_L, p_L)
    S_R = compute_entropy_from_samples(q_R, p_R)
    S_LR = compute_entropy_from_samples(q, p) # Global entropy
    
    return S_L + S_R - S_LR

def mut_info_observer_old(q, p, idx_0, idx_f, idx_obs):
    # Split into L and R
    q_a, p_a = q[:,idx_0:idx_f], p[:, idx_0:idx_f]
    q_obs, p_obs = q[:,idx_obs], p[:, idx_obs]
    
    S_a = compute_entropy_from_samples(q_a, p_a)
    S_obs = compute_entropy_from_samples(q_obs, p_obs)
    S_joint = compute_entropy_from_samples(q, p) # Global entropy
    
    return S_a + S_obs - S_joint

def mut_info_observer(q, p, idx_0, idx_f, idx_obs):
    # Split into L and R

    indices_a = list(np.arange(idx_0,idx_f))

    S_a = compute_entropy_from_samples(q, p, indices_a)
    S_obs = compute_entropy_from_samples(q, p, [idx_obs])
    S_joint = compute_entropy_from_samples(q, p,indices_a+[idx_obs]) # Global entropy
    
    return S_a + S_obs - S_joint



def compute_fidelity(q_R_samples, p_R_samples, target_q, target_p, sigma_vac):
    """
    Computes fidelity with a target Coherent State defined by (target_q, target_p).
    q_R_samples: (M, 1) array of position samples for the specific site being teleported
    """
    # Wigner function of the target coherent state (Gaussian)
    # W(q, p) = (1/pi) * exp( - (q-q0)^2/sigma^2 - (p-p0)^2 * sigma^2 ) assuming vacuum width
    
    # Assuming units where vacuum variance is 0.5 (sigma_q^2 = 0.5)
    # The prefactor for normalized Wigner is 1/pi
    
    delta_q = q_R_samples - target_q
    delta_p = p_R_samples - target_p
    
    # Exponent for Gaussian Wigner function
    # Note: Check your units. If vacuum variance is sigma^2:
    exponent = -(delta_q**2 + delta_p**2) # Simplified for symmetric vacuum
    
    w_values = (1.0 / np.pi) * np.exp(exponent)
    
    # Average over trajectories and multiply by 2*pi (phase space factor)
    fidelity = (2 * np.pi) * np.mean(w_values)
    
    return fidelity


def sample_tfd_state(cov_mat, M, N_site):
    """
    Sample from the TFD covariance matrix correctly preserving correlations.
    
    Parameters:
    cov_mat : (4*N_site, 4*N_site) array
              Covariance matrix in basis [q_L, q_R, p_L, p_R] or [all_q, all_p]
    M       : int
              Number of trajectories (samples)
    N_site  : int
              Number of sites in ONE ring (Total modes = 2 * N_site)
              
    Returns:
    q : (M, 2*N_site) array
    p : (M, 2*N_site) array
    """
    
    # 1. Define the total dimension
    # If you have 2 rings of N sites, total phase space dim is 4*N
    dim = cov_mat.shape[0]
    
    # 2. Define the mean (TFD and Vacuum are centered at 0)
    mean_vector = np.zeros(dim)
    
    # 3. Sample from the Multivariate Normal Distribution
    # This is the crucial step that handles all q-p and L-R correlations
    samples = np.random.multivariate_normal(mean_vector, cov_mat, size=M)
    # samples shape is now (M, 4*N_site)
    
    # 4. Unpack based on your specific ordering
    # You stated your ordering is q1...qN_total, p1...pN_total.
    # So the first half of columns are Qs, second half are Ps.
    
    total_sites = 2 * N_site # Total number of q's
    
    q_samples = samples[:, :total_sites]  # First half
    p_samples = samples[:, total_sites:]  # Second half
    
    return q_samples, p_samples

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


def tfd_cov(N,k,m_squared):  
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

    Gamma_reconstructed, nu, eps_reconstructed = build_thermal_state_from_modular_hamiltonian(HL)

    Gamma_TFD = gaussian_purification(Gamma_reconstructed)

    return(Gamma_TFD)

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

def insert_observer_twa(state, N_site, insert_idx, Gamma_2mode):
    """
    state: Tuple (q, p) each shape (M, 2*N_site)
    Gamma_2mode: (4, 4) Covariance matrix of the TMS state
    """
    q, p = state
    M = q.shape[0]
    
    # 1. Sample the 2-mode squeezed state (Signal + Observer)
    # Gamma_2mode basis: [q_sig, q_obs, p_sig, p_obs]
    tms_samples = np.random.multivariate_normal(np.zeros(4), Gamma_2mode, size=M)
    
    q_sig = tms_samples[:, 0]
    q_obs = tms_samples[:, 1]
    p_sig = tms_samples[:, 2]
    p_obs = tms_samples[:, 3]
    
    # 2. Replace the Signal site trajectories
    # We are physically "removing" the old L_0 state and putting in a new one
    q_new = q.copy()
    p_new = p.copy()
    
    q_new[:, insert_idx] = q_sig
    p_new[:, insert_idx] = p_sig
    
    # 3. Append the Observer mode (which doesn't evolve)
    # We add it as the very last column
    q_expanded = np.hstack([q_new, q_obs.reshape(-1, 1)])
    p_expanded = np.hstack([p_new, p_obs.reshape(-1, 1)])
    
    return (q_expanded, p_expanded)

import numpy as np

def ring_force(q_ring, m2, k, lam):
    """
    q_ring: (M, N) for ONE ring.
    Returns F(q) = dp/dt for that ring under:
      V = 1/2 m2 Σ q^2 + (k/2) Σ (q_i - q_{i+1})^2 + lam Σ q^4
    """
    q_left  = np.roll(q_ring,  1, axis=1)
    q_right = np.roll(q_ring, -1, axis=1)
    spring = k * (q_left + q_right - 2.0*q_ring)
    return -(m2*q_ring) - 4.0*lam*(q_ring**3) + spring

def step_verlet_LR(state, dt, params, evolve_left: bool, evolve_right: bool):
    """
    Symplectic step for two decoupled rings.
    state: (q, p), each shape (M, 2N) with ordering [L(0..N-1), R(0..N-1)].
    """
    q, p = state
    N = params["N_site"]
    m2 = params["m_squared"]
    k  = params["k_coupling"]
    lam = params["lam"]

    qL, qR = q[:, :N], q[:, N:]
    pL, pR = p[:, :N], p[:, N:]

    # --- half kick ---
    if evolve_left:
        pL_half = pL + 0.5*dt*ring_force(qL, m2, k, lam)
    else:
        pL_half = pL

    if evolve_right:
        pR_half = pR + 0.5*dt*ring_force(qR, m2, k, lam)
    else:
        pR_half = pR

    # --- drift ---
    if evolve_left:
        qL_new = qL + dt*pL_half
    else:
        qL_new = qL

    if evolve_right:
        qR_new = qR + dt*pR_half
    else:
        qR_new = qR

    # --- second half kick ---
    if evolve_left:
        pL_new = pL_half + 0.5*dt*ring_force(qL_new, m2, k, lam)
    else:
        pL_new = pL_half

    if evolve_right:
        pR_new = pR_half + 0.5*dt*ring_force(qR_new, m2, k, lam)
    else:
        pR_new = pR_half

    q_new = np.hstack([qL_new, qR_new])
    p_new = np.hstack([pL_new, pR_new])
    return (q_new, p_new)

def step_verlet_LR_obs_safe(state, dt, params, evolve_left: bool, evolve_right: bool):
    """
    Symplectic velocity-Verlet step for two decoupled rings + frozen observer.

    Ordering in q and p:
      q = [qL(0..N-1), qR(0..N-1), qobs]
      p = [pL(0..N-1), pR(0..N-1), pobs]

    state: (q, p) with shapes (M, 2N+1).
    """
    q, p = state
    N = params["N_site"]
    m2 = params["m_squared"]
    k  = params["k_coupling"]
    lam = params["lam"]

    # slices
    L = slice(0, N)
    R = slice(N, 2*N)
    obs = 2*N

    # --- half kick ---
    p_half = p.copy()

    if evolve_left:
        p_half[:, L] += 0.5 * dt * ring_force(q[:, L], m2, k, lam)
    if evolve_right:
        p_half[:, R] += 0.5 * dt * ring_force(q[:, R], m2, k, lam)

    # observer untouched:
    # p_half[:, obs] unchanged

    # --- drift ---
    q_new = q.copy()
    if evolve_left:
        q_new[:, L] += dt * p_half[:, L]
    if evolve_right:
        q_new[:, R] += dt * p_half[:, R]

    # observer untouched:
    # q_new[:, obs] unchanged

    # --- second half kick ---
    p_new = p_half.copy()
    if evolve_left:
        p_new[:, L] += 0.5 * dt * ring_force(q_new[:, L], m2, k, lam)
    if evolve_right:
        p_new[:, R] += 0.5 * dt * ring_force(q_new[:, R], m2, k, lam)

    # observer untouched:
    # p_new[:, obs] unchanged

    return (q_new, p_new)



def empirical_cov_qp(q, p, reg=1e-8):
    """
    q,p: (M, total_modes) in your ordering [qL,qR,qobs] and [pL,pR,pobs]
    Returns Gamma in [q_all, p_all] ordering, regularized.
    """
    X = np.hstack([q, p])
    Gamma = np.cov(X, rowvar=False)
    Gamma = 0.5*(Gamma + Gamma.T)
    Gamma += reg * np.eye(Gamma.shape[0])
    return Gamma

def fidelity_map_twa(q, p, V_in, mode_indices):
    Gamma_emp = empirical_cov_from_samples(q,p)
    return np.array([gaussian_fidelity_mixed(V_in, mode_block_qp(Gamma_emp, j))
                     for j in mode_indices])

def apply_quadratic_coupling_exact(state, G, tau):
    """
    state: (q, p) with shapes (M, n_total)
           ordering q=[all x], p=[all p]
    G: (2*n_total, 2*n_total) symmetric matrix defining H = 1/2 R^T G R
       in ordering R=[x_all, p_all]
    tau: coupling duration
    """
    q, p = state
    M, n_total = q.shape
    assert p.shape == (M, n_total)
    assert G.shape == (2*n_total, 2*n_total)

    Omega = symplectic_form(n_total)
    A = Omega @ G                       # generator
    S = expm(tau * A)                   # symplectic map

    R = np.hstack([q, p])               # (M, 2*n_total)
    R_new = R @ S.T                     # apply to row-vectors
    q_new = R_new[:, :n_total]
    p_new = R_new[:, n_total:]
    return (q_new, p_new)


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
    


def mi_ksg(X, Y, k=5):
    # X: (M, dx), Y: (M, dy)
    # Uses max-norm; standard KSG estimator
    M = X.shape[0]
    XY = np.hstack([X, Y])

    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(XY)
    dists, _ = nbrs.kneighbors(XY)
    eps = dists[:, k] - 1e-15

    nx = NearestNeighbors(metric='chebyshev').fit(X)
    ny = NearestNeighbors(metric='chebyshev').fit(Y)

    nx_count = np.array([len(nx.radius_neighbors([X[i]], eps[i], return_distance=False)[0]) - 1 for i in range(M)])
    ny_count = np.array([len(ny.radius_neighbors([Y[i]], eps[i], return_distance=False)[0]) - 1 for i in range(M)])

    from scipy.special import digamma
    return digamma(k) + digamma(M) - np.mean(digamma(nx_count + 1) + digamma(ny_count + 1))

def mut_info_observer_ksg(q, p, idx_0, idx_f, idx_obs, k=5):
    A = np.hstack([q[:, idx_0:idx_f], p[:, idx_0:idx_f]])
    O = np.hstack([q[:, [idx_obs]], p[:, [idx_obs]]])
    return mi_ksg(A, O, k=k)

def signal_map_vs_observer(q, p, idx_obs, exclude_obs=True):
    """
    Returns an array scores[j] = ||Cov([q_j,p_j],[q_obs,p_obs])||_F^2
    for j over system sites (L and R). Fast.
    """
    M, n_total = q.shape

    XO = np.hstack([q[:, [idx_obs]], p[:, [idx_obs]]])  # (M,2)
    XO = XO - XO.mean(axis=0, keepdims=True)

    last = n_total-1 if exclude_obs else n_total
    scores = np.zeros(last, dtype=np.float64)

    for j in range(last):
        Xj = np.hstack([q[:, [j]], p[:, [j]]])          # (M,2)
        Xj = Xj - Xj.mean(axis=0, keepdims=True)
        C = (Xj.T @ XO) / M                             # (2,2)
        scores[j] = np.sum(C*C)

    return scores

def ring_energy(q_ring, p_ring, m2, k):
    q_next = np.roll(q_ring, -1, axis=1)
    spring = 0.5*k*np.sum((q_next - q_ring)**2, axis=1)
    onsite = 0.5*m2*np.sum(q_ring**2, axis=1)
    kin = 0.5*np.sum(p_ring**2, axis=1)
    return kin + onsite + spring

def total_energy_LR_obs(q, p, N, m2, k):
    qL,qR = q[:, :N], q[:, N:2*N]
    pL,pR = p[:, :N], p[:, N:2*N]
    return ring_energy(qL,pL,m2,k) + ring_energy(qR,pR,m2,k)

def energy_quadratic_from_samples(q, p, G):
    """
    q,p: (M, n_total) in [x_all], [p_all]
    G: (2*n_total, 2*n_total)
    returns sample-mean of 1/2 R^T G R
    """
    R = np.hstack([q, p])              # (M, 2n)
    GR = R @ G                         # (M, 2n)
    E = 0.5 * np.mean(np.sum(R * GR, axis=1))
    return float(E)

def step_coupling_window_twa(state, dt, params, G_cpl_obs):
    # 1) half step nonlinear ring dynamics
    state = step_verlet_LR_obs_safe(state, 0.5*dt, params, evolve_left=True, evolve_right=True)

    # 2) exact linear coupling map
    state = apply_quadratic_coupling_exact(state, G=G_cpl_obs, tau=dt)

    # 3) half step nonlinear ring dynamics
    state = step_verlet_LR_obs_safe(state, 0.5*dt, params, evolve_left=True, evolve_right=True)
    return state


import numpy as np

def wigner_overlap_with_gaussian_target(q, p, V_target, d_target=None):
    """
    Estimate Tr(rho_out rho_target) using Monte Carlo over output Wigner samples (q,p).

    q, p: arrays of shape (M,) or (M,1) for ONE mode
    V_target: 2x2 covariance of target in (q,p) ordering (vacuum is 0.5*I)
    d_target: length-2 mean [q0, p0] (default 0)
    Returns: overlap in [0,1] if target is pure and out is physical.
    """
    q = np.asarray(q).reshape(-1)
    p = np.asarray(p).reshape(-1)
    M = q.shape[0]
    z = np.stack([q, p], axis=1)  # (M,2)

    if d_target is None:
        d = np.zeros(2)
    else:
        d = np.asarray(d_target).reshape(2)

    V = 0.5*(V_target + V_target.T)
    Vinv = np.linalg.inv(V)
    detV = np.linalg.det(V)

    dz = z - d[None, :]
    # quadratic form (z-d)^T Vinv (z-d) for each sample
    quad = np.einsum("bi,ij,bj->b", dz, Vinv, dz)

    # Tr(rho_out rho_tar) = average[ (1/sqrt(detV)) * exp(-0.5*quad) ]
    overlap = np.mean((1.0/np.sqrt(detV)) * np.exp(-0.5*quad))
    return float(overlap)


# --- Simulation Setup ---

# System Parameters
N = 10           # Number of sites in the ring
M = 2000        # Number of trajectories (samples)
params = {'m_squared': 13, 'k_coupling': 5.0, "momentum":1,'lam': 0.001, 'N_site': N}
T = 4         # Total simulation time
dt = .005        # Time step
steps = int(T / dt)
t_couple = 3
steps_coupling = int(t_couple/dt)

# Masks
# shape (M, 2*N) -> Broadcastable
mask_L = np.zeros((1, 2 * N))
mask_L[0, :N] = 1.0  # Activate Left

mask_L_observer = np.zeros((1, 2 * N+1))
mask_L_observer[0, :N] = 1.0  # Activate Left

mask_R = np.zeros((1, 2 * N))
mask_R[0, N:] = 1.0  # Activate Right

mask_R_observer = np.zeros((1, 2 * N+1))
mask_R_observer[0, N:2*N] = 1.0  # Activate Right

mask_All = np.ones((1, 2 * N))

#params = {
#    'omega': 1.0,
#   'lam': 0.5,  # Non-zero lambda turns on Chaos/Scrambling
#    'J': 0.5     # Coupling between sites
#}

# 1. Initialize State (Sampling from Vacuum/Gaussian)
# For a real TFD, you would sample from your Covariance Matrix here using np.random.multivariate_normal
# Here we use a simple uncorrelated vacuum approximation for demonstration

Gamma_TFD = tfd_cov(N,k=params["k_coupling"],m_squared=params["m_squared"])
q,p = sample_tfd_state(Gamma_TFD, M, N)
state = (q, p)




print("1. Evolving LEFT Backward...")
for k in range(steps):
    # dt is NEGATIVE
    #state = rk4_step_masked(state, -dt, params, mask_L)
    state = step_verlet_LR(state, -dt, params, evolve_left=True, evolve_right=False)

dq = np.zeros(q.shape)
dp = np.zeros(p.shape)
dp[:, 0] = 1.0  # Initial perturbation at site 0

# Store OTOC data
otoc_history = np.zeros((2*steps,2*N))
sites_to_watch = [0, 1, 5, 9, 10] # Watch origin, neighbor, and far sites

print("2. Inserting Information...")
# Example: Inject a momentum kick (displacement) at Left Site 0
# Operator V = exp(i * epsilon * q_L0) -> shifts p_L0
epsilon = 0.5
insert_idx = 1
#state[1][:, insert_idx] += epsilon


Gamma_2mode = two_mode_squeezed_state(r=1)
state_with_observer_0 = insert_observer_twa(state, N, insert_idx, Gamma_2mode)
state_with_observer = insert_observer_twa(state, N, insert_idx, Gamma_2mode)
q0,p0=state_with_observer_0
#print(mut_info_observer_ksg(q0, p0, 0, N, 2*N, k=5))

#q_insert,p_insert = state

#target_p = p_insert.copy()

mut_info_L = []
mut_info_R = []
tot_mut_info_obs = []
total_energy = []

scores_array = np.zeros((2*steps+steps_coupling,2*N))


print("3. Evolving LEFT Forward...")
for s in range(steps):
    # dt is POSITIVE
    #state = rk4_step_masked(state, dt, params, mask_L)
    #state_with_observer = rk4_step_masked_observer(state_with_observer, dt, params, mask_L_observer)
    state_with_observer = step_verlet_LR_obs_safe(state_with_observer, dt, params, evolve_left=True, evolve_right=False)
    q_obs,p_obs = state_with_observer
    scores = signal_map_vs_observer(q_obs, p_obs, 2*N)
    scores_array[s,:] = scores/np.linalg.norm(scores) 
    total_energy.append(sum(total_energy_LR_obs(q_obs,p_obs,N,params["m_squared"],params["k_coupling"]))/M)
    #mut_info_L.append(mut_info_observer_ksg(q_obs, p_obs, 0, N, 2*N, k=5))
    #mut_info_R.append(mut_info_observer_ksg(q_obs, p_obs, N, 2*N, 2*N, k=5))    
    mut_info_L.append(mut_info_observer(q_obs, p_obs, 0, N, 2*N))
    mut_info_R.append(mut_info_observer(q_obs, p_obs, N, 2*N, 2*N))
    tot_mut_info_obs.append(mut_info_observer(q_obs, p_obs, 0, 2*N, 2*N))    
    #current_dq = get_derivatives_masked(state,params,mask_L)[0]    
    #otoc_vals = np.mean(current_dq**2, axis=0) # Shape (N,)
    #otoc_history[s,:]=otoc_vals




print("4. Coupling Left and Right...")


carrier_indices1 = np.arange(0,insert_idx)
carrier_indices2 = np.arange(insert_idx+1,N)
carrier_indices = np.concatenate((carrier_indices1,carrier_indices2))

bdy_1_idx = np.arange(N)
bdy_2_idx = np.arange(N,2*N)

n_total = 2*N
H_coupling = np.zeros((2*n_total, 2*n_total))
mu = 1

omega0 = np.sqrt(params["m_squared"] + 2*params["k_coupling"])


for j in carrier_indices:
    x_L = bdy_1_idx[j]
    x_R = bdy_2_idx[j]
    # x coupling
    H_coupling[x_L, x_R] = H_coupling[x_R, x_L] = mu * omega0/ 2
    # p coupling
    H_coupling[x_L + n_total, x_R + n_total] = H_coupling[x_R + n_total, x_L + n_total] = mu / (2*omega0)


#state_with_observer = apply_quadratic_coupling_exact(state=state_with_observer, G=H_coupling_obs, tau = 3)



k = params["k_coupling"]
m_squared = params["m_squared"]

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

#H_coupling += H_LR

H_coupling_obs = pad_matrix_for_observer(H_coupling)


for s in range(steps_coupling):
    #state_with_observer = apply_quadratic_coupling_exact(state=state_with_observer, G=H_coupling_obs, tau = dt)
    state_with_observer = step_coupling_window_twa(state_with_observer, dt, params, H_coupling_obs)
    q_obs,p_obs = state_with_observer
    scores = signal_map_vs_observer(q_obs, p_obs, 2*N)
    scores_array[steps+s,:] = scores/np.linalg.norm(scores)
    total_energy.append(energy_quadratic_from_samples(q_obs[:, :2*N], p_obs[:, :2*N], H_coupling))

    #mut_info_L.append(mut_info_observer_ksg(q_obs, p_obs, 0, N, 2*N, k=5))
    #mut_info_R.append(mut_info_observer_ksg(q_obs, p_obs, N, 2*N, 2*N, k=5))    
    mut_info_L.append(mut_info_observer(q_obs, p_obs, 0, N, 2*N))
    mut_info_R.append(mut_info_observer(q_obs, p_obs, N, 2*N, 2*N))
    tot_mut_info_obs.append(mut_info_observer(q_obs, p_obs, 0, 2*N, 2*N))        
    #current_dq = get_derivatives_masked(state,params,mask_L)[0]    
    #otoc_vals = np.mean(current_dq**2, axis=0) # Shape (N,)
    #otoc_history[s,:]=otoc_vals




# The wormhole interaction V = exp(i * g * Sum qL qR)
"""
g = 0.5
q_curr, p_curr = state

q0 = q_curr.copy()
p0 = p_curr.copy()

q_L = q_curr[:, :N]
q_R = q_curr[:, N:]


# Shift momenta
p_curr[:, :N] += g * q_R
p_curr[:, N:2*N] += g * q_L
 
p_curr[:, insert_idx] = p0[:,insert_idx]
p_curr[:, insert_idx+N] = p0[:,insert_idx+N]

state = (q_curr, p_curr)
"""
#q_curr,p_curr = state_with_observer
#state_with_observer = apply_LR_coupling_kick_obs_safe(state_with_observer,g=5,N=N)

print("5. Evolving RIGHT Forward...")
# Usually we evolve the whole system now to see the signal emerge
for s in range(steps):
    #state = rk4_step_masked(state, dt, params, mask_R)
    #state_with_observer = rk4_step_masked_observer(state_with_observer, dt, params, mask_R_observer)    
    state_with_observer = step_verlet_LR_obs_safe(state_with_observer, dt, params, evolve_left=False, evolve_right=True)
   
    q_obs,p_obs = state_with_observer   
    scores = signal_map_vs_observer(q_obs, p_obs, 2*N)
    scores_array[steps+steps_coupling+s,:] = scores/np.linalg.norm(scores)
    total_energy.append(sum(total_energy_LR_obs(q_obs,p_obs,N,params["m_squared"],params["k_coupling"]))/M)

    #mut_info_L.append(mut_info_observer_ksg(q_obs, p_obs, 0, N, 2*N, k=5))
    #mut_info_R.append(mut_info_observer_ksg(q_obs, p_obs, N, 2*N, 2*N, k=5))    

    mut_info_L.append(mut_info_observer(q_obs, p_obs, 0, N, 2*N))
    mut_info_R.append(mut_info_observer(q_obs, p_obs, N, 2*N, 2*N))    
    tot_mut_info_obs.append(mut_info_observer(q_obs, p_obs, 0, 2*N, 2*N))        
    #current_dq = get_derivatives_masked(state,params,mask_R)[0]
    #otoc_vals = np.mean(current_dq**2, axis=0) # Shape (N,)
    #otoc_history[s+steps,:]=otoc_vals

# --- Plotting ---

#times = [0,steps//2,steps-1,steps+steps_coupling//2,steps+steps_coupling-1,3/2*steps+steps_coupling,2*steps+steps_coupling-1]
times = [0,steps-1,steps+steps_coupling-1,2*steps+steps_coupling-1]

for t in range(len(times)):
    plt.plot(np.arange(2*N),scores_array[int(times[t]),:],label=f"t={times[t]*dt:.2f}")
plt.xlabel("site")
plt.ylabel("correlation with observer")
plt.legend()
plt.show()
#plt.savefig("plots/observer-correlation-vs-site.pdf")



print("stop")

time_axis = np.concatenate((np.arange(steps),np.arange(steps,steps+steps_coupling),np.arange(steps+steps_coupling,2*steps+steps_coupling))) * dt
plt.plot(time_axis,total_energy)
plt.xlabel("time")
plt.ylabel("energy")
plt.show()
#plt.savefig("plots/energy-vs-time.pdf")


plt.plot(time_axis,mut_info_L,color="black",label="left")
plt.plot(time_axis,mut_info_R,color="red",label="right")
plt.plot(time_axis,tot_mut_info_obs,color="blue",label="total")
plt.xlabel("time")
plt.ylabel("mutual information")
plt.legend()
plt.show()
#plt.savefig("mutual-inforamtion-vs-time.pdf")


def compute_MI_with_observer(Gamma, observer_idx, target_indices):
    # Gamma: 2n x 2n covariance matrix
    Gamma_obs = extract_subsystem_covariance(Gamma, [observer_idx])
    Gamma_target = extract_subsystem_covariance(Gamma, target_indices)
    Gamma_joint = extract_subsystem_covariance(Gamma, target_indices + [observer_idx])
    
    S_obs = von_neumann_entropy_alt(Gamma_obs)
    S_target = von_neumann_entropy_alt(Gamma_target)
    S_joint = von_neumann_entropy_alt(Gamma_joint)
    
    return S_obs + S_target - S_joint


def mut_info_vs_segments(Gamma,center_idx,n_one_side):
    N = n_one_side
 
    mut_info_insert_regions = []
    mut_info_telep_regions = []

    lengths_array = np.linspace(1,N // 2,N // 2)

    observer_idx = 2*N
    n_L = n_one_side
    n = 2*N

    for i in range(1,lengths_array.shape[0]):
        #if i == 0:
        #segment_telep = [teleported_idx]
        if center_idx - i >= 0 and center_idx + i < N:
            segment_insert = np.arange(center_idx - i, center_idx + i+1)
        if center_idx - i < 0:
            diff = np.abs(center_idx - i)
            segment_insert_1 = np.arange(N-diff,N)
            segment_insert_2 = np.arange(0,center_idx+i+1)
            segment_insert = np.concatenate((segment_insert_1,segment_insert_2))
        if center_idx + i >= N:
            diff = center_idx + i - N
            segment_insert_1 = np.arange(center_idx-i,N)
            segment_insert_2 = np.arange(0,diff+1)
            segment_insert = np.concatenate((segment_insert_1,segment_insert_2))
        segment_insert = np.ndarray.tolist(segment_insert)
        mut_info_insert_regions.append(compute_MI_with_observer(Gamma, observer_idx, segment_insert))
        if center_idx  - i >= 0  and center_idx  + i < N:
            segment_telep = np.arange(center_idx + N - i, center_idx + N + i+1)
        if center_idx - i < 0 :
            diff = np.abs(center_idx - i)
            segment_telep_1 = np.arange(2*N-diff,2*N)
            segment_telep_2 = np.arange(N ,center_idx + N + i+1)
            segment_telep = np.concatenate((segment_telep_1,segment_telep_2))
        if center_idx + i >= N:
            diff = center_idx + i - N
            segment_telep_1 = np.arange(center_idx + N - i,2*N)
            segment_telep_2 = np.arange(N,N + diff+1)
            segment_telep = np.concatenate((segment_telep_1,segment_telep_2))
        segment_telep = np.ndarray.tolist(segment_telep)
        mut_info_telep_regions.append(compute_MI_with_observer(Gamma, observer_idx, segment_telep))



    mut_info_insert_regions.append(compute_MI_with_observer(Gamma, observer_idx, list(range(n_L))))
    mut_info_telep_regions.append(compute_MI_with_observer(Gamma, observer_idx, list(range(n_L,n))))



    full_lengths_array = 2 * lengths_array + 1
    full_lengths_array[-1] = N    

    Gamma_2mode = two_mode_squeezed_state(r=1)

    plt.plot(full_lengths_array,mut_info_insert_regions,color='k',label = "insert side")
    plt.plot(full_lengths_array,mut_info_telep_regions,color='red', label ="teleport side")
    plt.axhline(compute_MI_with_observer(Gamma_2mode,1,[0]), color = "blue", label = "total mutual info with observer")
    plt.xlabel("length of segment")
    plt.ylabel("mutual info with observer")
    plt.title("mutual information of segments with observer")
    plt.legend()
    plt.show()
    #plt.savefig("plots/mutual-information-segment-with-observer.pdf")

q_obs,p_obs = state_with_observer
X_final = np.hstack([q_obs, p_obs])    
Gamma_final_with_observer = np.cov(X_final, rowvar=False) 

mut_info_vs_segments(Gamma=Gamma_final_with_observer,center_idx=N//2,n_one_side=N)






plt.figure(figsize=(10, 6))
for site in sites_to_watch:
    plt.plot(time_axis, otoc_history[:,site],label=f"site {site}")

plt.yscale('log') # Log scale to see exponential growth (Lyapunov regime)
plt.xlabel('Time')
plt.ylabel('OTOC (Operator Growth)')
plt.title(r'Scrambling with $\lambda$={params["lam"]}')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()
#plt.savefig(f"plots/scrambling-with-lambda=[lam].pdf")

# --- 6. Check Fidelity ---
# Extract the Right side of the final state and compare to target

#target_site_idx = 0 

fidelity_list = []
for i in range(2*N):
    target_site_idx = i
    fidelity_list.append(compute_fidelity(
    q_R_samples = state[0][:, target_site_idx], 
    p_R_samples = state[1][:, target_site_idx], 
    target_q = 0,      # You didn't shift q
    target_p  = epsilon,  # You shifted p by epsilon
    sigma_vac = 0.5
))

plt.plot(np.arange(2*N),fidelity_list)
plt.ylabel("fidelity")
plt.xlabel("site")
plt.show()
#plt.savefig("plots/fidelity-vs-site.pdf")

#print(f"Teleportation Fidelity: {fidelity:.4f}")

print("Protocol Complete.")


