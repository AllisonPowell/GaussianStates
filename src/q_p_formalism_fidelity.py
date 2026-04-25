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

def insert_signal_twa(state, N_site, insert_idx, Gamma_insert):
    """
    state: Tuple (q, p) each shape (M, 2*N_site)
    Gamma_2mode: (4, 4) Covariance matrix of the TMS state
    """
    q, p = state
    M = q.shape[0]
    
    # 1. Sample the 2-mode squeezed state (Signal + Observer)
    # Gamma_2mode basis: [q_sig, q_obs, p_sig, p_obs]
    tms_samples = np.random.multivariate_normal(np.zeros(2), Gamma_insert, size=M)
    
    q_sig = tms_samples[:, 0]
    p_sig = tms_samples[:, 1]

    
    # 2. Replace the Signal site trajectories
    # We are physically "removing" the old L_0 state and putting in a new one
    q_new = q.copy()
    p_new = p.copy()
    
    q_new[:, insert_idx] = q_sig
    p_new[:, insert_idx] = p_sig
    
    return (q_new, p_new)


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


def ring_force_cubic(q_ring, m2, k, lam):
    """
    q_ring: (M, N) for ONE ring.
    Returns F(q) = dp/dt for that ring under:
      V = 1/2 m2 Σ q^2 + (k/2) Σ (q_i - q_{i+1})^2 + lam Σ q^3
    """
    q_left  = np.roll(q_ring,  1, axis=1)
    q_right = np.roll(q_ring, -1, axis=1)
    spring = k * (q_left + q_right - 2.0*q_ring)
    return -(m2*q_ring) - 3.0*lam*(q_ring**2) + spring





def step_verlet_LR(state, dt, params, evolve_left: bool, evolve_right: bool, kick: bool):
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
        if kick==True:
            pL_half = pL + 0.5*dt*ring_force(qL, m2, k, lam)
        else:
            pL_half = pL + 0.5*dt*ring_force(qL, m2, k, 0)
    else:
        pL_half = pL

    if evolve_right:
        if kick==True:
            pR_half = pR + 0.5*dt*ring_force(qR, m2, k, lam)
        else:
            pR_half = pR + 0.5*dt*ring_force(qR, m2, k, 0)
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
        if kick == True:
            pL_new = pL_half + 0.5*dt*ring_force(qL_new, m2, k, lam)
        else:
            pL_new = pL_half + 0.5*dt*ring_force(qL_new, m2, k, 0)
    else:
        pL_new = pL_half

    if evolve_right:
        if kick==True:
            pR_new = pR_half + 0.5*dt*ring_force(qR_new, m2, k, lam)
        else:
          pR_new = pR_half + 0.5*dt*ring_force(qR_new, m2, k, 0)  
    else:
        pR_new = pR_half

    q_new = np.hstack([qL_new, qR_new])
    p_new = np.hstack([pL_new, pR_new])
    return (q_new, p_new)

def step_verlet_LR_obs_safe(state, dt, params, evolve_left: bool, evolve_right: bool, kick: bool):
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
        if kick==True:
            p_half[:, L] += 0.5 * dt * ring_force(q[:, L], m2, k, lam)
        else:
            p_half[:, L] += 0.5 * dt * ring_force(q[:, L], m2, k, 0)
    if evolve_right:
        if kick==True:
            p_half[:, R] += 0.5 * dt * ring_force(q[:, R], m2, k, lam)
        else:
            p_half[:, R] += 0.5 * dt * ring_force(q[:, R], m2, k, 0)
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
        if kick==True:
            p_new[:, L] += 0.5 * dt * ring_force(q_new[:, L], m2, k, lam)
        else:
            p_new[:, L] += 0.5 * dt * ring_force(q_new[:, L], m2, k, 0)
    if evolve_right:
        if kick==True:
           p_new[:, R] += 0.5 * dt * ring_force(q_new[:, R], m2, k, lam) 
        else:
            p_new[:, R] += 0.5 * dt * ring_force(q_new[:, R], m2, k, 0)

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
    state = step_verlet_LR_obs_safe(state, 0.5*dt, params, evolve_left=True, evolve_right=True,kick=False)

    # 2) exact linear coupling map
    state = apply_quadratic_coupling_exact(state, G=G_cpl_obs, tau=dt)

    # 3) half step nonlinear ring dynamics
    state = step_verlet_LR_obs_safe(state, 0.5*dt, params, evolve_left=True, evolve_right=True,kick=False)
    return state



def step_coupling_window_twa_no_obs(state, dt, params, G_cpl):
    # 1) half step nonlinear ring dynamics
    state = step_verlet_LR(state, 0.5*dt, params, evolve_left=True, evolve_right=True)

    # 2) exact linear coupling map
    state = apply_quadratic_coupling_exact(state, G=G_cpl, tau=dt)

    # 3) half step nonlinear ring dynamics
    state = step_verlet_LR(state, 0.5*dt, params, evolve_left=True, evolve_right=True)
    return state

def teleportation_protocol(s,theta,insert_idx,t0,t_couple,dt,state_TFD,H_coupling,params):
    state = state_TFD
    steps = int(t0/dt)
    steps_coupling = int(t_couple/dt)


    for t in range(steps):
        #if t%20 == 0:
            #state = step_verlet_LR(state, -dt, params, evolve_left=True, evolve_right=False,kick=True)
        #else:
        state = step_verlet_LR(state, -dt, params, evolve_left=True, evolve_right=False,kick=False)
    

    #Gamma_2mode = two_mode_squeezed_state(r=1)

    Rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    Squeeze = 0.5 * np.array([[np.exp(-2*s), 0],
                          [0, np.exp(2*s)]])

    Gamma_insert = Rot @ Squeeze @ Rot.T

    #state_with_observer = insert_observer_twa(state, N, insert_idx, Gamma_2mode)
    state_no_observer = insert_signal_twa(state, N, insert_idx, Gamma_insert)



    for t in range(steps):
        #state_with_observer = step_verlet_LR_obs_safe(state_with_observer, dt, params, evolve_left=True, evolve_right=False)
        #if t%20==0:
            #state_no_observer = step_verlet_LR(state_no_observer, dt, params, evolve_left=True, evolve_right=False,kick=True)
        #else:
        state_no_observer = step_verlet_LR(state_no_observer, dt, params, evolve_left=True, evolve_right=False,kick=False)

        #H_coupling_obs = pad_matrix_for_observer(H_coupling)

    for t in range(steps_coupling):
        #state_with_observer = step_coupling_window_twa(state_with_observer, dt, params, H_coupling_obs)
        state_no_observer = step_coupling_window_twa(state_no_observer, dt, params, H_coupling) 
        #state_no_observer = step_coupling_window_twa_no_obs(state_no_observer, dt, params, H_coupling)     
    for t in range(steps):   
        #state_with_observer = step_verlet_LR_obs_safe(state_with_observer, dt, params, evolve_left=False, evolve_right=True)
        #if t%20==0:
            #state_no_observer = step_verlet_LR(state_no_observer, dt, params, evolve_left=False, evolve_right=True, kick=True)
        #else:
        state_no_observer = step_verlet_LR(state_no_observer, dt, params, evolve_left=False, evolve_right=True, kick=False)
           
    return state_no_observer#,state_with_observer
    
def coupling_hamiltonian(N,mu,insert_idx,params):

    carrier_indices1 = np.arange(0,insert_idx)
    carrier_indices2 = np.arange(insert_idx+1,N)
    carrier_indices = np.concatenate((carrier_indices1,carrier_indices2))

    bdy_1_idx = np.arange(N)
    bdy_2_idx = np.arange(N,2*N)

    n_total = 2*N
    H_coupling = np.zeros((2*n_total, 2*n_total))


    omega0 = np.sqrt(params["m_squared"] + 2*params["k_coupling"])


    for j in carrier_indices:
        x_L = bdy_1_idx[j]
        x_R = bdy_2_idx[j]
        # x coupling
        H_coupling[x_L, x_R] = H_coupling[x_R, x_L] = mu * omega0/ 2
        # p coupling
        H_coupling[x_L + n_total, x_R + n_total] = H_coupling[x_R + n_total, x_L + n_total] = mu / (2*omega0)
    return H_coupling

def make_input_covariance(s, theta):
    Rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    Squeeze = 0.5 * np.array([[np.exp(-2*s), 0],
                              [0, np.exp( 2*s)]])
    return sym(Rot @ Squeeze @ Rot.T)



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

def decoder_from_X_flip(X):
    U, s, Vt = np.linalg.svd(X)

    s1, s2 = s
    D = np.diag((s2,s1))
    r = 0.5*np.log(s2/s1)
    squeeze = np.diag([np.exp(-r), np.exp(r)])

    eta = np.sqrt(s1 * s2)
    loss = np.diag((s1,s2))

    return U @ squeeze @ Vt

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

def entanglement_fidelity_gaussian(X, Y, S,subtract_Y,r=1.0):
    V0 = tmsv_cov(r)
    V1 = apply_channel_to_second_mode_xxpp(V0, X, Y)
    V1_dec = decode_on_B_xxpp(V1,inv(S),Y,subtract_Y)
    # zero means:
    #mu0 = np.zeros(4)
    #mu1 = np.zeros(4)  
    return fidelity_stable(V0,V1_dec)

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

def plot_wigner_ellipse(Gamma_mode, ax, label='', color='blue'):
    from scipy.linalg import eigh
    W = Gamma_mode[:2, :2].real  # just x, p block
    vals, vecs = eigh(W)
    width, height = 2 * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                      edgecolor=color, fc='None', lw=2, label=label)
    ax.add_patch(ellipse)

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

def tfd_cov_ring_from_normal_modes(N, k, m2, beta, eps_omega=1e-15):
    """
    Construct the *pure* TFD covariance matrix for the ring Hamiltonian
        H = 1/2 p^T p + 1/2 x^T V x
    at inverse temperature beta, using the normal-mode diagonalization of V.

    Output ordering (4N×4N) is:
        [x_L(1..N), x_R(1..N), p_L(1..N), p_R(1..N)]   (xxpp with LR split)

    This construction is an *analytic* Gaussian purification mode-by-mode, so in exact arithmetic
    symplectic eigenvalues of the 4N-mode state are exactly 0.5.
    """
    V = build_ring_potential(N, k, m2)

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

def fidelity_vs_block_size(
    block_sizes,
    obs_idx,
    insert_idx,
    center_idx,
    input_ensemble,   # list of (s, theta) you use for fitting
    t0,
    t_couple,
    dt,
    state_TFD,
    H_coupling,
    params
):
    N = params["N_site"]
    Fms = []
    Fmf = []

    #state_with_obs,state_no_obs=teleportation_protocol(s=0,theta=0,insert_idx=insert_idx,t0=t0,t_couple=t_couple,dt=dt,state_TFD=state_TFD,H_coupling=H_coupling,params=params)
    for m in block_sizes:
        # --- 2) Define right segment of length m ---
        #right_seg = right_segment_ids_centered(center_idx, bdy_len, m)

        # --- 3) Build V_OR_xxpp and compute passive decoder O ---
        #N_tot = 2*N+1
        #obs_idx = 2*N
        #V_OR_xxpp = build_V_OR_xxpp(Gamma_final_obs, obs_idx, right_seg,N_tot)


        #O, v = build_passive_decoder_from_observer(V_OR_xxpp, m)

        Vins, Vouts = [], []
        q_outs,p_outs = [],[]


        for s, theta in input_ensemble:
            # Run your usual protocol (NO observer) to get global Gamma_final
            state_no_obs=teleportation_protocol(s,theta,insert_idx=insert_idx,t0=t0,t_couple=t_couple,dt=dt,state_TFD=state_TFD,H_coupling=H_coupling,params=params)


            #B_xxpp = extract_block_xxpp(Gamma_final, right_seg)   # 2m x 2m

            # Passive decode
            #B_dec = passive_decode_right_block(B_xxpp, O)

            # Take decoded first mode as Vout (2x2)
            #Vout = first_mode_from_block(B_dec)

            q_no_obs,p_no_obs = state_no_obs
            X_no_obs = np.hstack([q_no_obs, p_no_obs])
            sigma_no_obs = np.cov(X_no_obs, rowvar=False) 

            q_outs.append(q_no_obs[:,center_idx+N])
            p_outs.append(p_no_obs[:,center_idx+N])

            #q_with_obs,p_with_obs = state_with_obs
            #X_with_obs = np.hstack([q_with_obs, p_with_obs])
            #sigma_with_obs = np.cov(X_with_obs, rowvar=False) 

            if len(block_sizes)==1:
                Vout = extract_subsystem_covariance(sigma_no_obs,[center_idx+N])
            #print(center_idx,extract_subsystem_covariance(sigma_no_obs,[11]))



            # Your input Vin is the inserted mode covariance (2x2)
            Vin = make_input_covariance(s, theta)  # returns 2x2 in (x,p)

            #f = wigner_overlap_with_gaussian_target(q_no_obs[:,center_idx+N], p_no_obs[:,center_idx+N], Vin, d_target=None)
            """
            if center_idx == 1:
                fig, ax = plt.subplots()
                plot_wigner_ellipse(Vin, ax, label='Input', color='green')            
                plot_wigner_ellipse(Vout, ax, label='Output', color='red')

                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                ax.set_xlabel("Position Quadrature")
                ax.set_ylabel("Momentum Quadrature")
                ax.set_aspect('equal')
                ax.legend()
                plt.title("Input vs Output Wigner Ellipses")
                plt.grid(True)
                plt.show()
            """
           
            #print(center_idx,"f=",f)
            #wigner_overlap.append(f)
            Vins.append(Vin)
            Vouts.append(Vout)



        # --- 5) Fit a single-mode Gaussian channel for this decoded mode ---
        X, Y = fit_gaussian_channel(Vins, Vouts)



        rot1,loss,squeeze,rot2=decompose_X(X)
        print(center_idx,"rot1=",rot1)
        print("rot2=",rot2)
        print("loss=",loss)
        print("squeeze=",squeeze)
        print( "Y=",Y)

        S_dec_symp = decoder_from_X_symplectic(X)  # your preferred
        S_dec_flip = decoder_from_X_flip(X)  # your preferred

        Fs = entanglement_fidelity_gaussian(X, Y, S_dec_symp, subtract_Y=False,r=1.0)
        Ff = entanglement_fidelity_gaussian(X, Y, S_dec_flip, subtract_Y=False,r=1.0)      
        
        Fms.append(Fs)
        Fmf.append(Ff)
        print("F_flip=",Ff)
        print("F_symp=",Fs)

        print("F_flip=",Ff)
        print("F_symp=",Fs)

        #wigner_overlap=[]
        #for v in range(len(Vouts)):  
            #f = wigner_overlap_with_gaussian_target(q_outs[v], p_outs[v] Vin, d_target=None)



        
        #wigner_f = sum(wigner_overlap)/len(wigner_overlap)


    return np.array(Fms),np.array(Fmf)

def fidelity_vs_site(
    insert_idx,
    input_ensemble,   # list of (s, theta) you use for fitting
    H_coupling,
    N,
    t0,
    t_couple,
    dt,
    state_TFD,
    params,
    wormhole):


    Vins = []

    Vouts = [[] for i in range(N)]


    for s, theta in input_ensemble:
        # Run your usual protocol (NO observer) to get global Gamma_final
        state_no_obs=teleportation_protocol(s,theta,insert_idx=insert_idx,t0=t0,t_couple=t_couple,dt=dt,state_TFD=state_TFD,H_coupling=H_coupling,params=params)
        q_no_obs,p_no_obs = state_no_obs
        X_no_obs = np.hstack([q_no_obs, p_no_obs])
        sigma_no_obs = np.cov(X_no_obs, rowvar=False)       
            
        Vins.append(make_input_covariance(s,theta))
        for i in range(N):
            Vouts[i].append(extract_subsystem_covariance(sigma_no_obs,[i+N]))
            
        

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



N = 10           # Number of sites in the ring
M = 2000        # Number of trajectories (samples)
params = {'m_squared': 13, 'k_coupling': 5, "momentum":1,'lam': 0.4, 'N_site': N}        # Total simulation time
t0 = 4
t_couple = 3
dt = .005        # Time step
steps = int(t0 / dt)
steps_coupling = int(t_couple/dt)
insert_idx = 1



#Gamma_TFD = tfd_cov(N,k=params["k_coupling"],m_squared=params["m_squared"])

Gamma_TFD = tfd_cov_ring_from_normal_modes(N, params["k_coupling"], params["m_squared"], beta=1, eps_omega=1e-15)
q,p = sample_tfd_state(Gamma_TFD, M, N)
state_TFD = (q, p)


site_fidelities_symp=[]
site_fidelities_flip=[]
wigner_fidelities = []

block_sizes = [1]
#block_sizes = [1,2,4,6,8,10]

N = 10
obs_idx = 2*N
insert_idx = 1
teleported_idx = insert_idx+N
bdy_len = N

H_coupling=coupling_hamiltonian(N,mu=1,insert_idx=insert_idx,params=params)

Ss = np.linspace(-1.5, 1.5, 4)
Thetas = np.linspace(0, 2*np.pi, 3, endpoint=False)
input_ensemble = [(s, th) for s in Ss for th in Thetas]  # 120 points, deterministic

sites=np.arange(N,2*N)

"""
Fs,Ff = fidelity_vs_site(
    insert_idx,
    input_ensemble,
    H_coupling,
    N,
    t0,
    t_couple,
    dt,
    state_TFD,
    params,
    wormhole=False)

plt.plot(sites,Fs,label="symplectic")
plt.plot(sites,Ff,label="allow flip")
plt.xlabel("site")
plt.ylabel("fidelity")
plt.legend()
plt.show()
"""
#print("done")



lambda_vals = np.linspace(0,6,30)
lam_fid_symp = []
lam_fid_flip = []
for lam in lambda_vals:
    params = {'m_squared': 13, 'k_coupling': 5, "momentum":1,'lam': lam, 'N_site': N}   
    Fs,Ff = fidelity_vs_site(
    insert_idx,
    input_ensemble,   # list of (s, theta) you use for fitting
    H_coupling,
    N,
    t0,
    t_couple,
    dt,
    state_TFD,
    params,
    wormhole=False)
    lam_fid_symp.append(Fs[insert_idx])
    lam_fid_flip.append(Ff[insert_idx])

#plt.plot(lambda_vals,lam_fid_symp)
plt.plot(lambda_vals,lam_fid_flip)
plt.xlabel("added non-gaussianity")
plt.ylabel("fidelity")
plt.savefig("plots/fidelity_vs_lam.pdf")
#plt.show()


"""
for f in range(len(sites)):
    #Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    #plt.plot(block_sizes,Fs,label=sites[f])
    Fs,Ff= fidelity_vs_block_size(
    block_sizes=block_sizes,
    obs_idx=2*N,
    insert_idx=insert_idx,
    center_idx=sites[f]-N,
    input_ensemble=input_ensemble,   # list of (s, theta) you use for fitting
    t0=t0,
    t_couple=t_couple,
    dt=dt,
    state_TFD=state_TFD,
    H_coupling=H_coupling,
    params=params)
    site_fidelities_symp.append(Fs)
    site_fidelities_flip.append(Ff)

plt.plot(sites,site_fidelities_symp,label="symplectic")
plt.plot(sites,site_fidelities_flip,label="allow flip")
plt.xlabel("site")
plt.ylabel("fidelity")
plt.legend()
#plt.show()
plt.savefig("plots/site_vs_fidelity.pdf")
"""
"""
plt.xlabel("decoder block size")
plt.ylabel("fidelity")
plt.legend()
plt.show()
"""  



"""
times_evolve= np.linspace(2.4,2.8,9)
time_fidelities_symp = []
time_fidelities_flip = []
for t in range(len(times_evolve)):
    #Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    #plt.plot(block_sizes,Fs,label=sites[f])
    Fs,Ff= fidelity_vs_block_size(
    block_sizes=block_sizes,
    obs_idx=2*N,
    insert_idx=insert_idx,
    center_idx=insert_idx,
    input_ensemble=input_ensemble,   # list of (s, theta) you use for fitting
    t0=times_evolve[t],
    t_couple=t_couple,
    dt=dt,
    state_TFD=state_TFD,
    H_coupling=H_coupling,
    params=params)
    time_fidelities_symp.append(Fs)
    time_fidelities_flip.append(Ff)

plt.plot(times_evolve,time_fidelities_symp,label="symplectic")
plt.plot(times_evolve,time_fidelities_flip,label="allow flip")
plt.xlabel("times")
plt.ylabel("fidelity")
plt.legend()
#plt.show()
plt.savefig("plots/time_vs_fidelity.pdf")
"""

print("done")

