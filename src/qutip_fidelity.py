import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, expm, sqrtm, schur, block_diag, eigh, det, polar
from thewalrus.symplectic import xpxp_to_xxpp, sympmat
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
import math

from qutip import (
    destroy,
    tensor,
    qeye,
    expect,
    sesolve,
    basis,
    thermal_dm,
    displace,
    squeeze,
    Qobj,
    expand_operator
)

def symplectic_form(n):
    """Returns the 2n × 2n symplectic form Omega"""
    return np.block(
        [
            [np.zeros((n, n), dtype=np.float64), np.eye(n, dtype=np.float64)],
            [-np.eye(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64)],
        ]
    )


def symplectic_eigenvalues(Gamma):
    """
    Compute the symplectic eigenvalues ν_i of a covariance matrix Γ.
    """
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * Omega @ Gamma)
    ν = np.sort(np.abs(eigvals))[::2]  # Take only one of each ν_i pair
    return ν



def extract_subsystem_covariance(Gamma, indices):
    indices = np.array(indices)
    x_idx = indices
    p_idx = indices + Gamma.shape[0] // 2
    full_idx = np.concatenate([x_idx, p_idx])
    return Gamma[np.ix_(full_idx, full_idx)]


def von_neumann_entropy_alt(Gamma):
    n = Gamma.shape[0] // 2
    Omega = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    eigvals = np.linalg.eigvals(1j * Omega @ Gamma)
    nu = np.sort(np.abs(eigvals))[::2]
    nu = np.clip(nu, 0.500001, None)
    return sum((nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5))


def get_mutual_info_LR(q, p, N):
    # Split into L and R
    q_L, p_L = q[:, :N], p[:, :N]
    q_R, p_R = q[:, N:], p[:, N:]

    S_L = compute_entropy_from_samples(q_L, p_L)
    S_R = compute_entropy_from_samples(q_R, p_R)
    S_LR = compute_entropy_from_samples(q, p)  # Global entropy

    return S_L + S_R - S_LR


def mut_info_observer_old(q, p, idx_0, idx_f, idx_obs):
    # Split into L and R
    q_a, p_a = q[:, idx_0:idx_f], p[:, idx_0:idx_f]
    q_obs, p_obs = q[:, idx_obs], p[:, idx_obs]

    S_a = compute_entropy_from_samples(q_a, p_a)
    S_obs = compute_entropy_from_samples(q_obs, p_obs)
    S_joint = compute_entropy_from_samples(q, p)  # Global entropy

    return S_a + S_obs - S_joint


def mut_info_observer(q, p, idx_0, idx_f, idx_obs):
    # Split into L and R

    indices_a = list(np.arange(idx_0, idx_f))

    S_a = compute_entropy_from_samples(q, p, indices_a)
    S_obs = compute_entropy_from_samples(q, p, [idx_obs])
    S_joint = compute_entropy_from_samples(
        q, p, indices_a + [idx_obs]
    )  # Global entropy

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
    exponent = -(delta_q**2 + delta_p**2)  # Simplified for symmetric vacuum

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

    total_sites = 2 * N_site  # Total number of q's

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
    D = np.diag(
        np.repeat(nu, 1)
    )  # no double since epsilons already doubled for 2x2 blocks

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
        V[i, i] = m2 + 2 * k
        V[i, (i + 1) % N] = -k
        V[i, (i - 1) % N] = -k

    # Diagonalize V = O diag(omega^2) O^T
    omega2, O = np.linalg.eigh(V)
    omega = np.sqrt(omega2)

    # Thermal variances in normal modes
    coth = lambda x: 1 / np.tanh(x)

    var_q = (1 / (2 * omega)) * coth(beta * omega / 2)
    var_p = (omega / 2) * coth(beta * omega / 2)

    Gamma_xx = O @ np.diag(var_q) @ O.T
    Gamma_pp = O @ np.diag(var_p) @ O.T

    Gamma = np.block([[Gamma_xx, np.zeros((N, N))], [np.zeros((N, N)), Gamma_pp]])

    return 0.5 * (Gamma + Gamma.T)


def symplectic_direct_sum(S1, S2):
    n = S1.shape[0]
    A1 = S1[0 : n // 2, 0 : n // 2]
    B1 = S1[0 : n // 2, n // 2 : n]
    C1 = S1[n // 2 : n, 0 : n // 2]
    D1 = S1[n // 2 : n, n // 2 : n]

    A2 = S2[0 : n // 2, 0 : n // 2]
    B2 = S2[0 : n // 2, n // 2 : n]
    C2 = S2[n // 2 : n, 0 : n // 2]
    D2 = S2[n // 2 : n, n // 2 : n]

    A_block = block_diag(A1, A2)
    B_block = block_diag(B1, B2)
    C_block = block_diag(C1, C2)
    D_block = block_diag(D1, D2)

    S_tot = np.block([[A_block, B_block], [C_block, D_block]])
    return S_tot


def williamson_strawberry(V):
    tol = 1e-11
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

    if diffn >= 10 ** (-5):
        raise ValueError("The input matrix is not symmetric")

    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
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
    Db = np.diag(
        [1 / dd[i, i + n] for i in range(n)] + [1 / dd[i, i + n] for i in range(n)]
    )
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
    C_xxpp = np.block(
        [[C_top, np.zeros_like(C_top)], [np.zeros_like(C_bottom), C_bottom]]
    )  # 2n x 2n

    D_xxpp = Db_xxpp  # this is already diag(nu_1,...,nu_n, nu_1,...,nu_n)

    V_pure_will_xxpp = np.block([[D_xxpp, C_xxpp], [C_xxpp, D_xxpp]])

    S_total = symplectic_direct_sum(S_xxpp.T, S_xxpp.T)  # or S_xxpp ⊕ I if you prefer
    V_pure_phys_xxpp = S_total @ V_pure_will_xxpp @ S_total.T

    return V_pure_phys_xxpp


def tfd_cov(N, k, m_squared):
    HL = np.zeros((2 * N, 2 * N))

    for i in range(2 * N):
        if i < N - 1:
            HL[i, i] = m_squared + 2 * k  # on-site + two neighbors
            HL[i, i + 1] = -k
            HL[i + 1, i] = -k

        if i == N - 1:
            HL[i, 0] = -k
            HL[0, i] = -k
            HL[i, i] = m_squared + 2 * k
        if i > N - 1:
            HL[i, i] = 1

    Gamma_reconstructed, nu, eps_reconstructed = (
        build_thermal_state_from_modular_hamiltonian(HL)
    )

    Gamma_TFD = gaussian_purification(Gamma_reconstructed)

    return Gamma_TFD




def teleportation_protocol(
    s, theta, insert_idx, t_scramble, t_couple, dt, C_back, H_coupling, H_scramble_side, N_modes,N_cutoff,x_full_env,p_full_env):
    #U_inject_side = build_injection_unitary_side(
    #N_modes, N_cutoff, insert_idx,
    #alpha=0.0, r=s, phi_sq=np.pi, phi_rot=theta).full()
    #C_in = apply_left(C_back, U_inject_side)
    phi_insert = phi_squeezed_vacuum(N_cutoff, s, theta) 
    dim_list=[N_cutoff]*3
    C_in,left_dims_new,right_dims_new = replace_mode_in_bipartite_C(
    C=C_back,
    left_dims=dim_list,
    right_dims=dim_list,
    side= "left",             # "left" or "right"
    mode_index= insert_idx,       # index within that side
    phi=phi_insert)
    N_modes_total = len(left_dims_new) + len(right_dims_new)
    psi_in, d_in, V_in = covariance_from_C(C_in, N_modes_total, N_cutoff, x_full_env, p_full_env)
    left_dims = [N_cutoff]*3
    right_dims = [N_cutoff]*4
    U_fwd_side  = (-1j * H_scramble_side * t_scramble).expm()
    U_fwd = U_fwd_side.full()
    C_fwd = apply_left(C_in, U_fwd)

    n_steps = int(t_couple/dt)
    #C_coup = strang_coupling(C_fwd, H0_side, H_coupling, t_couple=t_couple, n_steps=n_steps, N_modes=N_modes,N_cutoff=N_cutoff)
    C_coup = strang_coupling_env_spectator(C_fwd, H0_side, H_coupling, t_couple, n_steps, left_dims, right_dims)
    U_right_ext = np.kron(U_fwd, np.eye(N_cutoff))
    C_fin = apply_right(C_coup, U_right_ext)
    psi_out, d_out, V_out = covariance_from_C(C_fin, N_modes_total, N_cutoff, x_full_env, p_full_env)
    return C_fin,V_in,V_out

def coupling_hamiltonian(N, mu, insert_idx, params):

    carrier_indices1 = np.arange(0, insert_idx)
    carrier_indices2 = np.arange(insert_idx + 1, N)
    carrier_indices = np.concatenate((carrier_indices1, carrier_indices2))

    bdy_1_idx = np.arange(N)
    bdy_2_idx = np.arange(N, 2 * N)

    n_total = 2 * N
    H_coupling = np.zeros((2 * n_total, 2 * n_total))

    omega0 = np.sqrt(params["m_squared"] + 2 * params["k_coupling"])

    for j in carrier_indices:
        x_L = bdy_1_idx[j]
        x_R = bdy_2_idx[j]
        # x coupling
        H_coupling[x_L, x_R] = H_coupling[x_R, x_L] = mu * omega0 / 2
        # p coupling
        H_coupling[x_L + n_total, x_R + n_total] = H_coupling[
            x_R + n_total, x_L + n_total
        ] = mu / (2 * omega0)
    return H_coupling


def make_input_covariance(s, theta):
    Rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Squeeze = 0.5 * np.array([[np.exp(-2 * s), 0], [0, np.exp(2 * s)]])
    return sym(Rot @ Squeeze @ Rot.T)


def sym(A):
    return 0.5 * (A + A.T)


def pack_params(X, Y):
    # Y symmetric
    return np.array(
        [X[0, 0], X[0, 1], X[1, 0], X[1, 1], Y[0, 0], Y[0, 1], Y[1, 1]], dtype=float
    )


def unpack_params(p):
    a, b, c, d, y11, y12, y22 = p
    X = np.array([[a, b], [c, d]], dtype=float)
    Y = np.array([[y11, y12], [y12, y22]], dtype=float)
    return X, Y


def residuals(p, Vins, Vouts):
    X, Y = unpack_params(p)
    r = []
    for Vin, Vout in zip(Vins, Vouts):
        E = sym(Vout - (X @ Vin @ X.T + Y))
        r.extend([E[0, 0], E[0, 1], E[1, 1]])  # 3 independent comps
    return np.array(r, dtype=float)


def fit_gaussian_channel(Vins, Vouts, X0=None, Y0=None, lam=1e-3, iters=200):
    Vins = [sym(V) for V in Vins]
    Vouts = [sym(V) for V in Vouts]

    if X0 is None:
        X0 = np.eye(2)
    if Y0 is None:
        # crude initial Y as average difference
        Y0 = sym(
            np.mean([Vout - X0 @ Vin @ X0.T for Vin, Vout in zip(Vins, Vouts)], axis=0)
        )

    p = pack_params(X0, Y0)

    for _ in range(iters):
        r = residuals(p, Vins, Vouts)
        cost = r @ r

        # numerical Jacobian (7 params)
        J = np.zeros((len(r), len(p)))
        eps = 1e-6
        for j in range(len(p)):
            dp = np.zeros_like(p)
            dp[j] = eps
            r2 = residuals(p + dp, Vins, Vouts)
            J[:, j] = (r2 - r) / eps

        # LM step: (J^T J + lam I) delta = J^T r
        A = J.T @ J + lam * np.eye(len(p))
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
    O1 = U.copy()
    O2 = Vt.copy()

    if det(U) < 0:
        O1[:, 1] *= -1

    s1, s2 = s
    D = np.diag((s2, s1))
    r = 0.5 * np.log(s2 / s1)
    squeeze = np.diag([np.exp(-r), np.exp(r)])

    eta = np.sqrt(s1 * s2)

    if det(U) < 0:
        loss = np.diag((eta, -eta))
    else:
        loss = np.diag((eta, eta))

    return O1 @ squeeze @ O2


def decoder_from_X_flip(X):
    U, s, Vt = np.linalg.svd(X)

    s1, s2 = s
    D = np.diag((s2, s1))
    r = 0.5 * np.log(s2 / s1)
    squeeze = np.diag([np.exp(-r), np.exp(r)])

    eta = np.sqrt(s1 * s2)
    loss = np.diag((s1, s2))

    return U @ squeeze @ Vt


def fidelity_stable(V1, V2):
    V1 = 0.5 * (V1 + V1.T)
    V2 = 0.5 * (V2 + V2.T)
    n = V1.shape[0] // 2
    omega = symplectic_form(n)

    Vsum = V1 + V2
    V_aux = omega.T @ np.linalg.inv(Vsum) @ (0.25 * omega + V2 @ omega @ V1)

    I = np.eye(2 * n)
    A = V_aux @ omega

    # A^{-2} = solve(A, solve(A, I))
    Ainv2 = np.linalg.solve(A, np.linalg.solve(A, I))
    inside = I + 0.25 * Ainv2

    F_tot4 = np.linalg.det(2 * (sqrtm(inside) + I) @ V_aux)
    F_tot = np.real_if_close(F_tot4) ** 0.25
    F0 = F_tot / (np.linalg.det(Vsum) ** 0.25)

    return float(np.real(F0))


def decode_on_B_xxpp(V_RB_xxpp, S_dec, Y, subtract_Y):
    I2 = np.eye(2)
    # xxpp ordering: (xR, xB, pR, pB)
    # decoding acts on (xB,pB) => indices [1,3], not contiguous.
    V = 0.5 * (V_RB_xxpp + V_RB_xxpp.T)
    idx_R = [0, 2]
    idx_B = [1, 3]

    Vout = V.copy()

    # transform blocks: B -> S_dec B S_dec^T, C -> C S_dec^T
    A = V[np.ix_(idx_R, idx_R)]
    B = V[np.ix_(idx_B, idx_B)]
    C = V[np.ix_(idx_R, idx_B)]

    if subtract_Y == True:
        B -= Y

    B2 = S_dec @ B @ S_dec.T
    C2 = C @ S_dec.T

    Vout[np.ix_(idx_R, idx_R)] = A
    Vout[np.ix_(idx_B, idx_B)] = B2
    Vout[np.ix_(idx_R, idx_B)] = C2
    Vout[np.ix_(idx_B, idx_R)] = C2.T

    return 0.5 * (Vout + Vout.T)


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

    cov = 0.5 * np.block([[ch * np.eye(2), sh * Z], [sh * Z, ch * np.eye(2)]])

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
    A = V[np.ix_(idx_R, idx_R)]  # Cov of R
    B = V[np.ix_(idx_B, idx_B)]  # Cov of B
    C = V[np.ix_(idx_R, idx_B)]  # Cross-cov R-B

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


def entanglement_fidelity_gaussian(X, Y, S, subtract_Y, r=1.0):
    V0 = tmsv_cov(r)
    V1 = apply_channel_to_second_mode_xxpp(V0, X, Y)
    V1_dec = decode_on_B_xxpp(V1, inv(S), Y, subtract_Y)
    # zero means:
    # mu0 = np.zeros(4)
    # mu1 = np.zeros(4)
    return fidelity_stable(V0, V1_dec)


def right_segment_ids_centered(center_idx, n, m):
    i = m // 2
    if m == 1:
        segment_telep = np.array([center_idx + n])
    elif center_idx - i >= 0 and center_idx + i < n:
        segment_telep = np.arange(center_idx + n - i, center_idx + n + i)
    elif center_idx - i < 0:
        diff = np.abs(center_idx - i)
        segment_telep_1 = np.arange(2 * n - diff, 2 * n)
        segment_telep_2 = np.arange(n, center_idx + n + i)
        segment_telep = np.concatenate((segment_telep_1, segment_telep_2))
    elif center_idx + i >= n:
        diff = center_idx + i - n
        segment_telep_1 = np.arange(center_idx + n - i, 2 * n)
        segment_telep_2 = np.arange(n, n + diff)
        segment_telep = np.concatenate((segment_telep_1, segment_telep_2))
    return segment_telep


def decompose_X(X):
    U, s, Vt = np.linalg.svd(X)
    O1 = U.copy()
    O2 = Vt.copy()

    if det(U) < 0:
        O1[:, 1] *= -1

    s1, s2 = s
    D = np.diag((s2, s1))
    r = 0.5 * np.log(s2 / s1)
    squeeze = np.diag([np.exp(-r), np.exp(r)])

    eta = np.sqrt(s1 * s2)

    if det(U) < 0:
        loss = np.diag((eta, -eta))
    else:
        loss = np.diag((eta, eta))

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

    V = 0.5 * (V_target + V_target.T)
    Vinv = np.linalg.inv(V)
    detV = np.linalg.det(V)

    dz = z - d[None, :]
    # quadratic form (z-d)^T Vinv (z-d) for each sample
    quad = np.einsum("bi,ij,bj->b", dz, Vinv, dz)

    # Tr(rho_out rho_tar) = average[ (1/sqrt(detV)) * exp(-0.5*quad) ]
    overlap = np.mean((1.0 / np.sqrt(detV)) * np.exp(-0.5 * quad))
    return float(overlap)


def plot_wigner_ellipse(Gamma_mode, ax, label="", color="blue"):
    from scipy.linalg import eigh

    W = Gamma_mode[:2, :2].real  # just x, p block
    vals, vecs = eigh(W)
    width, height = 2 * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ellipse = Ellipse(
        xy=(0, 0),
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        fc="None",
        lw=2,
        label=label,
    )
    ax.add_patch(ellipse)

def build_TFD_C_fast(evals, evecs, beta, keep_states):
    evals = np.asarray(evals, float)
    evals0 = evals - evals.min()
    w = np.exp(-beta * evals0)

    idx = np.argsort(evals)
    if keep_states is not None:
        idx = idx[:keep_states]

    w_kept = w[idx]
    w_kept /= w_kept.sum()

    # V is D x K with columns = eigenvectors
    V = np.column_stack([evecs[n].full().ravel() for n in idx])  # (D, K)
    C = V @ np.diag(np.sqrt(w_kept)) @ V.T                      # (D, D)
    return C

def apply_left(C, U_L):
    return U_L @ C

def apply_right(C, U_R):
    return C @ U_R.T

def build_injection_unitary_side(N_modes, N_cutoff, insert_idx, alpha, r, phi_sq, phi_rot):
    a = destroy(N_cutoff)
    I = qeye(N_cutoff)

    # single-mode ops
    S = squeeze(N_cutoff, r, phi_sq)
    D = displace(N_cutoff, alpha)
    R = (1j * phi_rot * a.dag() * a).expm()
    U_local = D * S * R

    # wrap to side space
    ops = [I]*N_modes
    ops[insert_idx] = U_local
    return tensor(ops)

def wrap(local_op, mode, N_modes, I):
    ops = [I] * (N_modes)
    ops[mode] = local_op
    return tensor(ops)

def wrap_full(local_op, mode, N_total_modes, I):
    ops = [I] * N_total_modes
    ops[mode] = local_op
    return tensor(ops)


def C_from_vec(v, D_L,D_R):
    return np.array(v).reshape((D_L, D_R), order="F")


def apply_Hint_step_sesolve_old(C, H_int, dt, N_modes, N_cutoff):
    """
    Apply exp(-i H_int dt) to vec(C) WITHOUT building U_int.
    Uses sesolve as a matrix-free action.
    """
    D_side = N_cutoff**N_modes
    v = vec_from_C(C)                 # length D_side^2 = N_cutoff^(2*N_modes)
    N_total = 2 * N_modes

    psi = Qobj(v, dims=[[N_cutoff]*N_total, [1]*N_total])  # match H_int dims
    out = sesolve(H_int, psi, [0, dt], progress_bar=None)
    v2 = out.states[-1].full().ravel()
    return C_from_vec(v2, D_side,D_side)

def apply_Hint_step_sesolve(C, H_int, dt, N_modes, N_cutoff):
    D_side = N_cutoff**N_modes
    v = vec_from_C(C)
    N_total = 2 * N_modes

    psi = Qobj(v, dims=[[N_cutoff]*N_total, [1]])  
    out = sesolve(H_int, psi, [0, dt], progress_bar=None)
    v2 = out.states[-1].full().ravel()
    return C_from_vec(v2, D_side,D_side)

def extend_right_unitary_with_env(U_right: np.ndarray, d_env: int):
    return np.kron(U_right, np.eye(d_env, dtype=np.complex128))

def extend_H_with_env_identity(H_int: Qobj, d_env: int) -> Qobj:
    return qt.tensor(H_int, qt.qeye(d_env))

def strang_coupling_env_spectator(
    C: np.ndarray,
    H0_side: Qobj,         # local H on ONE SIDE (3-mode), same for L and R(system) if you want
    H_int_6mode: Qobj,     # interaction Hamiltonian on OLD 6-mode space (3L+3R), NO env
    t_couple: float,
    n_steps: int,
    left_dims: list[int],  # e.g. [d,d,d]
    right_dims: list[int], # e.g. [d,d,d,d] (includes env as last)
):
    """
    Strang splitting with env spectator:
      - Left local uses U0_half (3-mode)
      - Right local uses U0_half ⊗ I_env
      - Interaction uses H_int ⊗ I_env (so env does nothing)
    """
    dt = t_couple / n_steps
    d_env = right_dims[-1]
    d_right_sys = int(np.prod(right_dims[:-1]))

    # Local half step on 3-mode side (as Qobj -> numpy)
    U0_half = (-1j * H0_side * (dt/2)).expm().full()  # shape (d^3, d^3)

    # Extend right local step to include env as identity
    U0_half_right = np.kron(U0_half, np.eye(d_env, dtype=np.complex128))  # (d^3*d_env, d^3*d_env)

    # Extend interaction Hamiltonian to include env spectator
    H_int_ext = qt.tensor(H_int_6mode, qt.qeye(d_env))

    for _ in range(n_steps):
        # half local on both sides
        C = apply_left(C, U0_half)
        C = apply_right(C, U0_half_right)

        # full interaction step on total space (now 7 modes)
        C = apply_Hint_step_sesolve_bipartite(
            C=C,
            H_int_ext=H_int_ext,
            dt=dt,
            left_dims=left_dims,
            right_dims=right_dims,
        )

        # half local again
        C = apply_left(C, U0_half)
        C = apply_right(C, U0_half_right)

    return C

def strang_coupling_old(C, H0_side, H_int, t_couple, n_steps, N_modes, N_cutoff):
    """
    Strang splitting for coupling window:
      e^{-i(H0_L+H0_R+H_int)t} ≈ [e^{-i(H0_L+H0_R)dt/2} e^{-i H_int dt} e^{-i(H0_L+H0_R)dt/2}]^n
    with local parts applied via left/right multiplication on C, and interaction via sesolve.
    """
    dt = t_couple / n_steps
    D_side = N_cutoff**N_modes

    # local half-step on ONE side
    U0_half = (-1j * H0_side * (dt/2)).expm().full()

    for _ in range(n_steps):
        # half local on both sides
        C = apply_left(C, U0_half)
        C = apply_right(C, U0_half)

        # full interaction step (matrix-free)
        C = apply_Hint_step_sesolve(C, H_int, dt, N_modes, N_cutoff)

        # half local again
        C = apply_left(C, U0_half)
        C = apply_right(C, U0_half)

    return C

import numpy as np
import qutip as qt
from qutip import Qobj, sesolve

def apply_Hint_step_sesolve_bipartite(
    C: np.ndarray,
    H_int_ext: Qobj,
    dt: float,
    left_dims: list[int],
    right_dims: list[int],
):
    """
    One interaction step on the FULL Hilbert space (left + right (+ env if present)),
    evolving the flattened state vector with sesolve.

    C: (D_L, D_R) coefficient matrix
    H_int_ext: Qobj Hamiltonian on total modes (dims = left_dims + right_dims)
              IMPORTANT: this should already include ⊗ I_env if env is spectator.
    """
    D_L = int(np.prod(left_dims))
    D_R = int(np.prod(right_dims))
    if C.shape != (D_L, D_R):
        raise ValueError(f"C shape {C.shape} does not match (D_L,D_R)=({D_L},{D_R}).")

    # vec convention must match your vec_from_C/C_from_vec
    v = vec_from_C(C)  # shape (D_L*D_R,)
    dims_total = left_dims + right_dims

    psi = Qobj(v, dims=[dims_total, [1]])  # ket on total modes
    out = sesolve(H_int_ext, psi, [0, dt], progress_bar=None)
    v2 = out.states[-1].full().ravel()

    return C_from_vec(v2, D_L,D_R)  # returns (D_L, D_R)

def vec_from_C(C):
    # column-stacking convention (Fortran order), consistent with your earlier code
    return C.reshape((-1,), order="F")

def ket_from_C_old(C, N_modes, N_cutoff):
    """
    C is D_side x D_side with D_side = N_cutoff**N_modes.
    Interpreted as |psi> = sum_{i,j} C_{ij} |i>_L ⊗ |j>_R  (pure state).
    """
    N_total = 2 * N_modes
    v = vec_from_C(C)
    psi = Qobj(v, dims=[[N_cutoff]*N_total, [1]*N_total])
    # normalize defensively (numerics / truncation)
    return psi.unit()

def ket_from_C(C, N_modes_total, N_cutoff):
    v = vec_from_C(C)
    psi = Qobj(v, dims=[[N_cutoff]*N_modes_total, [1]])  
    return psi.unit()

def build_quadrature_ops(N_modes, N_cutoff):
    """
    Returns lists x_ops, p_ops for all modes in the FULL (2*N_modes)-mode Hilbert space.
    """
    N_total = 2 * N_modes
    a = destroy(N_cutoff)
    I = qeye(N_cutoff)

    x_loc = (a + a.dag()) / np.sqrt(2)
    p_loc = (a - a.dag()) / (1j * np.sqrt(2))

    x_ops = []
    p_ops = []
    for m in range(N_total):
        ops = [I]*N_total
        ops[m] = x_loc
        x_ops.append(tensor(ops))

        ops = [I]*N_total
        ops[m] = p_loc
        p_ops.append(tensor(ops))
    return x_ops, p_ops

def covariance_from_ket(psi, x_ops, p_ops):
    """
    Returns:
      d: mean vector in ordering [x1..xn, p1..pn]
      V: covariance matrix in same ordering
    """
    n = len(x_ops)
    R_ops = x_ops + p_ops  # ordering: xx..x pp..p
    d = np.array([expect(op, psi) for op in R_ops], dtype=complex)
    d = np.real_if_close(d).astype(float)

    V = np.zeros((2*n, 2*n), dtype=float)
    for j in range(2*n):
        for k in range(2*n):
            # symmetrized second moment: 0.5 <Rj Rk + Rk Rj> - <Rj><Rk>
            sym_moment = 0.5 * expect(R_ops[j]*R_ops[k] + R_ops[k]*R_ops[j], psi)
            V[j, k] = float(np.real(sym_moment) - d[j]*d[k])
    return d, V

def covariance_from_C(C, N_modes_total, N_cutoff, x_ops, p_ops):
    psi = ket_from_C(C, N_modes_total, N_cutoff)
    if x_ops is None or p_ops is None:
        x_ops, p_ops = build_quadrature_ops(N_modes_total, N_cutoff)
    d, V = covariance_from_ket(psi, x_ops, p_ops)
    return psi, d, V



def swap_operator_single_mode(dim):
    """
    SWAP gate on two qudits of dimension dim, as a Qobj operator on dim^2 space.
    """
    swap = 0
    for i in range(dim):
        for j in range(dim):
            ket_ij = tensor(basis(dim, i), basis(dim, j))
            ket_ji = tensor(basis(dim, j), basis(dim, i))
            swap = swap + ket_ji * ket_ij.dag()
    return swap

def replace_mode_pure_stinespring(psi, mode_index, phi_insert):
    """
    Pure-state replacement via environment:
      - add env in vacuum |0>
      - SWAP(system mode_index, env)
      - overwrite system mode_index by preparing |phi_insert> (pure)

    This is a unitary+local-prep construction that yields a pure global state on N+1 modes.

    Args:
        psi: Qobj ket on N modes
        mode_index: which mode to replace
        phi_insert: Qobj ket (single mode) to insert

    Returns:
        psi_out: Qobj ket on N+1 modes (system modes + env mode at the end)
    """
    N = len(psi.dims[0])
    d = psi.dims[0][mode_index]
    assert phi_insert.isket
    assert phi_insert.shape[0] == d

    # Add environment mode in |0>
    env0 = basis(d, 0)
    psi_ext = tensor(psi, env0)  # N+1 modes, env is last

    # Build SWAP between mode_index and env(last)
    # We need a SWAP acting on the full (N+1)-mode space.
    swap_local = swap_operator_single_mode(d)

    # Embed SWAP into full space via tensor identities
    ops = []
    for m in range(N+1):
        if m == mode_index:
            ops.append(None)  # placeholder for first factor of swap
        elif m == N:
            ops.append(None)  # placeholder for second factor of swap
        else:
            ops.append(qeye(psi_ext.dims[0][m]))

    # Construct full operator by placing swap_local on the two slots
    # Easiest: use expand_operator
    swap_full = expand_operator(
        swap_local,
        dims=psi_ext.dims[0],
        targets=[mode_index, N]
    )

    psi_swapped = swap_full * psi_ext

    # Now system mode_index contains old content of env (vacuum), env holds old mode_index state.
    # Prepare system mode_index into phi_insert:
    # Do it by replacing that subsystem state *conditionally*? For a pure product on that mode:
    # We can project system mode_index to |0> then apply unitary mapping |0>->|phi>.
    # For truncated HO, build a unitary U such that U|0>=|phi> by Householder in Fock basis.

    U = householder_unitary_send_basis0_to_state(phi_insert)

    U_full = expand_operator(U, dims=psi_swapped.dims[0], targets=[mode_index])
    psi_out = U_full * psi_swapped

    return psi_out.unit()


def householder_unitary_send_basis0_to_state(phi, eps=1e-12):
    """
    Build a unitary U on single-mode Hilbert space such that U|0> = |phi>.
    Deterministic Householder reflection.
    """
    d = phi.shape[0]
    v = phi.full().reshape(-1)
    v = v / (np.linalg.norm(v) + eps)

    e0 = np.zeros(d, dtype=complex)
    e0[0] = 1.0

    if np.linalg.norm(v - e0) < 1e-10:
        return qeye(d)

    u = e0 - v
    u = u / (np.linalg.norm(u) + eps)

    # Householder: H = I - 2 |u><u|
    H = np.eye(d, dtype=complex) - 2.0 * np.outer(u, np.conjugate(u))
    # H e0 = v  (up to global phase)
    return Qobj(H)


import numpy as np

# ----------------------------
# Linear-algebra helpers
# ----------------------------

def _unitary_with_first_column(phi: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Build a d×d unitary U with U[:,0] = phi (normalized).
    Uses a deterministic Gram-Schmidt completion starting from the standard basis.
    """
    phi = np.asarray(phi, dtype=np.complex128).reshape(-1)
    d = phi.size
    nrm = np.linalg.norm(phi)
    if nrm < eps:
        raise ValueError("phi has ~zero norm.")
    phi = phi / nrm

    # Start with columns: [phi, e1, e2, ...] then orthonormalize
    cols = [phi]
    for k in range(d):
        e = np.zeros(d, dtype=np.complex128)
        e[k] = 1.0
        cols.append(e)

    # Classical Gram-Schmidt (fine for moderate d)
    Q = []
    for v in cols:
        w = v.astype(np.complex128, copy=True)
        for q in Q:
            w -= np.vdot(q, w) * q
        nw = np.linalg.norm(w)
        if nw > 1e-12:
            Q.append(w / nw)
        if len(Q) == d:
            break

    if len(Q) != d:
        raise RuntimeError("Failed to construct a full unitary basis (unexpected).")

    U = np.column_stack(Q)
    # Force exact first column
    U[:, 0] = phi
    # Re-orthonormalize remaining columns against phi (tiny cleanup)
    for j in range(1, d):
        U[:, j] -= np.vdot(phi, U[:, j]) * phi
        U[:, j] /= np.linalg.norm(U[:, j])
    return U


def apply_unitary_on_axis(psi: np.ndarray, U: np.ndarray, axis: int) -> np.ndarray:
    """
    Apply a single-mode unitary U on a given tensor axis:
        psi'[k,...] = sum_j U[k,j] * psi[j,...]   along that axis.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    d = U.shape[0]
    if U.shape != (d, d):
        raise ValueError("U must be square (d×d).")
    if psi.shape[axis] != d:
        raise ValueError(f"Axis dim mismatch: psi.shape[{axis}]={psi.shape[axis]} vs d={d}")

    # tensordot contracts U's column index (1) with psi's 'axis'
    out = np.tensordot(U, psi, axes=(1, axis))  # out has U's row axis in front
    # Move that new axis into the original position
    out = np.moveaxis(out, 0, axis)
    return out


# ----------------------------
# Core: "remove mode i, insert phi, purify by moving old mode to env"
# ----------------------------

def replace_mode_with_pure_and_env(
    psi: np.ndarray,
    i: int,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Pure NumPy tensor implementation of:

      - psi is an N-index coefficient tensor with shape (d0, d1, ..., d_{N-1})
      - replace mode i with pure inserted vector phi (length d_i)
      - append an environment mode of dimension d_i
      - swap axes i and env
      - prepare mode i from |0> -> |phi> via a unitary isometry

    Returns:
      psi_new: (d0,...,d_{N-1}, d_i) tensor (env is appended as last axis).
              The system mode i is now |phi>, and the old mode-i content has been moved to env.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    N = psi.ndim
    if not (0 <= i < N):
        raise ValueError("i out of range.")
    di = psi.shape[i]

    phi = np.asarray(phi, dtype=np.complex128).reshape(-1)
    if phi.size != di:
        raise ValueError(f"phi length {phi.size} must equal d_i={di}.")

    # 1) Append env vacuum |0> on a new last axis of size d_i:
    #    psi_env[..., e] = psi[...] * delta_{e,0}
    psi_env = np.zeros(psi.shape + (di,), dtype=np.complex128)
    psi_env[..., 0] = psi

    # 2) Swap axes i and env (env_index = N). After swap:
    #    - axis i becomes the new env axis (currently fixed to |0>)
    #    - last axis becomes the old mode-i content
    psi_swapped = np.swapaxes(psi_env, i, N)  # shape still has N+1 axes

    # 3) "Prepare" step: apply U on axis i with U|0> = |phi>
    U = _unitary_with_first_column(phi)  # first column maps |0> -> |phi>
    psi_prepared = apply_unitary_on_axis(psi_swapped, U, axis=i)

    # Done: mode i now factors as |phi>, env (last axis) carries old mode-i amplitudes
    return psi_prepared


# ----------------------------
# Convenience: verify factorization of the inserted mode
# ----------------------------

def check_insert_is_product(psi_new: np.ndarray, i: int, phi: np.ndarray, tol: float = 1e-10) -> float:
    """
    Returns || psi_new - (phi ⊗ rest) || where rest is obtained by projecting mode i onto phi.
    Small value means mode i is (approximately) a product factor |phi>.
    """
    psi_new = np.asarray(psi_new, dtype=np.complex128)
    phi = np.asarray(phi, dtype=np.complex128).reshape(-1)
    phi = phi / np.linalg.norm(phi)

    # Project mode i onto phi to get "rest"
    # rest[...] = sum_k conj(phi[k]) * psi_new[...,k,...]
    rest = np.tensordot(np.conjugate(phi), psi_new, axes=(0, i))  # removes axis i
    # Rebuild product tensor with phi on axis i
    prod = np.tensordot(phi, rest, axes=0)  # phi axis first
    prod = np.moveaxis(prod, 0, i)
    return float(np.linalg.norm(psi_new - prod))


# ----------------------------
# Bipartite coefficient matrix C_{αβ} (left vs right space)
# ----------------------------

def replace_mode_in_bipartite_C(
    C: np.ndarray,
    left_dims: list[int],
    right_dims: list[int],
    side: str,             # "left" or "right"
    mode_index: int,       # index within that side
    phi: np.ndarray,
):
    """
    C is a bipartite coefficient matrix C_{αβ} with shape (DL, DR),
    where DL = prod(left_dims) and DR = prod(right_dims).
    We interpret the *global* ordering of modes as:
        [left modes..., right modes...]
    and then append env as the final axis (so env "lives on the right" in the matrix view).

    After replacement:
      - left total dimension stays DL
      - right total dimension becomes DR * d_mode (because env axis is appended at the end)

    Returns:
      C_new: shape (DL, DR * d_mode)
      new_left_dims: unchanged
      new_right_dims: right_dims + [d_mode]   (env appended)
    """
    C = np.asarray(C, dtype=np.complex128)
    DL = int(np.prod(left_dims))
    DR = int(np.prod(right_dims))
    if C.shape != (DL, DR):
        raise ValueError(f"C shape {C.shape} must be (DL,DR)=({DL},{DR}).")

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")

    nL = len(left_dims)
    nR = len(right_dims)

    if side == "left":
        if not (0 <= mode_index < nL):
            raise ValueError("mode_index out of range for left.")
        global_i = mode_index
        di = left_dims[mode_index]
    else:
        if not (0 <= mode_index < nR):
            raise ValueError("mode_index out of range for right.")
        global_i = nL + mode_index
        di = right_dims[mode_index]

    phi = np.asarray(phi, dtype=np.complex128).reshape(-1)
    if phi.size != di:
        raise ValueError(f"phi length {phi.size} must equal d_i={di} for the selected mode.")

    # 1) Reshape C into a full tensor of modes: (*left_dims, *right_dims)
    psi = C.reshape(tuple(left_dims) + tuple(right_dims))

    # 2) Apply the same tensor operation
    psi_new = replace_mode_with_pure_and_env(psi, global_i, phi)

    # 3) Reshape back to a bipartite matrix, with env appended on the right
    new_right_dims = list(right_dims) + [di]
    C_new = psi_new.reshape(DL, int(np.prod(new_right_dims)))

    return C_new, list(left_dims), new_right_dims


def phi_squeezed_vacuum(d: int, r: float, theta: float):
    """
    Build phi[n] = <n | R(theta) S(r) |0> in the truncated Fock basis n=0..d-1.

    Convention:
      even n=2k:
        phi[2k] = sqrt((2k)!)/(2^k k! sqrt(cosh r)) * (-e^{i2theta} tanh r)^k
      odd n: 0
    """
    phi = np.zeros(d, dtype=np.complex128)

    ch = math.cosh(r)
    th = math.tanh(r)
    phase = np.exp(1j * 2.0 * theta)

    # Fill even entries
    for k in range((d + 1) // 2):
        n = 2 * k
        # prefactor = sqrt((2k)!)/(2^k k! sqrt(cosh r))
        pref = math.sqrt(math.factorial(2 * k)) / ((2.0 ** k) * math.factorial(k) * math.sqrt(ch))
        phi[n] = pref * ((-phase * th) ** k)

    # Normalize in the truncated space (important when d is small)
    norm = np.linalg.norm(phi)
    if norm == 0:
        raise ValueError("phi norm is zero; check d and r.")
    phi /= norm
    return phi


def fidelity_vs_block_size(
    insert_idx,
    input_ensemble,  # list of (s, theta) you use for fitting
    t_scramble,
    t_couple,
    dt,
    C_back,
    H_coupling,
    H_scramble_side,
    N_modes,
    N_cutoff,
    x_full_env,
    p_full_env,
    center_idx
):
    Fms = []
    Fmf = []

    Vins, Vouts = [], []

    for s, theta in input_ensemble:
        # Run your usual protocol (NO observer) to get global Gamma_final
        C,V_in,V_out = teleportation_protocol(s, 
        theta, 
        insert_idx, 
        t_scramble, 
        t_couple, 
        dt, 
        C_back, 
        H_coupling,
        H_scramble_side,
        N_modes,
        N_cutoff,
        x_full_env,
        p_full_env)
            
        Vins.append(extract_subsystem_covariance(V_in,[insert_idx]))
        Vouts.append(extract_subsystem_covariance(V_out,[center_idx]))

        # --- 5) Fit a single-mode Gaussian channel for this decoded mode ---
    X, Y = fit_gaussian_channel(Vins, Vouts)

    rot1, loss, squeeze, rot2 = decompose_X(X)
    print(center_idx, "rot1=", rot1)
    print("rot2=", rot2)
    print("loss=", loss)
    print("squeeze=", squeeze)
    print("Y=", Y)

    S_dec_symp = decoder_from_X_symplectic(X)  # your preferred
    S_dec_flip = decoder_from_X_flip(X)  # your preferred

    Fs = entanglement_fidelity_gaussian(X, Y, S_dec_symp, subtract_Y=False, r=1.0)
    Ff = entanglement_fidelity_gaussian(X, Y, S_dec_flip, subtract_Y=False, r=1.0)

    Fms.append(Fs)
    Fmf.append(Ff)


    return np.array(Fms), np.array(Fmf)


# 1. Configuration
N_modes = 3  # Modes per side (Ring: 0-1, 1-2, 2-0)
N_cutoff = 8  # Lowered for N=3 to keep memory usage safe
beta = 1  # Inverse temperature (Tuning this is critical)
lam = 0.0
#chi = 0.15  # Non-Gaussianity (x^3)
g_nn = 5  # Ring coupling strength
g_int = 1  # L-R coupling strength
t_scramble = 2  # Scrambling time (tune this for peak fidelity)
t_couple = 3
keep_states = 30
insert_idx = 1
k = 5
m_squared = 13
omega0 = np.sqrt(m_squared + 2 * k)
mu_x = omega0 / 2
mu_p = 1 / (2 * omega0)
dt = .05

N = N_modes

# build x_i and p_i for all modes
a = destroy(N_cutoff)
I = qeye(N_cutoff)
x_loc = (a + a.dag()) / np.sqrt(2)
p_loc = (a - a.dag()) / (1j * np.sqrt(2))


x_ops = [wrap(x_loc, i, N_modes, I) for i in range(N_modes)]
p_ops = [wrap(p_loc, i, N_modes, I) for i in range(N_modes)]


# 3. Construct Ring Hamiltonians
H0_side = 0
for i in range(N_modes):
    H0_side += 0.5 * p_ops[i] ** 2 + 0.5 * m_squared * x_ops[i] ** 2

# ring couplings: left
for i in range(N_modes):
    j = (i + 1) % N_modes
    H0_side += 0.5 * k * (x_ops[i] - x_ops[j]) ** 2

Hng_side = 0
for i in range(N_modes):  # say: only scramble left
    # Hng += chi * x_ops[i]**3
    Hng_side += lam * x_ops[i] ** 4

H_scramble_side = H0_side + Hng_side

# 4. Generate the TFD State
# build TFD from H0_side eigenstates (correct)
evals, evecs = H0_side.eigenstates(eigvals=keep_states)
evals = np.array(evals, float)


C_tfd =build_TFD_C_fast(evals, evecs, beta, keep_states=keep_states)

# evolve backwards
U_back_side = ( 1j * H_scramble_side * t_scramble).expm()
U_back = U_back_side.full()
C_back = apply_left(C_tfd, U_back)


#coupling hamiltonian 

N_total = 2 * N_modes

x_full = [wrap_full(x_loc, i, N_total, I) for i in range(N_total)]
p_full = [wrap_full(p_loc, i, N_total, I) for i in range(N_total)]


#x_ops, p_ops = build_quadrature_ops(N_modes, N_cutoff)


H_int = 0
for i in range(N_modes):
    if i == insert_idx:
        continue
    Li = i
    Ri = i + N_modes
    H_int += mu_x * x_full[Li] * x_full[Ri]
    H_int += mu_p * p_full[Li] * p_full[Ri]

H_coupling = H_int


N_total_env = 2 * N_modes + 1

x_full_env = [wrap_full(x_loc, i, N_total_env, I) for i in range(N_total_env)]
p_full_env = [wrap_full(p_loc, i, N_total_env, I) for i in range(N_total_env)]



site_fidelities_symp = []
site_fidelities_flip = []
wigner_fidelities = []

block_sizes = [1]
# block_sizes = [1,2,4,6,8,10]

Ss = np.linspace(-1.5, 1.5, 4)
Thetas = np.linspace(0, 2 * np.pi, 3, endpoint=False)
input_ensemble = [(s, th) for s in Ss for th in Thetas]  # 120 points, deterministic

sites = np.arange(N, 2 * N)

for f in range(len(sites)):
    #Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    #plt.plot(block_sizes,Fs,label=sites[f])
    Fs,Ff= fidelity_vs_block_size(
    insert_idx,
    input_ensemble,
    t_scramble,
    t_couple,
    dt,
    C_back,
    H_coupling,
    H_scramble_side,
    N_modes,
    N_cutoff,
    x_full_env=x_full_env,
    p_full_env=p_full_env,
    center_idx=sites[f])
    site_fidelities_symp.append(Fs)
    site_fidelities_flip.append(Ff)

"""
plt.xlabel("decoder block size")
plt.ylabel("fidelity")
plt.legend()
plt.show()
"""



plt.plot(sites,site_fidelities_symp,label="symplectic")
plt.plot(sites,site_fidelities_flip,label="allow flip")
plt.xlabel("site")
plt.ylabel("fidelity")
plt.legend()
plt.show()

"""
times_evolve = np.linspace(3.4, 3.7, 12)
time_fidelities_symp = []
time_fidelities_flip = []
for t in range(len(times_evolve)):
    # Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    # plt.plot(block_sizes,Fs,label=sites[f])
    Fs, Ff = fidelity_vs_block_size(
        block_sizes=block_sizes,
        obs_idx=2 * N,
        insert_idx=insert_idx,
        center_idx=insert_idx,
        input_ensemble=input_ensemble,  # list of (s, theta) you use for fitting
        t0=times_evolve[t],
        t_couple=t_couple,
        dt=dt,
        state_TFD=state_TFD,
        H_coupling=H_coupling,
        params=params,
    )
    time_fidelities_symp.append(Fs)
    time_fidelities_flip.append(Ff)

plt.plot(times_evolve, time_fidelities_symp, label="symplectic")
plt.plot(times_evolve, time_fidelities_flip, label="allow flip")
plt.xlabel("times")
plt.ylabel("fidelity")
plt.legend()
plt.show()
"""
print("done")
