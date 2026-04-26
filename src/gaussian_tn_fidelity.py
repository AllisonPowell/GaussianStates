# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, expm, sqrtm, schur, block_diag, eigh, det, polar
from thewalrus.symplectic import xpxp_to_xxpp, sympmat
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
import math

import numpy as np
from scipy.linalg import eigh, expm

from tenpy.networks.site import BosonSite
from tenpy.networks.mps import MPS
from tenpy.linalg import np_conserved as npc

import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='LOG.log',
    filemode='w'
)

def log_print(*args):
    message = " ".join(map(str, args))
    logging.info(message)


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
    expand_operator,
    entropy_vn
)

def symplectic_form(n):
    #Returns the 2n × 2n symplectic form Omega
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


# ============================================================
# Bosonic local operators
# ============================================================

def local_boson_ops(Nmax):

    d = Nmax + 1

    b = np.zeros((d,d),dtype=complex)

    for n in range(1,d):
        b[n-1,n] = np.sqrt(n)

    bd = b.conj().T

    x = (b + bd)/np.sqrt(2)
    p = (b - bd)/(1j*np.sqrt(2))

    I = np.eye(d)

    return dict(x=x,p=p,b=b,bd=bd,I=I)


# ============================================================
# Kronecker helpers
# ============================================================

def kron_all(ops):

    out = ops[0]

    for A in ops[1:]:
        out = np.kron(out,A)

    return out


def embed_one_site(op,site,L,d):

    ops = [np.eye(d) for _ in range(L)]
    ops[site] = op

    return kron_all(ops)


def embed_two_site(op, i, j, L, d):
    """
    Embed a 2-site operator op acting on sites i,j into an L-site Hilbert space.
    op should be shape (d*d, d*d), with basis ordering |a,b>.
    """
    if i == j:
        raise ValueError("i and j must be distinct.")

    # If swapped, also swap tensor-leg ordering of the operator
    if i > j:
        op4 = op.reshape(d, d, d, d)          # (out_i, out_j, in_i, in_j)
        op4 = np.transpose(op4, (1, 0, 3, 2)) # swap sites
        op = op4.reshape(d*d, d*d)
        i, j = j, i

    op4 = op.reshape(d, d, d, d)  # (out_i, out_j, in_i, in_j)

    I = np.eye(d, dtype=complex)
    O = np.zeros((d**L, d**L), dtype=complex)

    for out_i in range(d):
        for out_j in range(d):
            for in_i in range(d):
                for in_j in range(d):
                    coeff = op4[out_i, out_j, in_i, in_j]
                    if abs(coeff) < 1e-15:
                        continue

                    factors = [I.copy() for _ in range(L)]

                    Ei = np.zeros((d, d), dtype=complex)
                    Ej = np.zeros((d, d), dtype=complex)

                    Ei[out_i, in_i] = 1.0   # |out_i><in_i|
                    Ej[out_j, in_j] = 1.0   # |out_j><in_j|

                    factors[i] = Ei
                    factors[j] = Ej

                    O += coeff * kron_all(factors)

    return O

# ============================================================
# Ring Hamiltonian
# ============================================================

def build_ring_H(Nsites,Nmax,m2,k,lam):

    ops = local_boson_ops(Nmax)

    x = ops["x"]
    p = ops["p"]

    d = Nmax+1
    L = Nsites

    H = np.zeros((d**L,d**L),dtype=complex)

    # onsite terms

    for i in range(L):

        H += 0.5*embed_one_site(p@p,i,L,d)
        H += 0.5*m2*embed_one_site(x@x,i,L,d)

        if lam>0:
            H += lam*embed_one_site(x@x@x@x,i,L,d)

    # ring nearest neighbour

    x2 = x@x
    xx = np.kron(x,x)

    for i in range(L):

        j = (i+1)%L

        H += 0.5*k*embed_one_site(x2,i,L,d)
        H += 0.5*k*embed_one_site(x2,j,L,d)
        H += -k*embed_two_site(xx,i,j,L,d)

    return 0.5*(H+H.conj().T)


# ============================================================
# TFD construction
# ============================================================

def build_tfd_tensor(H_side,Nsites,Nmax,beta):

    d = Nmax+1

    evals,evecs = eigh(H_side)

    evals -= evals.min()

    w = np.exp(-beta*evals)

    Z = np.sum(w)

    C = evecs @ np.diag(np.sqrt(w/Z)) @ evecs.conj().T

    psi = C.reshape([d]*Nsites + [d]*Nsites)

    return psi


# ============================================================
# Insert message with environment
# ============================================================

def insert_with_env(psi_tensor,insert_idx,phi):

    d = psi_tensor.shape[0]

    psi_env = np.zeros(psi_tensor.shape+(d,),dtype=complex)

    psi_env[...,0] = psi_tensor

    psi_swapped = np.swapaxes(psi_env,insert_idx,psi_env.ndim-1)

    U = np.eye(d)

    U[:,0] = phi

    psi_out = np.tensordot(U,psi_swapped,axes=(1,insert_idx))
    psi_out = np.moveaxis(psi_out,0,insert_idx)

    return psi_out

def insert_state_replace(psi_tensor, insert_idx, phi):

    d = psi_tensor.shape[0]

    # move target axis to front
    psi_perm = np.moveaxis(psi_tensor, insert_idx, 0)

    # contract with phi
    psi_new = np.tensordot(phi, psi_perm, axes=(0,0))

    # reinsert axis
    psi_new = np.expand_dims(psi_new, axis=0)
    psi_new = np.moveaxis(psi_new, 0, insert_idx)

    return psi_new

# ============================================================
# Convert tensor → MPS
# ============================================================

def tensor_to_mps(psi_tensor,Nmax):

    nsites = psi_tensor.ndim

    sites = [BosonSite(Nmax=Nmax,conserve=None) for _ in range(nsites)]

    psi_npc = npc.Array.from_ndarray_trivial(
        psi_tensor,
        labels=[f'p{i}' for i in range(nsites)]
    )

    psi = MPS.from_full(sites,psi_npc,normalize=True)

    psi.canonical_form()

    return psi


# ============================================================
# Trotter gates
# ============================================================

def onsite_unitary_old(Nmax,dt,m2,k,lam):

    ops = local_boson_ops(Nmax)

    x = ops["x"]
    p = ops["p"]

    h = 0.5*p@p + 0.5*(m2+2*k)*x@x + lam*x@x@x@x

    return expm(-1j*dt*h)


def bond_unitary_old(Nmax,dt,k):

    ops = local_boson_ops(Nmax)

    x = ops["x"]

    d = Nmax+1

    h = -k*np.kron(x,x)

    return expm(-1j*dt*h)

from scipy.linalg import expm
import numpy as np

def coupling_bond_unitary(Nmax, dt, g, m_squared, k):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]

    omega0 = np.sqrt(m_squared + 2 * k)

    H_pair = g * (
        (omega0 / 2.0) * np.kron(x, x)
        + (1.0 / (2.0 * omega0)) * np.kron(p, p)
    )

    Uc = expm(-1j * dt * H_pair)
    return Uc

# ============================================================
# Apply gates
# ============================================================

def apply_one_site_old(psi,i,U):

    op = npc.Array.from_ndarray_trivial(U,labels=['p','p*'])

    psi.apply_local_op(i,op,unitary=True)

    return psi


def apply_two_site_adjacent(psi,i,U):

    d = int(np.sqrt(U.shape[0]))

    op = npc.Array.from_ndarray_trivial(
        U.reshape(d,d,d,d),
        labels=['p0','p1','p0*','p1*']
    )

    psi.apply_local_op(i,op,unitary=True)

    return psi



def SWAP_gate(Nmax):

    d = Nmax + 1

    U = np.zeros((d*d, d*d), dtype=complex)

    for i in range(d):
        for j in range(d):

            in_index  = i*d + j
            out_index = j*d + i

            U[out_index, in_index] = 1.0

    return U 


from scipy.linalg import expm

def gaussian_mode(Nmax, r=0.5, theta=0.0):

    ops = local_boson_ops(Nmax)

    b = ops["b"]
    bd = ops["bd"]

    d = Nmax+1

    vacuum = np.zeros(d)
    vacuum[0] = 1

    # squeezing
    Hs = 0.5*(b@b - bd@bd)
    S = expm(r*Hs)

    # rotation
    n = bd@b
    R = expm(-1j*theta*n)

    U = R @ S

    phi = U @ vacuum

    return phi / np.linalg.norm(phi)


def covariance_matrix(psi_tensor, Nsites, Nmax):

    ops = local_boson_ops(Nmax)

    x = ops["x"]
    p = ops["p"]

    d = Nmax+1
    L = Nsites

    psi = psi_tensor.reshape(d**L)

    # build operator list
    R_ops = []

    for i in range(L):
        R_ops.append(embed_one_site(x,i,L,d))
        R_ops.append(embed_one_site(p,i,L,d))

    n = 2*L

    Gamma = np.zeros((n,n))

    # compute means
    means = np.zeros(n)

    for i in range(n):
        means[i] = np.real(np.vdot(psi, R_ops[i] @ psi))

    # compute covariance
    for i in range(n):
        for j in range(n):

            A = R_ops[i] @ R_ops[j]
            B = R_ops[j] @ R_ops[i]

            val = 0.5*np.vdot(psi,(A+B)@psi)

            Gamma[i,j] = np.real(val) - means[i]*means[j]

    return Gamma

import numpy as np

def one_site_rho_matrix(psi, i):
    rho = psi.get_rho_segment([i]).to_ndarray()
    return rho.reshape(rho.shape[0], rho.shape[0])

def two_site_rho_matrix(psi, i, j):
    rho = psi.get_rho_segment([i, j]).to_ndarray()

    # reorder to matrix form
    # expected raw shape is (d,d,d,d)
    d = rho.shape[0]
    rho = np.transpose(rho, (0, 1, 2, 3)).reshape(d*d, d*d)
    return rho

def covariance_matrix_from_mps(psi, Nsites, Nmax):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]
    d = Nmax + 1

    local_ops = [x, p]
    nquad = 2 * Nsites
    Gamma = np.zeros((nquad, nquad), dtype=float)
    means = np.zeros(nquad, dtype=float)

    def op_for_index(a):
        site = a % Nsites
        kind = 0 if a < Nsites else 1   # 0 -> x, 1 -> p
        return site, local_ops[kind]

    # one-point functions
    for a in range(nquad):
        i, Oi = op_for_index(a)
        rho_i = one_site_rho_matrix(psi, i)
        means[a] = np.real(np.trace(rho_i @ Oi))

    # covariance entries
    for a in range(nquad):
        i, Oi = op_for_index(a)
        for b in range(nquad):
            j, Oj = op_for_index(b)

            if i == j:
                rho_i = one_site_rho_matrix(psi, i)
                sym_op = 0.5 * (Oi @ Oj + Oj @ Oi)
                val = np.real(np.trace(rho_i @ sym_op))
            else:
                rho_ij = two_site_rho_matrix(psi, i, j)
                Oij = np.kron(Oi, Oj)
                val = np.real(np.trace(rho_ij @ Oij))

            Gamma[a, b] = val - means[a] * means[b]

    return 0.5 * (Gamma + Gamma.T)

def covariance_from_single_mode_state(phi, Nmax):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]

    phi = phi / np.linalg.norm(phi)

    mx = np.real(np.vdot(phi, x @ phi))
    mp = np.real(np.vdot(phi, p @ phi))

    xx = np.real(np.vdot(phi, x @ x @ phi))
    pp = np.real(np.vdot(phi, p @ p @ phi))
    xp = np.real(np.vdot(phi, 0.5 * (x @ p + p @ x) @ phi))

    V = np.array([
        [xx - mx*mx, xp - mx*mp],
        [xp - mx*mp, pp - mp*mp]
    ])
    return 0.5 * (V + V.T)

def apply_two_site_nonlocal(psi, i, j, U,Nmax):

    if j < i:
        i, j = j, i

    swap = SWAP_gate(Nmax)   # precompute if possible

    # move j next to i
    for k in range(j-1, i, -1):
        apply_two_site_adjacent(psi, k, swap)

    # apply desired gate
    apply_two_site_adjacent(psi, i, U)

    # swap back
    for k in range(i+1, j):
        apply_two_site_adjacent(psi, k, swap)

    return psi

def apply_wormhole_coupling(
        psi,
        N,
        insert_idx,
        t_couple,
        dt,
        Nmax,
        g,
        m_squared,
        k):

    steps_couple = int(t_couple/dt)

    Uc = coupling_bond_unitary(Nmax, dt, g, m_squared, k)

    for _ in range(steps_couple):

        for i in range(N):

            if i == insert_idx:
                continue

            left = i
            right = N + i

            apply_two_site_nonlocal(psi, left, right, Uc, Nmax)

    psi.canonical_form()

    return psi


# ============================================================
# Protocol
# ============================================================

def teleportation_protocol_old(s,theta,N,Nmax,insert_idx,psi_tensor):

    # insert gaussian mode

    phi = gaussian_mode(
        Nmax=Nmax,
        r=s,       # squeezing strength
        theta=theta    # phase rotation
    )

    psi_insert = insert_with_env(psi_tensor,insert_idx,phi)

    # convert to MPS

    psi = tensor_to_mps(psi_insert,Nmax)

    # forward evolve left

    U1 = onsite_unitary(Nmax,dt,m2,k,lam)
    U2 = bond_unitary(Nmax,dt,k)

    steps = int(t_scramble/dt)

    for _ in range(steps):

        for i in range(N):
            apply_one_site(psi,i,U1)

        for i in range(N):
            apply_two_site_adjacent(psi,i,U2)   # ring bond

    # coupling
    apply_wormhole_coupling(
        psi,
        N,
        insert_idx,
        t_couple,
        dt,
        Nmax,
        g,
        m2,
        k)

    # evolve right

    for _ in range(steps):

        for i in range(N,2*N):
            apply_one_site(psi,i,U1)
        for i in range(N,2*N):
            apply_two_site_adjacent(psi,i,U2)
 

    return psi

def onsite_unitary(Nmax, dt, m2, k, lam):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]

    H = 0.5*p@p + 0.5*(m2+2*k)*x@x

    if lam > 0:
        H += lam*x@x@x@x

    U = expm(-1j*dt*H)
    return U


def bond_unitary(Nmax, dt, k):
    ops = local_boson_ops(Nmax)
    x = ops["x"]

    H = -k*np.kron(x,x)

    U = expm(-1j*dt*H)
    return U

def apply_one_site(psi, i, U):

    d = U.shape[0]

    op = npc.Array.from_ndarray_trivial(
        U.reshape(d, d),
        labels=['p', 'p*'] 
    )

    psi.apply_local_op(i, op, unitary=True)

def apply_coupling_bond(psi, i, j, U, Nmax, chi_max=256, svd_cut=1e-10):

    if j < i:
        i,j = j,i

    # bring j next to i
    for k in range(j-1, i, -1):
        apply_two_site_adjacent(psi, k, SWAP_gate(Nmax))

    apply_two_site_adjacent(psi, i, U)

    # move back
    for k in range(i+1, j):
        apply_two_site_adjacent(psi, k, SWAP_gate(Nmax))



def apply_ring_bond(psi, i, j, U, N, Nmax, chi_max=256, svd_cut=1e-10):

    # ordinary adjacent bond on the line
    if abs(i - j) == 1:
        apply_two_site_adjacent(psi, min(i, j), U)
        return

    # wraparound ring bond
    if abs(i - j) == N - 1:
        if i < j:
            left = i
            right = j
        else:
            left = j
            right = i

        # bring right next to left
        for k in range(right - 1, left, -1):
            apply_two_site_adjacent(psi, k, SWAP_gate(Nmax))

        apply_two_site_adjacent(psi, left, U)

        # move back
        for k in range(left + 1, right):
            apply_two_site_adjacent(psi, k, SWAP_gate(Nmax))

        return

    raise ValueError(f"Sites {i},{j} are not nearest neighbors on a ring of length {N}.")



def tebd_step_ring(psi, start, end, U1, U2,N,Nmax):

    # onsite step
    for i in range(start,end):
        apply_one_site(psi,i,U1)

    # even bonds
    for i in range(start,end-1,2):
        apply_two_site_adjacent(psi,i,U2)

    # odd bonds
    for i in range(start+1,end-1,2):
        apply_two_site_adjacent(psi,i,U2)

    # ring bond
    apply_ring_bond(psi,end-1,start,U2,N,Nmax)

def wormhole_unitary(Nmax, dt, g, omega0):

    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]

    H = g*(0.5*omega0*np.kron(x,x)
           + 0.5/omega0*np.kron(p,p))

    return expm(-1j*dt*H)

def tebd_step_coupled(psi, N, insert_idx, U1, U2, Uc,Nmax):

    # onsite evolution (both sides)
    for i in range(2*N):
        apply_one_site(psi, i, U1)

    # left bonds
    for i in range(0, N-1):
        apply_two_site_adjacent(psi, i, U2)

    apply_ring_bond(psi, N-1, 0, U2,N,Nmax)

    # right bonds
    for i in range(N, 2*N-1):
        apply_two_site_adjacent(psi, i, U2)

    apply_ring_bond(psi, 2*N-1, N, U2,N,Nmax)

    # wormhole coupling
    for i in range(N):
        if i == insert_idx:
            continue

        L = i
        R = N + i

        apply_coupling_bond(psi, L, R, Uc,Nmax)

def evolve_with_coupling(
        psi,
        N,
        insert_idx,
        t_couple,
        dt,
        Nmax,
        g,
        m2,
        k,
        lam):

    omega0 = np.sqrt(m2 + 2*k)

    U1 = onsite_unitary(Nmax, dt, m2, k, lam)
    U2 = bond_unitary(Nmax, dt, k)
    Uc = wormhole_unitary(Nmax, dt, g, omega0)

    steps = int(t_couple/dt)

    for _ in range(steps):

        tebd_step_coupled(
            psi,
            N,
            insert_idx,
            U1,
            U2,
            Uc,
            Nmax
        )





def teleportation_protocol(s, theta, N, Nmax, insert_idx, psi_tensor):
    N=3
    beta = 1.0

    m2 = 13
    k = 5
    lam = 0.0

    g = 1

    dt = 0.05
    t_scramble = 4
    t_couple = 3

    # insert gaussian mode
    phi = gaussian_mode(Nmax, r=s, theta=theta)

    psi_insert = insert_with_env(psi_tensor, insert_idx, phi)
    #insert_state_replace(psi_tensor, insert_idx, phi)
    # convert to MPS
    psi = tensor_to_mps(psi_insert, Nmax)

    # TEBD operators
    U1 = onsite_unitary(Nmax, dt, m2, k, lam)
    U2 = bond_unitary(Nmax, dt, k)

    steps = int(t_scramble/dt)
    
    # forward evolve left
    for _ in range(steps):
        tebd_step_ring(psi, 0, N, U1, U2,N,Nmax)

    # wormhole coupling
    evolve_with_coupling(
    psi,
    N,
    insert_idx,
    t_couple,
    dt,
    Nmax,
    g,
    m2,
    k,
    lam)

    # evolve right
    for _ in range(steps):
        tebd_step_ring(psi, N, 2*N, U1, U2,N,Nmax)

    return psi


def fidelity_vs_site_old(
    insert_idx,
    input_ensemble,  # list of (s, theta) you use for fitting
    N,
    Nmax,
    psi_tensor):


    Vins = []

    Vouts = [[] for _ in range(N)]


    for s, theta in input_ensemble:
        # Run your usual protocol (NO observer) to get global Gamma_final
        psi_out = teleportation_protocol(s,theta,N,Nmax,insert_idx,psi_tensor)
        Vout = covariance_matrix_from_mps(psi_out,2*N+1,Nmax)   
            
        Vins.append(covariance_from_single_mode_state(gaussian_mode(Nmax=Nmax,r=s,theta=theta),Nmax))
        for i in range(N):
            Vouts[i].append(extract_subsystem_covariance(Vout,[i+N+1]))
            print(i,Vouts[i][-1])
        
        #Vouts3.append(extract_subsystem_covariance(covariance_matrix_from_mps(psi_out,N,Nmax),[3]))
        #Vouts4.append(extract_subsystem_covariance(covariance_matrix_from_mps(psi_out,N,Nmax),[4]))
        #Vouts5.append(extract_subsystem_covariance(covariance_matrix_from_mps(psi_out,N,Nmax),[5]))


        # --- 5) Fit a single-mode Gaussian channel for this decoded mode ---


    #X3, Y3 = fit_gaussian_channel(Vins, Vouts3)
    #X4, Y4 = fit_gaussian_channel(Vins, Vouts4)
    #X5, Y5 = fit_gaussian_channel(Vins, Vouts4)

    fid_symp = []
    fid_flip = []
    for i in range(N):
        print(f"Vouts[{i}]={Vouts[0]}")

    for i in range(N):
        X, Y = fit_gaussian_channel(Vins, Vouts[i])
        rot1,loss,squeeze,rot2 = decompose_X(X)
        print(i+N)
        print(f"rot1={rot1}")
        print(f"rot2={rot2}")
        print(f"loss={loss}")
        print(f"squeeze={squeeze}")
        print(f"Y={Y}")

        S_dec_symp = decoder_from_X_symplectic(X)  
        S_dec_flip = decoder_from_X_flip(X) 

        Fs = entanglement_fidelity_gaussian(X, Y, S_dec_symp, subtract_Y=False, r=1.0)
        Ff = entanglement_fidelity_gaussian(X, Y, S_dec_flip, subtract_Y=False, r=1.0)

        fid_symp.append(Fs)
        fid_flip.append(Ff)

        print(f"fid_flip_3={Ff}")
        print(f"fid_symp_3={Fs}")

    """
    rot1_3, loss_3, squeeze_3, rot2_3 = decompose_X(X3)
    print(3)
    print(f"rot1={rot1_3}")
    print(f"rot2={rot2_3}")
    print(f"loss={loss_3}")
    print(f"squeeze={squeeze_3}")
    print(f"Y={Y3}")

    rot1_4, loss_4, squeeze_4, rot2_4 = decompose_X(X4)
    print(4)
    print(f"rot1={rot1_4}")
    print(f"rot2={rot2_4}")
    print(f"loss={loss_4}")
    print(f"squeeze={squeeze_4}")
    print(f"Y={Y4}")

    rot1_5, loss_5, squeeze_5, rot2_5 = decompose_X(X5)
    print(5)
    print(f"rot1={rot1_5}")
    print(f"rot2={rot2_5}")
    print(f"loss={loss_5}")
    print(f"squeeze={squeeze_5}")
    print(f"Y={Y5}")


  
    S_dec_symp_3 = decoder_from_X_symplectic(X3)  # your preferred
    S_dec_flip_3 = decoder_from_X_flip(X3)  # your preferred

    Fs3 = entanglement_fidelity_gaussian(X3, Y3, S_dec_symp_3, subtract_Y=False, r=1.0)
    Ff3 = entanglement_fidelity_gaussian(X3, Y3, S_dec_flip_3, subtract_Y=False, r=1.0)

    print(f"fid_flip_3={Ff3}")
    print(f"fid_symp_3={Fs3}")

    S_dec_symp_4 = decoder_from_X_symplectic(X4)  # your preferred
    S_dec_flip_4 = decoder_from_X_flip(X4)  # your preferred

    Fs4 = entanglement_fidelity_gaussian(X4, Y4, S_dec_symp_4, subtract_Y=False, r=1.0)
    Ff4 = entanglement_fidelity_gaussian(X4, Y4, S_dec_flip_4, subtract_Y=False, r=1.0)

    print(f"fid_flip_3={Ff4}")
    print(f"fid_symp_3={Fs4}")


    S_dec_symp_5 = decoder_from_X_symplectic(X5)  # your preferred
    S_dec_flip_5 = decoder_from_X_flip(X5)  # your preferred

    Fs5 = entanglement_fidelity_gaussian(X5, Y5, S_dec_symp_5, subtract_Y=False, r=1.0)
    Ff5 = entanglement_fidelity_gaussian(X5, Y5, S_dec_flip_5, subtract_Y=False, r=1.0)

    print(f"fid_flip_3={Ff5}")
    print(f"fid_symp_3={Fs5}")

    """

    return fid_symp,fid_flip


N=3
Nmax=9
beta = 1.0

m2 = 13
k = 5
lam = 0.0

g = 1

dt = 0.05
t_scramble = 4
t_couple = 3

# build dense ring Hamiltonian
H_quad = build_ring_H(N, Nmax, m2, k, lam)

# build TFD
psi_tensor = build_tfd_tensor(H_quad, N, Nmax, beta)

# backward modular evolution
U_back = expm(1j*t_scramble*H_quad)

psi_left = psi_tensor.reshape((Nmax+1)**N,-1)
psi_left = U_back @ psi_left
psi_tensor = psi_left.reshape([Nmax+1]*(2*N))



Ss = np.linspace(-1, 1, 4)
Thetas = np.linspace(0, 2 * np.pi, 3, endpoint=False)
input_ensemble = [(s, th) for s in Ss for th in Thetas]  # 120 points, deterministic

sites = np.arange(N, 2 * N)
#site_fidelities_symp = []
#site_fidelities_flip = []




# for f in range(len(sites)):
    #Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    #plt.plot(block_sizes,Fs,label=sites[f])
Fs,Ff= fidelity_vs_site(
    insert_idx=1,
    input_ensemble=input_ensemble,  # list of (s, theta) you use for fitting
    N=N,
    Nmax=Nmax,
    psi_tensor=psi_tensor
    )

print("Processing complete. Check LOG.log for details.")

plt.plot(sites,Fs,label="symplectic")
plt.plot(sites,Ff,label="allow flip")
plt.xlabel("site")
plt.ylabel("fidelity")
plt.legend()
plt.savefig("plots/site_vs_fidelity.pdf")
# plt.show()

"""
plt.xlabel("decoder block size")
plt.ylabel("fidelity")
plt.legend()
plt.show()
"""





"""
times_couple = np.linspace(1, 3, 5)
time_fidelities_symp = []
time_fidelities_flip = []

for t in range(len(times_couple)):
    # Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    # plt.plot(block_sizes,Fs,label=sites[f])
    Fs,Ff= fidelity_vs_block_size(
    insert_idx,
    input_ensemble,
    t_scramble,
    times_couple[t],
    dt,
    C_back,
    H_coupling,
    H_scramble_side,
    N_modes,
    N_cutoff,
    x_full_env=x_full_env,
    p_full_env=p_full_env,
    center_idx=3)
    time_fidelities_symp.append(Fs)
    time_fidelities_flip.append(Ff)

plt.plot(times_couple, time_fidelities_symp, label="symplectic")
plt.plot(times_couple, time_fidelities_flip, label="allow flip")
plt.xlabel("times")
plt.ylabel("fidelity")
plt.legend()
#plt.show()
plt.savefig("plots/time_vs_fidelity.pdf")
"""

log_print("done")

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, expm, sqrtm, schur, block_diag, eigh, det, polar
from thewalrus.symplectic import xpxp_to_xxpp, sympmat
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
import math

import numpy as np
from scipy.linalg import eigh, expm

from tenpy.networks.site import BosonSite
from tenpy.networks.mps import MPS
from tenpy.linalg import np_conserved as npc

import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='LOG.log',
    filemode='w'
)

def log_print(*args):
    message = " ".join(map(str, args))
    logging.info(message)


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
    expand_operator,
    entropy_vn
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


# ============================================================
# Bosonic local operators
# ============================================================

def local_boson_ops(Nmax):

    d = Nmax + 1

    b = np.zeros((d,d),dtype=complex)

    for n in range(1,d):
        b[n-1,n] = np.sqrt(n)

    bd = b.conj().T

    x = (b + bd)/np.sqrt(2)
    p = (b - bd)/(1j*np.sqrt(2))

    I = np.eye(d)

    return dict(x=x,p=p,b=b,bd=bd,I=I)


# ============================================================
# Kronecker helpers
# ============================================================

def kron_all(ops):

    out = ops[0]

    for A in ops[1:]:
        out = np.kron(out,A)

    return out


def embed_one_site(op,site,L,d):

    ops = [np.eye(d) for _ in range(L)]
    ops[site] = op

    return kron_all(ops)


def embed_two_site(op, i, j, L, d):
    """
    Embed a 2-site operator op acting on sites i,j into an L-site Hilbert space.
    op should be shape (d*d, d*d), with basis ordering |a,b>.
    """
    if i == j:
        raise ValueError("i and j must be distinct.")

    # If swapped, also swap tensor-leg ordering of the operator
    if i > j:
        op4 = op.reshape(d, d, d, d)          # (out_i, out_j, in_i, in_j)
        op4 = np.transpose(op4, (1, 0, 3, 2)) # swap sites
        op = op4.reshape(d*d, d*d)
        i, j = j, i

    op4 = op.reshape(d, d, d, d)  # (out_i, out_j, in_i, in_j)

    I = np.eye(d, dtype=complex)
    O = np.zeros((d**L, d**L), dtype=complex)

    for out_i in range(d):
        for out_j in range(d):
            for in_i in range(d):
                for in_j in range(d):
                    coeff = op4[out_i, out_j, in_i, in_j]
                    if abs(coeff) < 1e-15:
                        continue

                    factors = [I.copy() for _ in range(L)]

                    Ei = np.zeros((d, d), dtype=complex)
                    Ej = np.zeros((d, d), dtype=complex)

                    Ei[out_i, in_i] = 1.0   # |out_i><in_i|
                    Ej[out_j, in_j] = 1.0   # |out_j><in_j|

                    factors[i] = Ei
                    factors[j] = Ej

                    O += coeff * kron_all(factors)

    return O

# ============================================================
# Ring Hamiltonian
# ============================================================

def build_ring_H(Nsites,Nmax,m2,k,lam):

    ops = local_boson_ops(Nmax)

    x = ops["x"]
    p = ops["p"]

    d = Nmax+1
    L = Nsites

    H = np.zeros((d**L,d**L),dtype=complex)

    # onsite terms

    for i in range(L):

        H += 0.5*embed_one_site(p@p,i,L,d)
        H += 0.5*m2*embed_one_site(x@x,i,L,d)

        if lam>0:
            H += lam*embed_one_site(x@x@x@x,i,L,d)

    # ring nearest neighbour

    x2 = x@x
    xx = np.kron(x,x)

    for i in range(L):

        j = (i+1)%L

        H += 0.5*k*embed_one_site(x2,i,L,d)
        H += 0.5*k*embed_one_site(x2,j,L,d)
        H += -k*embed_two_site(xx,i,j,L,d)

    return 0.5*(H+H.conj().T)


# ============================================================
# TFD construction
# ============================================================

def build_tfd_tensor(H_side,Nsites,Nmax,beta):

    d = Nmax+1

    evals,evecs = eigh(H_side)

    evals -= evals.min()

    w = np.exp(-beta*evals)

    Z = np.sum(w)

    C = evecs @ np.diag(np.sqrt(w/Z)) @ evecs.conj().T

    psi = C.reshape([d]*Nsites + [d]*Nsites)

    return psi


# ============================================================
# Insert message with environment
# ============================================================

def insert_with_env(psi_tensor,insert_idx,phi):

    d = psi_tensor.shape[0]

    psi_env = np.zeros(psi_tensor.shape+(d,),dtype=complex)

    psi_env[...,0] = psi_tensor

    psi_swapped = np.swapaxes(psi_env,insert_idx,psi_env.ndim-1)

    U = np.eye(d)

    U[:,0] = phi

    psi_out = np.tensordot(U,psi_swapped,axes=(1,insert_idx))
    psi_out = np.moveaxis(psi_out,0,insert_idx)

    return psi_out

def insert_state_replace(psi_tensor, insert_idx, phi):

    d = psi_tensor.shape[0]

    # move target axis to front
    psi_perm = np.moveaxis(psi_tensor, insert_idx, 0)

    # contract with phi
    psi_new = np.tensordot(phi, psi_perm, axes=(0,0))

    # reinsert axis
    psi_new = np.expand_dims(psi_new, axis=0)
    psi_new = np.moveaxis(psi_new, 0, insert_idx)

    return psi_new

# ============================================================
# Convert tensor → MPS
# ============================================================

def tensor_to_mps(psi_tensor,Nmax):

    nsites = psi_tensor.ndim

    sites = [BosonSite(Nmax=Nmax,conserve=None) for _ in range(nsites)]

    psi_npc = npc.Array.from_ndarray_trivial(
        psi_tensor,
        labels=[f'p{i}' for i in range(nsites)]
    )

    psi = MPS.from_full(sites,psi_npc,normalize=True)

    psi.canonical_form()

    return psi


# ============================================================
# Trotter gates
# ============================================================

def onsite_unitary_old(Nmax,dt,m2,k,lam):

    ops = local_boson_ops(Nmax)

    x = ops["x"]
    p = ops["p"]

    h = 0.5*p@p + 0.5*(m2+2*k)*x@x + lam*x@x@x@x

    return expm(-1j*dt*h)


def bond_unitary_old(Nmax,dt,k):

    ops = local_boson_ops(Nmax)

    x = ops["x"]

    d = Nmax+1

    h = -k*np.kron(x,x)

    return expm(-1j*dt*h)

from scipy.linalg import expm
import numpy as np

def coupling_bond_unitary(Nmax, dt, g, m_squared, k):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]

    omega0 = np.sqrt(m_squared + 2 * k)

    H_pair = g * (
        (omega0 / 2.0) * np.kron(x, x)
        + (1.0 / (2.0 * omega0)) * np.kron(p, p)
    )

    Uc = expm(-1j * dt * H_pair)
    return Uc

# ============================================================
# Apply gates
# ============================================================

def apply_one_site_old(psi,i,U):

    op = npc.Array.from_ndarray_trivial(U,labels=['p','p*'])

    psi.apply_local_op(i,op,unitary=True)

    return psi


def apply_two_site_adjacent(psi,i,U):

    d = int(np.sqrt(U.shape[0]))

    op = npc.Array.from_ndarray_trivial(
        U.reshape(d,d,d,d),
        labels=['p0','p1','p0*','p1*']
    )

    psi.apply_local_op(i,op,unitary=True)

    return psi



def SWAP_gate(Nmax):

    d = Nmax + 1

    U = np.zeros((d*d, d*d), dtype=complex)

    for i in range(d):
        for j in range(d):

            in_index  = i*d + j
            out_index = j*d + i

            U[out_index, in_index] = 1.0

    return U 


from scipy.linalg import expm

def gaussian_mode(Nmax, r=0.5, theta=0.0):

    ops = local_boson_ops(Nmax)

    b = ops["b"]
    bd = ops["bd"]

    d = Nmax+1

    vacuum = np.zeros(d)
    vacuum[0] = 1

    # squeezing
    Hs = 0.5*(b@b - bd@bd)
    S = expm(r*Hs)

    # rotation
    n = bd@b
    R = expm(-1j*theta*n)

    U = R @ S

    phi = U @ vacuum

    return phi / np.linalg.norm(phi)


def covariance_matrix(psi_tensor, Nsites, Nmax):

    ops = local_boson_ops(Nmax)

    x = ops["x"]
    p = ops["p"]

    d = Nmax+1
    L = Nsites

    psi = psi_tensor.reshape(d**L)

    # build operator list
    R_ops = []

    for i in range(L):
        R_ops.append(embed_one_site(x,i,L,d))
        R_ops.append(embed_one_site(p,i,L,d))

    n = 2*L

    Gamma = np.zeros((n,n))

    # compute means
    means = np.zeros(n)

    for i in range(n):
        means[i] = np.real(np.vdot(psi, R_ops[i] @ psi))

    # compute covariance
    for i in range(n):
        for j in range(n):

            A = R_ops[i] @ R_ops[j]
            B = R_ops[j] @ R_ops[i]

            val = 0.5*np.vdot(psi,(A+B)@psi)

            Gamma[i,j] = np.real(val) - means[i]*means[j]

    return Gamma

import numpy as np

def one_site_rho_matrix(psi, i):
    rho = psi.get_rho_segment([i]).to_ndarray()
    return rho.reshape(rho.shape[0], rho.shape[0])

def two_site_rho_matrix(psi, i, j):
    rho = psi.get_rho_segment([i, j]).to_ndarray()

    # reorder to matrix form
    # expected raw shape is (d,d,d,d)
    d = rho.shape[0]
    rho = np.transpose(rho, (0, 1, 2, 3)).reshape(d*d, d*d)
    return rho

def covariance_matrix_from_mps(psi, Nsites, Nmax):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]
    d = Nmax + 1

    local_ops = [x, p]
    nquad = 2 * Nsites
    Gamma = np.zeros((nquad, nquad), dtype=float)
    means = np.zeros(nquad, dtype=float)

    def op_for_index(a):
        site = a % Nsites
        kind = 0 if a < Nsites else 1   # 0 -> x, 1 -> p
        return site, local_ops[kind]

    # one-point functions
    for a in range(nquad):
        i, Oi = op_for_index(a)
        rho_i = one_site_rho_matrix(psi, i)
        means[a] = np.real(np.trace(rho_i @ Oi))

    # covariance entries
    for a in range(nquad):
        i, Oi = op_for_index(a)
        for b in range(nquad):
            j, Oj = op_for_index(b)

            if i == j:
                rho_i = one_site_rho_matrix(psi, i)
                sym_op = 0.5 * (Oi @ Oj + Oj @ Oi)
                val = np.real(np.trace(rho_i @ sym_op))
            else:
                rho_ij = two_site_rho_matrix(psi, i, j)
                Oij = np.kron(Oi, Oj)
                val = np.real(np.trace(rho_ij @ Oij))

            Gamma[a, b] = val - means[a] * means[b]

    return 0.5 * (Gamma + Gamma.T)

def covariance_from_single_mode_state(phi, Nmax):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]

    phi = phi / np.linalg.norm(phi)

    mx = np.real(np.vdot(phi, x @ phi))
    mp = np.real(np.vdot(phi, p @ phi))

    xx = np.real(np.vdot(phi, x @ x @ phi))
    pp = np.real(np.vdot(phi, p @ p @ phi))
    xp = np.real(np.vdot(phi, 0.5 * (x @ p + p @ x) @ phi))

    V = np.array([
        [xx - mx*mx, xp - mx*mp],
        [xp - mx*mp, pp - mp*mp]
    ])
    return 0.5 * (V + V.T)

def apply_two_site_nonlocal(psi, i, j, U,Nmax):

    if j < i:
        i, j = j, i

    swap = SWAP_gate(Nmax)   # precompute if possible

    # move j next to i
    for k in range(j-1, i, -1):
        apply_two_site_adjacent(psi, k, swap)

    # apply desired gate
    apply_two_site_adjacent(psi, i, U)

    # swap back
    for k in range(i+1, j):
        apply_two_site_adjacent(psi, k, swap)

    return psi

def apply_wormhole_coupling(
        psi,
        N,
        insert_idx,
        t_couple,
        dt,
        Nmax,
        g,
        m_squared,
        k):

    steps_couple = int(t_couple/dt)

    Uc = coupling_bond_unitary(Nmax, dt, g, m_squared, k)

    for _ in range(steps_couple):

        for i in range(N):

            if i == insert_idx:
                continue

            left = i
            right = N + i

            apply_two_site_nonlocal(psi, left, right, Uc, Nmax)

    psi.canonical_form()

    return psi


# ============================================================
# Protocol
# ============================================================

def teleportation_protocol_old(s,theta,N,Nmax,insert_idx,psi_tensor):

    # insert gaussian mode

    phi = gaussian_mode(
        Nmax=Nmax,
        r=s,       # squeezing strength
        theta=theta    # phase rotation
    )

    psi_insert = insert_with_env(psi_tensor,insert_idx,phi)

    # convert to MPS

    psi = tensor_to_mps(psi_insert,Nmax)

    # forward evolve left

    U1 = onsite_unitary(Nmax,dt,m2,k,lam)
    U2 = bond_unitary(Nmax,dt,k)

    steps = int(t_scramble/dt)

    for _ in range(steps):

        for i in range(N):
            apply_one_site(psi,i,U1)

        for i in range(N):
            apply_two_site_adjacent(psi,i,U2)   # ring bond

    # coupling
    apply_wormhole_coupling(
        psi,
        N,
        insert_idx,
        t_couple,
        dt,
        Nmax,
        g,
        m2,
        k)

    # evolve right

    for _ in range(steps):

        for i in range(N,2*N):
            apply_one_site(psi,i,U1)
        for i in range(N,2*N):
            apply_two_site_adjacent(psi,i,U2)
 

    return psi

def onsite_unitary(Nmax, dt, m2, k, lam):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]

    H = 0.5*p@p + 0.5*(m2+2*k)*x@x

    if lam > 0:
        H += lam*x@x@x@x

    U = expm(-1j*dt*H)
    return U


def bond_unitary(Nmax, dt, k):
    ops = local_boson_ops(Nmax)
    x = ops["x"]

    H = -k*np.kron(x,x)

    U = expm(-1j*dt*H)
    return U

def apply_one_site(psi, i, U):

    d = U.shape[0]

    op = npc.Array.from_ndarray_trivial(
        U.reshape(d, d),
        labels=['p', 'p*'] 
    )

    psi.apply_local_op(i, op, unitary=True)

def apply_coupling_bond(psi, i, j, U, Nmax, chi_max=512, svd_cut=1e-10):

    if j < i:
        i,j = j,i

    # bring j next to i
    for k in range(j-1, i, -1):
        apply_two_site_adjacent(psi, k, SWAP_gate(Nmax))

    apply_two_site_adjacent(psi, i, U)

    # move back
    for k in range(i+1, j):
        apply_two_site_adjacent(psi, k, SWAP_gate(Nmax))



def apply_ring_bond(psi, i, j, U, N, Nmax, chi_max=512, svd_cut=1e-10):

    # ordinary adjacent bond on the line
    if abs(i - j) == 1:
        apply_two_site_adjacent(psi, min(i, j), U)
        return

    # wraparound ring bond
    if abs(i - j) == N - 1:
        if i < j:
            left = i
            right = j
        else:
            left = j
            right = i

        # bring right next to left
        for k in range(right - 1, left, -1):
            apply_two_site_adjacent(psi, k, SWAP_gate(Nmax))

        apply_two_site_adjacent(psi, left, U)

        # move back
        for k in range(left + 1, right):
            apply_two_site_adjacent(psi, k, SWAP_gate(Nmax))

        return

    raise ValueError(f"Sites {i},{j} are not nearest neighbors on a ring of length {N}.")



def tebd_step_ring_old(psi, start, end, U1, U2,N,Nmax):

    # onsite step
    for i in range(start,end):
        apply_one_site(psi,i,U1)

    # even bonds
    for i in range(start,end-1,2):
        apply_two_site_adjacent(psi,i,U2)

    # odd bonds
    for i in range(start+1,end-1,2):
        apply_two_site_adjacent(psi,i,U2)

    # ring bond
    apply_ring_bond(psi,end-1,start,U2,N,Nmax)

def tebd_step_ring(psi, start, end, U1, U2, N, Nmax):
    # 1. Pre-calculate half-step onsite unitary
    # Note: U1 is expm(-1j * dt * H_onsite). 
    # For a half step, we need expm(-1j * (dt/2) * H_onsite)
    # You should pass dt/2 to your onsite_unitary function to get U1_half
    
    dt = 0.05 # From your global vars
    U1_half = onsite_unitary(Nmax, dt/2, m2, k, lam)

    # --- A. Onsite Half-Step ---
    for i in range(start, end):
        apply_one_site(psi, i, U1_half)

    # --- B. Bond Full-Steps (Even then Odd) ---
    # Even bonds: (0,1), (2,3)
    for i in range(start, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2)

    # Odd bonds: (1,2), (3,4)
    for i in range(start + 1, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2)

    # --- C. Ring Wrap-around Bond ---
    apply_ring_bond(psi, end - 1, start, U2, N, Nmax)

    # --- D. Onsite Half-Step ---
    for i in range(start, end):
        apply_one_site(psi, i, U1_half)

def tebd_step_ring_2nd_order(psi, start, end, U1_half, U2_half, U2_full, N, Nmax):
    # onsite half
    for i in range(start, end):
        apply_one_site(psi, i, U1_half)

    # even half
    for i in range(start, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2_half)

    # odd half
    for i in range(start + 1, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2_half)

    # wrap full
    apply_ring_bond(psi, end - 1, start, U2_full, N, Nmax)

    # odd half
    for i in range(start + 1, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2_half)

    # even half
    for i in range(start, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2_half)

    # onsite half
    for i in range(start, end):
        apply_one_site(psi, i, U1_half)

def wormhole_unitary(Nmax, dt, g, omega0):

    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]

    H = g*(.5*omega0*np.kron(x,x)
           + 0.5/omega0*np.kron(p,p))

    return expm(-1j*dt*H)

def tebd_step_coupled_old(psi, N, insert_idx, U1, U2, Uc,Nmax):

    # onsite evolution (both sides)
    for i in range(2*N):
        apply_one_site(psi, i, U1)

    # left bonds
    for i in range(0, N-1):
        apply_two_site_adjacent(psi, i, U2)

    apply_ring_bond(psi, N-1, 0, U2,N,Nmax)

    # right bonds
    for i in range(N, 2*N-1):
        apply_two_site_adjacent(psi, i, U2)

    apply_ring_bond(psi, 2*N-1, N, U2,N,Nmax)

    # wormhole coupling
    for i in range(N):
        if i == insert_idx:
            continue

        L = i
        R = N + i

        apply_coupling_bond(psi, L, R, Uc,Nmax)

def tebd_step_coupled(psi, N, insert_idx, U1, U2, Uc, Nmax):
    # Left ring
    tebd_step_ring(psi, start=0, end=N, U1=U1, U2=U2, N=N, Nmax=Nmax)
    #tebd_step_ring_2nd_order(psi, 0, N, U1_half, U2_half, U2_full, N, Nmax)
    # Right ring
    tebd_step_ring(psi, start=N, end=2 * N, U1=U1, U2=U2, N=N, Nmax=Nmax)
    #tebd_step_ring_2nd_order(psi, N, 2*N, U1_half, U2_half, U2_full, N, Nmax)

    # Left-right couplings
    for i in range(N):
        if i == insert_idx:
            continue
        apply_coupling_bond(psi, i, N + i, Uc, Nmax)

def evolve_with_coupling(
        psi,
        N,
        insert_idx,
        t_couple,
        dt,
        Nmax,
        g,
        m2,
        k,
        lam):

    omega0 = np.sqrt(m2 + 2*k)

    U1 = onsite_unitary(Nmax, dt, m2, k, lam)
    U2 = bond_unitary(Nmax, dt, k)

    #U1_half = onsite_unitary(Nmax, dt/2, m2, k, lam)
    #U2_half = bond_unitary(Nmax, dt/2, k)
    #U2_full = bond_unitary(Nmax, dt, k)

    Uc = wormhole_unitary(Nmax, dt, g, omega0)

    steps = int(t_couple/dt)

    for _ in range(steps):

        tebd_step_coupled(
            psi,
            N,
            insert_idx,
            U1,
            U2,
            Uc,
            Nmax
        )


def local_number_distribution_from_mps(psi, i):
    rho_i = one_site_rho_matrix(psi, i)
    probs = np.real(np.diag(rho_i))
    probs = probs / probs.sum()
    return probs


def teleportation_protocol(s, theta, N, Nmax, insert_idx, psi_tensor):

    beta = 1.0

    m2 = 13
    k = 6
    lam = 0.0

    g = 1

    dt = 0.05
    t_scramble = 2
    t_couple = 3

    # insert gaussian mode
    phi = gaussian_mode(Nmax, r=s, theta=theta)

    psi_insert = insert_with_env(psi_tensor, insert_idx, phi)
    #insert_state_replace(psi_tensor, insert_idx, phi)
    # convert to MPS
    psi = tensor_to_mps(psi_insert, Nmax)

    # TEBD operators
    U1 = onsite_unitary(Nmax, dt, m2, k, lam)
    U2 = bond_unitary(Nmax, dt, k)

    #U1_half = onsite_unitary(Nmax, dt/2, m2, k, lam)
    #U2_half = bond_unitary(Nmax, dt/2, k)
    #U2_full = bond_unitary(Nmax, dt, k)

    steps = int(t_scramble/dt)
    
    # forward evolve left
    for _ in range(steps):
        tebd_step_ring(psi, 0, N, U1, U2,N,Nmax)
        #tebd_step_ring_2nd_order(psi, 0, N, U1_half, U2_half, U2_full, N, Nmax)
    # wormhole coupling
    evolve_with_coupling(
    psi,
    N,
    insert_idx,
    t_couple,
    dt,
    Nmax,
    g,
    m2,
    k,
    lam)

    # evolve right
    for _ in range(steps):
        tebd_step_ring(psi, N, 2*N, U1, U2,N,Nmax)
        #tebd_step_ring_2nd_order(psi, N, 2*N, U1_half, U2_half, U2_full, N, Nmax)
    return psi


def fidelity_vs_site(
    insert_idx,
    input_ensemble,  # list of (s, theta) you use for fitting
    N,
    Nmax,
    psi_tensor):


    Vins = []

    Vouts = [[] for _ in range(N)]


    for s, theta in input_ensemble:
        # Run your usual protocol (NO observer) to get global Gamma_final
        psi_out = teleportation_protocol(s,theta,N,Nmax,insert_idx,psi_tensor)
        Vout = covariance_matrix_from_mps(psi_out,2*N+1,Nmax)   
        for i in range(2*N+1):
            probs = local_number_distribution_from_mps(psi_out, i)
            print(f"site {i}: <n> = {sum(n*p for n,p in enumerate(probs)):.4f}, "f"P(Nmax) = {probs[-1]:.4e}, P(Nmax-1) = {probs[-2]:.4e}")

        Vins.append(covariance_from_single_mode_state(gaussian_mode(Nmax=Nmax,r=s,theta=theta),Nmax))
        
        for i in range(N):
            Vouts[i].append(extract_subsystem_covariance(Vout,[i+N]))
            print(i,Vouts[i][-1])
        
        #Vouts3.append(extract_subsystem_covariance(covariance_matrix_from_mps(psi_out,N,Nmax),[3]))
        #Vouts4.append(extract_subsystem_covariance(covariance_matrix_from_mps(psi_out,N,Nmax),[4]))
        #Vouts5.append(extract_subsystem_covariance(covariance_matrix_from_mps(psi_out,N,Nmax),[5]))


        # --- 5) Fit a single-mode Gaussian channel for this decoded mode ---


    #X3, Y3 = fit_gaussian_channel(Vins, Vouts3)
    #X4, Y4 = fit_gaussian_channel(Vins, Vouts4)
    #X5, Y5 = fit_gaussian_channel(Vins, Vouts4)

    fid_symp = []
    fid_flip = []
    for i in range(N):
        print(f"Vouts[{i}]={Vouts[0]}")

    for i in range(N):
        X, Y = fit_gaussian_channel(Vins, Vouts[i])
        rot1,loss,squeeze,rot2 = decompose_X(X)
        print(i+N)
        print(f"rot1={rot1}")
        print(f"rot2={rot2}")
        print(f"loss={loss}")
        print(f"squeeze={squeeze}")
        print(f"Y={Y}")

        S_dec_symp = decoder_from_X_symplectic(X)  
        S_dec_flip = decoder_from_X_flip(X) 

        Fs = entanglement_fidelity_gaussian(X, Y, S_dec_symp, subtract_Y=False, r=1.0)
        Ff = entanglement_fidelity_gaussian(X, Y, S_dec_flip, subtract_Y=False, r=1.0)

        fid_symp.append(Fs)
        fid_flip.append(Ff)

        print(f"fid_flip_3={Ff}")
        print(f"fid_symp_3={Fs}")

    """
    rot1_3, loss_3, squeeze_3, rot2_3 = decompose_X(X3)
    print(3)
    print(f"rot1={rot1_3}")
    print(f"rot2={rot2_3}")
    print(f"loss={loss_3}")
    print(f"squeeze={squeeze_3}")
    print(f"Y={Y3}")

    rot1_4, loss_4, squeeze_4, rot2_4 = decompose_X(X4)
    print(4)
    print(f"rot1={rot1_4}")
    print(f"rot2={rot2_4}")
    print(f"loss={loss_4}")
    print(f"squeeze={squeeze_4}")
    print(f"Y={Y4}")

    rot1_5, loss_5, squeeze_5, rot2_5 = decompose_X(X5)
    print(5)
    print(f"rot1={rot1_5}")
    print(f"rot2={rot2_5}")
    print(f"loss={loss_5}")
    print(f"squeeze={squeeze_5}")
    print(f"Y={Y5}")


  
    S_dec_symp_3 = decoder_from_X_symplectic(X3)  # your preferred
    S_dec_flip_3 = decoder_from_X_flip(X3)  # your preferred

    Fs3 = entanglement_fidelity_gaussian(X3, Y3, S_dec_symp_3, subtract_Y=False, r=1.0)
    Ff3 = entanglement_fidelity_gaussian(X3, Y3, S_dec_flip_3, subtract_Y=False, r=1.0)

    print(f"fid_flip_3={Ff3}")
    print(f"fid_symp_3={Fs3}")

    S_dec_symp_4 = decoder_from_X_symplectic(X4)  # your preferred
    S_dec_flip_4 = decoder_from_X_flip(X4)  # your preferred

    Fs4 = entanglement_fidelity_gaussian(X4, Y4, S_dec_symp_4, subtract_Y=False, r=1.0)
    Ff4 = entanglement_fidelity_gaussian(X4, Y4, S_dec_flip_4, subtract_Y=False, r=1.0)

    print(f"fid_flip_3={Ff4}")
    print(f"fid_symp_3={Fs4}")


    S_dec_symp_5 = decoder_from_X_symplectic(X5)  # your preferred
    S_dec_flip_5 = decoder_from_X_flip(X5)  # your preferred

    Fs5 = entanglement_fidelity_gaussian(X5, Y5, S_dec_symp_5, subtract_Y=False, r=1.0)
    Ff5 = entanglement_fidelity_gaussian(X5, Y5, S_dec_flip_5, subtract_Y=False, r=1.0)

    print(f"fid_flip_3={Ff5}")
    print(f"fid_symp_3={Fs5}")

    """

    return fid_symp,fid_flip


N=3
Nmax=8
beta = 1.0

m2 = 13
k = 6
lam = 0.0

g = 1

dt = 0.05
t_scramble = 2
t_couple = 3

# build dense ring Hamiltonian
H_quad = build_ring_H(N, Nmax, m2, k, lam)

# build TFD
psi_tensor = build_tfd_tensor(H_quad, N, Nmax, beta)

# backward modular evolution
U_back = expm(1j*t_scramble*H_quad)

psi_left = psi_tensor.reshape((Nmax+1)**N,-1)
psi_left = U_back @ psi_left
psi_tensor = psi_left.reshape([Nmax+1]*(2*N))



Ss = np.linspace(-.6, .6, 4)
Thetas = np.linspace(0, 2 * np.pi, 3, endpoint=False)
input_ensemble = [(s, th) for s in Ss for th in Thetas]  # 120 points, deterministic

sites = np.arange(N, 2 * N)
#site_fidelities_symp = []
#site_fidelities_flip = []




# for f in range(len(sites)):
    #Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    #plt.plot(block_sizes,Fs,label=sites[f])
Fs,Ff= fidelity_vs_site(
    insert_idx=1,
    input_ensemble=input_ensemble,  # list of (s, theta) you use for fitting
    N=N,
    Nmax=Nmax,
    psi_tensor=psi_tensor
    )

print("Processing complete. Check LOG.log for details.")

plt.plot(sites,Fs,label="symplectic")
plt.plot(sites,Ff,label="allow flip")
plt.xlabel("site")
plt.ylabel("fidelity")
plt.legend()
plt.savefig("plots/site_vs_fidelity.pdf")
# plt.show()

"""
plt.xlabel("decoder block size")
plt.ylabel("fidelity")
plt.legend()
plt.show()
"""





"""
times_couple = np.linspace(1, 3, 5)
time_fidelities_symp = []
time_fidelities_flip = []

for t in range(len(times_couple)):
    # Fs = fidelity_vs_block_size(block_sizes, obs_idx, teleported_idx, bdy_len, input_ensemble,H_coupling_OG,N=N,center_idx=sites[f]-N,wormhole=False)
    # plt.plot(block_sizes,Fs,label=sites[f])
    Fs,Ff= fidelity_vs_block_size(
    insert_idx,
    input_ensemble,
    t_scramble,
    times_couple[t],
    dt,
    C_back,
    H_coupling,
    H_scramble_side,
    N_modes,
    N_cutoff,
    x_full_env=x_full_env,
    p_full_env=p_full_env,
    center_idx=3)
    time_fidelities_symp.append(Fs)
    time_fidelities_flip.append(Ff)

plt.plot(times_couple, time_fidelities_symp, label="symplectic")
plt.plot(times_couple, time_fidelities_flip, label="allow flip")
plt.xlabel("times")
plt.ylabel("fidelity")
plt.legend()
#plt.show()
plt.savefig("plots/time_vs_fidelity.pdf")
"""

log_print("done")
