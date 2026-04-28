# -*- coding: utf-8 -*-
"""
Wormhole teleportation fidelity via Gaussian TN.

MPS sites use interleaved order ``L0,R0,L1,R1,...`` so each ``L_i–R_i`` pair is adjacent
(wormhole bonds are local). Run with the ``gaussian`` conda env (TenPy + QuTiP), e.g.:

    conda activate gaussian
    python src/gaussian_tn_fidelity_speedy.py --quick
    python src/gaussian_tn_fidelity_speedy.py --N 4 --Nmax 10
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import argparse
import logging
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.linalg import block_diag, det, eigh, expm, logm, schur, sqrtm
from tenpy.linalg import np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.site import BosonSite
from thewalrus.symplectic import xpxp_to_xxpp
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("LOG.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    # System
    N: int = 3          # sites per ring
    Nmax: int = 11       # bosonic truncation (Fock cutoff)
    beta: float = 1.0   # inverse temperature
    # Hamiltonian
    m2: float = 13.0    # mass squared
    k: float = 5.0      # hopping / spring constant
    lam: float = 0.0    # quartic coupling
    g: float = 1.0      # wormhole coupling strength
    # Time evolution
    dt: float = 0.05    # Trotter step
    t_scramble: float = 2.0
    t_couple: float = 3.0
    # Ensemble for channel fitting
    n_squeeze: int = 4          # number of squeezing values
    n_theta: int = 3            # number of phase angles
    squeeze_range: float = 0.6  # squeezing sampled in [-r, r]
    # Initial state: build quadratic TFD via ⊗_k thermo doubles in normal modes + passive
    # rotation (no dense d^N×d^N H). Falls back to ``build_tfd_tensor`` if disabled or too large.
    use_normal_mode_tfd: bool = True
    # Use symmetric Trotter on each ring during scramble/unscramble (interleaved layout).
    second_order_ring_tebd: bool = True


# Maximal single-particle Fock dimension (Nmax+1)^N allowed for normal-mode passive rotation.
_PASSIVE_UNITARY_MAX_DIM = 8000


# Preset configs for quick local testing vs full server runs
QUICK = Config(N=3, Nmax=4, t_scramble=0.5, t_couple=0.5, n_squeeze=2, n_theta=2)
FULL  = Config()
CUTE_CONFFIG = Config(N=3, Nmax=8, t_scramble=2, t_couple=3, n_squeeze=4, n_theta=3)



# ── Gaussian state utilities ──────────────────────────────────────────────────

def symplectic_form(n):
    return np.block([
        [np.zeros((n, n), dtype=np.float64),  np.eye(n, dtype=np.float64)],
        [-np.eye(n, dtype=np.float64),         np.zeros((n, n), dtype=np.float64)],
    ])


def symplectic_eigenvalues(Gamma):
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * Omega @ Gamma)
    return np.sort(np.abs(eigvals))[::2]


def extract_subsystem_covariance(Gamma, indices):
    indices = np.array(indices)
    x_idx = indices
    p_idx = indices + Gamma.shape[0] // 2
    full_idx = np.concatenate([x_idx, p_idx])
    return Gamma[np.ix_(full_idx, full_idx)]


def von_neumann_entropy(Gamma):
    nu = symplectic_eigenvalues(Gamma)
    nu = np.clip(nu, 0.500001, None)
    return float(np.sum((nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5)))


def sym(A):
    return 0.5 * (A + A.T)


def build_thermal_state_from_modular_hamiltonian(K, tol=1e-8):
    K = sym(K)
    eigvals, O = eigh(K)
    nu = np.where(
        np.abs(eigvals) < tol,
        0.5,
        0.5 / np.tanh(0.5 * eigvals),
    )
    Gamma = O @ np.diag(nu) @ O.T
    return sym(Gamma), nu, eigvals


def build_quadratic_thermal_covariance(m2, k, N, beta):
    V = np.zeros((N, N))
    for i in range(N):
        V[i, i] = m2 + 2 * k
        V[i, (i + 1) % N] = -k
        V[i, (i - 1) % N] = -k
    omega2, O = np.linalg.eigh(V)
    omega = np.sqrt(omega2)
    coth = lambda x: 1 / np.tanh(x)
    var_q = (1 / (2 * omega)) * coth(beta * omega / 2)
    var_p = (omega / 2) * coth(beta * omega / 2)
    Gamma_xx = O @ np.diag(var_q) @ O.T
    Gamma_pp = O @ np.diag(var_p) @ O.T
    Gamma = np.block([[Gamma_xx, np.zeros((N, N))], [np.zeros((N, N)), Gamma_pp]])
    return sym(Gamma)


def symplectic_direct_sum(S1, S2):
    n = S1.shape[0]
    h = n // 2
    A = block_diag(S1[:h, :h], S2[:h, :h])
    B = block_diag(S1[:h, h:], S2[:h, h:])
    C = block_diag(S1[h:, :h], S2[h:, :h])
    D = block_diag(S1[h:, h:], S2[h:, h:])
    return np.block([[A, B], [C, D]])


def williamson_strawberry(V):
    tol = 1e-11
    (n, m) = V.shape
    if n != m:
        raise ValueError("Matrix not square")
    if n % 2 != 0:
        raise ValueError("Matrix must have even dimension")
    if np.linalg.norm(V - V.T) >= 1e-5:
        raise ValueError("Matrix not symmetric")
    n = n // 2
    omega = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    if any(v <= 0 for v in np.linalg.eigvalsh(V)):
        raise ValueError("Matrix not positive definite")
    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = [I if s1[2*i, 2*i+1] > 0 else X for i in range(n)]
    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    dd = xpxp_to_xxpp(s1t)
    perm_indices = xpxp_to_xxpp(np.arange(2 * n))
    Ktt = Kt[:, perm_indices]
    Db = np.diag([1 / dd[i, i + n] for i in range(n)] + [1 / dd[i, i + n] for i in range(n)])
    S = Mm12 @ Ktt @ sqrtm(Db)
    eigvals, _ = eigh(sqrtm(V) @ omega @ sqrtm(V))
    v = np.sort(np.abs(eigvals.real))[::2]
    return np.linalg.inv(S).T, Db, v


def gaussian_purification(V):
    S_xxpp, Db_xxpp, nus = williamson_strawberry(V)
    alphas = np.sqrt(nus**2 - 0.25)
    C_xxpp = np.block([
        [np.diag(alphas),            np.zeros((len(alphas), len(alphas)))],
        [np.zeros((len(alphas), len(alphas))), np.diag(-alphas)],
    ])
    V_pure_will_xxpp = np.block([[Db_xxpp, C_xxpp], [C_xxpp, Db_xxpp]])
    S_total = symplectic_direct_sum(S_xxpp.T, S_xxpp.T)
    return S_total @ V_pure_will_xxpp @ S_total.T


def fidelity_stable(V1, V2):
    V1, V2 = sym(V1), sym(V2)
    n = V1.shape[0] // 2
    omega = symplectic_form(n)
    Vsum = V1 + V2
    V_aux = omega.T @ np.linalg.inv(Vsum) @ (0.25 * omega + V2 @ omega @ V1)
    I = np.eye(2 * n)
    A = V_aux @ omega
    Ainv2 = np.linalg.solve(A, np.linalg.solve(A, I))
    inside = I + 0.25 * Ainv2
    F_tot4 = np.linalg.det(2 * (sqrtm(inside) + I) @ V_aux)
    F0 = np.real_if_close(F_tot4) ** 0.25 / (np.linalg.det(Vsum) ** 0.25)
    return float(np.real(F0))


def reorder_to_block_form(Gamma):
    perm = [0, 2, 1, 3]
    return Gamma[np.ix_(perm, perm)]


def tmsv_cov(r):
    ch, sh = np.cosh(2 * r), np.sinh(2 * r)
    Z = np.diag([1, -1])
    cov = 0.5 * np.block([[ch * np.eye(2), sh * Z], [sh * Z, ch * np.eye(2)]])
    return reorder_to_block_form(cov)


def apply_channel_to_second_mode_xxpp(V_RB_xxpp, X, Y):
    V = sym(V_RB_xxpp)
    X, Y = np.asarray(X, float), sym(np.asarray(Y, float))
    idx_R, idx_B = [0, 2], [1, 3]
    A = V[np.ix_(idx_R, idx_R)]
    B = V[np.ix_(idx_B, idx_B)]
    C = V[np.ix_(idx_R, idx_B)]
    V_out = V.copy()
    V_out[np.ix_(idx_R, idx_R)] = A
    V_out[np.ix_(idx_R, idx_B)] = C @ X.T
    V_out[np.ix_(idx_B, idx_R)] = X @ C.T
    V_out[np.ix_(idx_B, idx_B)] = X @ B @ X.T + Y
    return sym(V_out)


def decode_on_B_xxpp(V_RB_xxpp, S_dec, Y, subtract_Y):
    V = sym(V_RB_xxpp)
    idx_R, idx_B = [0, 2], [1, 3]
    B = V[np.ix_(idx_B, idx_B)]
    C = V[np.ix_(idx_R, idx_B)]
    if subtract_Y:
        B = B - Y
    V_out = V.copy()
    V_out[np.ix_(idx_B, idx_B)] = S_dec @ B @ S_dec.T
    V_out[np.ix_(idx_R, idx_B)] = C @ S_dec.T
    V_out[np.ix_(idx_B, idx_R)] = S_dec @ C.T
    return sym(V_out)


def entanglement_fidelity_gaussian(X, Y, S, subtract_Y, r=1.0):
    V0 = tmsv_cov(r)
    V1 = apply_channel_to_second_mode_xxpp(V0, X, Y)
    V1_dec = decode_on_B_xxpp(V1, np.linalg.inv(S), Y, subtract_Y)
    return fidelity_stable(V0, V1_dec)


def pack_params(X, Y):
    return np.array([X[0,0], X[0,1], X[1,0], X[1,1], Y[0,0], Y[0,1], Y[1,1]], dtype=float)


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
        r.extend([E[0, 0], E[0, 1], E[1, 1]])
    return np.array(r, dtype=float)


def fit_gaussian_channel(Vins, Vouts, X0=None, Y0=None, lam=1e-3, iters=200):
    Vins  = [sym(V) for V in Vins]
    Vouts = [sym(V) for V in Vouts]
    if X0 is None:
        X0 = np.eye(2)
    if Y0 is None:
        Y0 = sym(np.mean([Vout - X0 @ Vin @ X0.T for Vin, Vout in zip(Vins, Vouts)], axis=0))
    p = pack_params(X0, Y0)
    for _ in range(iters):
        r = residuals(p, Vins, Vouts)
        cost = r @ r
        J = np.zeros((len(r), len(p)))
        eps = 1e-6
        for param_idx in range(len(p)):
            dp = np.zeros_like(p)
            dp[param_idx] = eps
            J[:, param_idx] = (residuals(p + dp, Vins, Vouts) - r) / eps
        A = J.T @ J + lam * np.eye(len(p))
        delta = np.linalg.solve(A, J.T @ r)
        p_new = p - delta
        if (residuals(p_new, Vins, Vouts) @ residuals(p_new, Vins, Vouts)) < cost:
            p = p_new
            lam *= 0.7
        else:
            lam *= 2.0
        if np.linalg.norm(delta) < 1e-10:
            break
    X, Y = unpack_params(p)
    return X, sym(Y)


def decompose_X(X):
    U, s, Vt = np.linalg.svd(X)
    O1 = U.copy()
    if det(U) < 0:
        O1[:, 1] *= -1
    s1, s2 = s
    r = 0.5 * np.log(s2 / s1)
    squeeze = np.diag([np.exp(-r), np.exp(r)])
    eta = np.sqrt(s1 * s2)
    loss = np.diag((eta, -eta)) if det(U) < 0 else np.diag((eta, eta))
    return O1, loss, squeeze, Vt


def decoder_from_X_symplectic(X):
    U, s, Vt = np.linalg.svd(X)
    O1 = U.copy()
    if det(U) < 0:
        O1[:, 1] *= -1
    s1, s2 = s
    r = 0.5 * np.log(s2 / s1)
    return O1 @ np.diag([np.exp(-r), np.exp(r)]) @ Vt


def decoder_from_X_flip(X):
    U, s, Vt = np.linalg.svd(X)
    s1, s2 = s
    r = 0.5 * np.log(s2 / s1)
    return U @ np.diag([np.exp(-r), np.exp(r)]) @ Vt


def wigner_overlap_with_gaussian_target(q, p, V_target, d_target=None):
    q, p = np.asarray(q).reshape(-1), np.asarray(p).reshape(-1)
    z = np.stack([q, p], axis=1)
    d = np.zeros(2) if d_target is None else np.asarray(d_target).reshape(2)
    V = sym(V_target)
    Vinv = np.linalg.inv(V)
    detV = np.linalg.det(V)
    dz = z - d[None, :]
    quad = np.einsum("bi,ij,bj->b", dz, Vinv, dz)
    return float(np.mean((1.0 / np.sqrt(detV)) * np.exp(-0.5 * quad)))


def plot_wigner_ellipse(Gamma_mode, ax, label="", color="blue"):
    W = Gamma_mode[:2, :2].real
    vals, vecs = eigh(W)
    width, height = 2 * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                      edgecolor=color, fc="None", lw=2, label=label)
    ax.add_patch(ellipse)


# ── Bosonic operators & local Hilbert space ───────────────────────────────────

def local_boson_ops(Nmax):
    d = Nmax + 1
    b = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        b[n - 1, n] = np.sqrt(n)
    bd = b.conj().T
    x = (b + bd) / np.sqrt(2)
    p = (b - bd) / (1j * np.sqrt(2))
    return dict(x=x, p=p, b=b, bd=bd, I=np.eye(d))


def kron_all(ops):
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out


def embed_one_site(op, site, L, d):
    ops = [np.eye(d) for _ in range(L)]
    ops[site] = op
    return kron_all(ops)


def embed_two_site(op, i, j, L, d):
    if i == j:
        raise ValueError("i and j must be distinct")
    if i > j:
        op4 = op.reshape(d, d, d, d)
        op4 = np.transpose(op4, (1, 0, 3, 2))
        op = op4.reshape(d * d, d * d)
        i, j = j, i
    op4 = op.reshape(d, d, d, d)
    I = np.eye(d, dtype=complex)
    O = np.zeros((d**L, d**L), dtype=complex)
    for oi in range(d):
        for oj in range(d):
            for ii in range(d):
                for ij in range(d):
                    coeff = op4[oi, oj, ii, ij]
                    if abs(coeff) < 1e-15:
                        continue
                    factors = [I.copy() for _ in range(L)]
                    Ei = np.zeros((d, d), dtype=complex)
                    Ej = np.zeros((d, d), dtype=complex)
                    Ei[oi, ii] = 1.0
                    Ej[oj, ij] = 1.0
                    factors[i] = Ei
                    factors[j] = Ej
                    O += coeff * kron_all(factors)
    return O


# ── Ring Hamiltonian & TFD ────────────────────────────────────────────────────

def build_ring_H(Nsites, Nmax, m2, k, lam):
    ops = local_boson_ops(Nmax)
    x, p = ops["x"], ops["p"]
    d = Nmax + 1
    L = Nsites
    H = np.zeros((d**L, d**L), dtype=complex)
    for i in range(L):
        H += 0.5 * embed_one_site(p @ p, i, L, d)
        H += 0.5 * m2 * embed_one_site(x @ x, i, L, d)
        if lam > 0:
            H += lam * embed_one_site(x @ x @ x @ x, i, L, d)
    xx = np.kron(x, x)
    for i in range(L):
        j = (i + 1) % L
        H += 0.5 * k * embed_one_site(x @ x, i, L, d)
        H += 0.5 * k * embed_one_site(x @ x, j, L, d)
        H += -k * embed_two_site(xx, i, j, L, d)
    return sym(np.real(H))


def build_tfd_tensor(H_side, Nsites, Nmax, beta):
    d = Nmax + 1
    evals, evecs = eigh(H_side)
    evals -= evals.min()
    w = np.exp(-beta * evals)
    Z = np.sum(w)
    C = evecs @ np.diag(np.sqrt(w / Z)) @ evecs.conj().T
    return C.reshape([d] * Nsites + [d] * Nsites)


def ring_coupling_matrix(N, m2, k):
    """Neighbor hopping matrix V for H_quad = sum_ij V_ij x_i x_j (+ momenta)."""
    V = np.zeros((N, N), dtype=float)
    for i in range(N):
        V[i, i] = m2 + 2 * k
        V[i, (i + 1) % N] = -k
        V[i, (i - 1) % N] = -k
    return sym(V)


def thermo_field_double_pair_matrix(Nmax, omega, beta):
    """Truncated |TFD⟩ diagonal on |n,n⟩ with P(n) ∝ e^{-βω n} / Z on each mode."""
    d = Nmax + 1
    beta_w = beta * omega
    tail = np.exp(-beta_w * np.arange(d))
    Z = tail.sum()
    coeffs = np.sqrt(tail / Z).astype(complex)
    psi = np.zeros((d, d), dtype=complex)
    np.fill_diagonal(psi, coeffs)
    return psi


def passive_fock_unitary_first_quantization(O: np.ndarray, Nmax: int) -> np.ndarray:
    """Second-quantized passive (particle-number conserving) unitary from orthogonal O.

    Uses U_Fock = exp(K) with K_{ij} = (log O)_{ij} antisymmetrized (real orthogonal O).
    Requires ``import qutip``.
    """
    import qutip as qt

    O = np.asarray(O, dtype=float)
    N = O.shape[0]
    d = Nmax + 1
    K = np.real(logm(O.astype(complex)))
    K = (K - K.T) / 2.0
    a_list = []
    for i in range(N):
        ops = [qt.qeye(d) for _ in range(N)]
        ops[i] = qt.destroy(d)
        a_list.append(qt.tensor(ops))
    Hgen = qt.Qobj(np.zeros([d**N] * 2, dtype=complex), dims=[[d] * N] * 2)
    for i in range(N):
        for j in range(N):
            Hgen += K[i, j] * (a_list[i].dag() * a_list[j])
    U = Hgen.expm()
    return np.asarray(U.full(), dtype=complex)


def tensor_product_pair_states(pair_tensors):
    """Outer product of disjoint two-site tensors with axes order L0,R0,L1,R1, ..."""
    out = pair_tensors[0]
    for k in range(1, len(pair_tensors)):
        out = np.tensordot(out, pair_tensors[k], axes=0)
    return out


def mode_interleaved_to_lr_blocks(T: np.ndarray, N: int) -> np.ndarray:
    """[m0_L,m0_R,m1_L,m1_R,...] → [m0_L,...,m_{N-1}_L, m0_R,...,m_{N-1}_R]."""
    assert T.ndim == 2 * N
    order = list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2))
    return np.transpose(T, order)


def permute_lr_block_to_interleaved(T: np.ndarray, N: int) -> np.ndarray:
    """[L0,...,L_{N-1}, R0,...,R_{N-1}] → [L0,R0,L1,R1,...]."""
    order = []
    for i in range(N):
        order.append(i)
        order.append(N + i)
    return np.transpose(T, order)


def apply_passive_LR_site_basis(M_mode: np.ndarray, O: np.ndarray, Nmax: int) -> np.ndarray:
    """Transform LR coefficient matrix from normal-mode to site tensor-product basis."""
    N = O.shape[0]
    d = Nmax + 1
    U = passive_fock_unitary_first_quantization(O, Nmax)
    return U @ M_mode @ U.T


def build_tfd_tensor_normal_modes(N: int, Nmax: int, m2: float, k: float, beta: float):
    """Quadratic-ring TFD as ⊗_k |TFD(ω_k)⟩ in normal modes, rotated to site basis.

    Avoids constructing the dense d^N × d^N Hamiltonian or its spectrum. Still forms a
    full rank-(d^{2N}) tensor for ``tensor_to_mps`` (same memory footprint as
    ``build_tfd_tensor``); CPU cost is dominated by two Fock-space rotations O(d^{3N}).

    Raises ``MemoryError`` if ``(Nmax+1)**N`` exceeds ``_PASSIVE_UNITARY_MAX_DIM``.
    """
    d = Nmax + 1
    dim_single = d**N
    if dim_single > _PASSIVE_UNITARY_MAX_DIM:
        raise MemoryError(
            f"normal-mode passive rotation dimension {dim_single} exceeds "
            f"_PASSIVE_UNITARY_MAX_DIM={_PASSIVE_UNITARY_MAX_DIM}; reduce N or Nmax "
            "or increase the limit after confirming available RAM."
        )

    V = ring_coupling_matrix(N, m2, k)
    omega2, O = np.linalg.eigh(V)
    omega = np.sqrt(np.maximum(omega2, 0.0))

    pairs = [thermo_field_double_pair_matrix(Nmax, omega[k], beta) for k in range(N)]
    T_mode_LR = tensor_product_pair_states(pairs)
    T_block = mode_interleaved_to_lr_blocks(T_mode_LR, N)
    M = T_block.reshape(dim_single, dim_single)
    M_site = apply_passive_LR_site_basis(M, O, Nmax)
    return M_site.reshape([d] * (2 * N))


# ── State preparation helpers ─────────────────────────────────────────────────

def gaussian_mode(Nmax, r=0.5, theta=0.0):
    ops = local_boson_ops(Nmax)
    b, bd = ops["b"], ops["bd"]
    d = Nmax + 1
    vacuum = np.zeros(d)
    vacuum[0] = 1.0
    S = expm(r * 0.5 * (b @ b - bd @ bd))
    R = expm(-1j * theta * (bd @ b))
    phi = R @ S @ vacuum
    return phi / np.linalg.norm(phi)


def insert_with_env(psi_tensor, insert_idx, phi):
    d = psi_tensor.shape[0]
    psi_env = np.zeros(psi_tensor.shape + (d,), dtype=complex)
    psi_env[..., 0] = psi_tensor
    psi_swapped = np.swapaxes(psi_env, insert_idx, psi_env.ndim - 1)
    U = np.eye(d)
    U[:, 0] = phi
    psi_out = np.tensordot(U, psi_swapped, axes=(1, insert_idx))
    return np.moveaxis(psi_out, 0, insert_idx)


def tensor_to_mps(psi_tensor, Nmax):
    nsites = psi_tensor.ndim
    sites = [BosonSite(Nmax=Nmax, conserve=None) for _ in range(nsites)]
    psi_npc = npc.Array.from_ndarray_trivial(
        psi_tensor, labels=[f"p{i}" for i in range(nsites)])
    psi = MPS.from_full(sites, psi_npc, normalize=True)
    psi.canonical_form()
    return psi


# ── Covariance matrix from MPS ────────────────────────────────────────────────

def one_site_rho_matrix(psi, i):
    rho = psi.get_rho_segment([i]).to_ndarray()
    d = rho.shape[0]
    return rho.reshape(d, d)


def two_site_rho_matrix(psi, i, j):
    rho = psi.get_rho_segment([i, j]).to_ndarray()
    d = rho.shape[0]
    return rho.reshape(d * d, d * d)


def covariance_matrix_from_mps(psi, Nsites, Nmax):
    ops = local_boson_ops(Nmax)
    x, p = ops["x"], ops["p"]
    local_ops = [x, p]
    nquad = 2 * Nsites
    Gamma = np.zeros((nquad, nquad), dtype=float)
    means = np.zeros(nquad, dtype=float)

    def site_and_op(a):
        site = a % Nsites
        kind = 0 if a < Nsites else 1
        return site, local_ops[kind]

    for a in range(nquad):
        i, Oi = site_and_op(a)
        means[a] = np.real(np.trace(one_site_rho_matrix(psi, i) @ Oi))

    for a in range(nquad):
        i, Oi = site_and_op(a)
        for b in range(nquad):
            j, Oj = site_and_op(b)
            if i == j:
                rho_i = one_site_rho_matrix(psi, i)
                val = np.real(np.trace(rho_i @ sym(Oi @ Oj)))
            else:
                rho_ij = two_site_rho_matrix(psi, i, j)
                val = np.real(np.trace(rho_ij @ np.kron(Oi, Oj)))
            Gamma[a, b] = val - means[a] * means[b]

    return sym(Gamma)


def covariance_from_single_mode_state(phi, Nmax):
    ops = local_boson_ops(Nmax)
    x, p = ops["x"], ops["p"]
    phi = phi / np.linalg.norm(phi)
    mx = np.real(np.vdot(phi, x @ phi))
    mp = np.real(np.vdot(phi, p @ phi))
    xx = np.real(np.vdot(phi, x @ x @ phi))
    pp = np.real(np.vdot(phi, p @ p @ phi))
    xp = np.real(np.vdot(phi, sym(x @ p) @ phi))
    V = np.array([[xx - mx*mx, xp - mx*mp], [xp - mx*mp, pp - mp*mp]])
    return sym(V)


def local_number_distribution_from_mps(psi, i):
    rho_i = one_site_rho_matrix(psi, i)
    probs = np.real(np.diag(rho_i))
    return probs / probs.sum()


# ── TEBD gates ────────────────────────────────────────────────────────────────

def onsite_unitary(Nmax, dt, m2, k, lam):
    ops = local_boson_ops(Nmax)
    x, p = ops["x"], ops["p"]
    H = 0.5 * p @ p + 0.5 * (m2 + 2 * k) * x @ x
    if lam > 0:
        H += lam * x @ x @ x @ x
    return expm(-1j * dt * H)


def bond_unitary(Nmax, dt, k):
    x = local_boson_ops(Nmax)["x"]
    return expm(-1j * dt * (-k * np.kron(x, x)))


def wormhole_unitary(Nmax, dt, g, omega0):
    ops = local_boson_ops(Nmax)
    x, p = ops["x"], ops["p"]
    H = g * (0.5 * omega0 * np.kron(x, x) + 0.5 / omega0 * np.kron(p, p))
    return expm(-1j * dt * H)


def SWAP_gate(Nmax):
    d = Nmax + 1
    U = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            U[j * d + i, i * d + j] = 1.0
    return U


def site_left_interleaved(k: int) -> int:
    """Physical MPS index of left-ring site ``k`` (interleaved layout ``L0,R0,L1,R1,...``)."""
    return 2 * k


def site_right_interleaved(k: int) -> int:
    """Physical MPS index of right-ring site ``k``."""
    return 2 * k + 1


# ── TEBD gate application ─────────────────────────────────────────────────────

def apply_one_site(psi, i, U):
    d = U.shape[0]
    op = npc.Array.from_ndarray_trivial(U.reshape(d, d), labels=["p", "p*"])
    psi.apply_local_op(i, op, unitary=True)


def apply_two_site_adjacent(psi, i, U):
    d = int(np.sqrt(U.shape[0]))
    op = npc.Array.from_ndarray_trivial(U.reshape(d, d, d, d), labels=["p0", "p1", "p0*", "p1*"])
    psi.apply_local_op(i, op, unitary=True)


def apply_coupling_bond(psi, i, j, U, Nmax):
    if j < i:
        i, j = j, i
    swap = SWAP_gate(Nmax)
    for k in range(j - 1, i, -1):
        apply_two_site_adjacent(psi, k, swap)
    apply_two_site_adjacent(psi, i, U)
    for k in range(i + 1, j):
        apply_two_site_adjacent(psi, k, swap)


def apply_ring_bond(psi, i, j, U, N, Nmax):
    if abs(i - j) == 1:
        apply_two_site_adjacent(psi, min(i, j), U)
        return
    if abs(i - j) != N - 1:
        raise ValueError(f"Sites {i},{j} are not nearest neighbours on a ring of size {N}.")
    left, right = (i, j) if i < j else (j, i)
    swap = SWAP_gate(Nmax)
    for k in range(right - 1, left, -1):
        apply_two_site_adjacent(psi, k, swap)
    apply_two_site_adjacent(psi, left, U)
    for k in range(left + 1, right):
        apply_two_site_adjacent(psi, k, swap)


# ── TEBD sweeps ───────────────────────────────────────────────────────────────

def tebd_step_ring(psi, start, end, U1, U2, N, Nmax):
    """First-order Suzuki-Trotter step on a ring segment [start, end)."""
    for i in range(start, end):
        apply_one_site(psi, i, U1)
    for i in range(start, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2)
    for i in range(start + 1, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2)
    apply_ring_bond(psi, end - 1, start, U2, N, Nmax)


def tebd_step_ring_2nd_order(psi, start, end, U1_half, U2_half, U2_full, N, Nmax):
    """Second-order (symmetric) Suzuki-Trotter step on a ring segment [start, end).
    Pass U1_half = expm(-i*(dt/2)*H_onsite), U2_half = expm(-i*(dt/2)*H_bond),
    U2_full = expm(-i*dt*H_bond).
    """
    for i in range(start, end):
        apply_one_site(psi, i, U1_half)
    for i in range(start, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2_half)
    for i in range(start + 1, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2_half)
    apply_ring_bond(psi, end - 1, start, U2_full, N, Nmax)
    for i in range(start + 1, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2_half)
    for i in range(start, end - 1, 2):
        apply_two_site_adjacent(psi, i, U2_half)
    for i in range(start, end):
        apply_one_site(psi, i, U1_half)


def tebd_step_coupled(psi, N, insert_idx, U1, U2, Uc, Nmax):
    """One Trotter step with both rings evolving and wormhole couplings active."""
    tebd_step_ring(psi, start=0,   end=N,   U1=U1, U2=U2, N=N, Nmax=Nmax)
    tebd_step_ring(psi, start=N,   end=2*N, U1=U1, U2=U2, N=N, Nmax=Nmax)
    for i in range(N):
        if i == insert_idx:
            continue
        apply_coupling_bond(psi, i, N + i, Uc, Nmax)


def tebd_step_interleaved_ring_side(psi, side: str, U1, U2, N: int, Nmax: int):
    """One first-order ring Trotter step on ``left`` or ``right`` interleaved subchain."""
    P = site_left_interleaved if side == "left" else site_right_interleaved
    for j in range(N):
        apply_one_site(psi, P(j), U1)
    for j in range(0, N - 1, 2):
        apply_coupling_bond(psi, P(j), P(j + 1), U2, Nmax)
    for j in range(1, N - 1, 2):
        apply_coupling_bond(psi, P(j), P(j + 1), U2, Nmax)
    apply_coupling_bond(psi, P(N - 1), P(0), U2, Nmax)


def tebd_step_interleaved_ring_side_2nd_order(
        psi, side: str, U1_half, U2_half, U2_full, N: int, Nmax: int):
    """Symmetric Trotter step on one interleaved ring (matches ``tebd_step_ring_2nd_order``)."""
    P = site_left_interleaved if side == "left" else site_right_interleaved
    for j in range(N):
        apply_one_site(psi, P(j), U1_half)
    for j in range(0, N - 1, 2):
        apply_coupling_bond(psi, P(j), P(j + 1), U2_half, Nmax)
    for j in range(1, N - 1, 2):
        apply_coupling_bond(psi, P(j), P(j + 1), U2_half, Nmax)
    apply_coupling_bond(psi, P(N - 1), P(0), U2_full, Nmax)
    for j in range(1, N - 1, 2):
        apply_coupling_bond(psi, P(j), P(j + 1), U2_half, Nmax)
    for j in range(0, N - 1, 2):
        apply_coupling_bond(psi, P(j), P(j + 1), U2_half, Nmax)
    for j in range(N):
        apply_one_site(psi, P(j), U1_half)


def tebd_step_coupled_interleaved(psi, N, insert_idx, U1, U2, Uc, Nmax):
    """Coupling Trotter step: both rings + adjacent L_i–R_i wormhole bonds (no SWAP)."""
    tebd_step_interleaved_ring_side(psi, "left", U1, U2, N, Nmax)
    tebd_step_interleaved_ring_side(psi, "right", U1, U2, N, Nmax)
    for i in range(N):
        if i == insert_idx:
            continue
        apply_two_site_adjacent(psi, site_left_interleaved(i), Uc)


def evolve_with_coupling(psi, N, insert_idx, t_couple, dt, Nmax, g, m2, k, lam):
    omega0 = np.sqrt(m2 + 2 * k)
    U1 = onsite_unitary(Nmax, dt, m2, k, lam)
    U2 = bond_unitary(Nmax, dt, k)
    Uc = wormhole_unitary(Nmax, dt, g, omega0)
    steps = int(t_couple / dt)
    for _ in tqdm(range(steps), desc="  coupling", leave=False):
        tebd_step_coupled_interleaved(psi, N, insert_idx, U1, U2, Uc, Nmax)


# ── Teleportation protocol ────────────────────────────────────────────────────

def teleportation_protocol(s, theta, insert_idx, cfg, psi_tensor):
    """
    Run the full wormhole teleportation protocol for a single input state.

    Inserts a Gaussian mode (squeezed by s, rotated by theta) at insert_idx,
    scrambles the left ring, applies wormhole coupling, then unscrambles
    the right ring.

    MPS sites use **interleaved** order (``L0,R0,L1,R1,...``); ``insert_idx`` is still the
    **left** ring label ``i``, applied on physical tensor axis ``2*i``.
    """
    phi = gaussian_mode(cfg.Nmax, r=s, theta=theta)
    insert_axis = site_left_interleaved(insert_idx)
    psi_insert = insert_with_env(psi_tensor, insert_axis, phi)
    psi = tensor_to_mps(psi_insert, cfg.Nmax)

    U1 = onsite_unitary(cfg.Nmax, cfg.dt, cfg.m2, cfg.k, cfg.lam)
    U2 = bond_unitary(cfg.Nmax, cfg.dt, cfg.k)
    U1_half = onsite_unitary(cfg.Nmax, cfg.dt / 2, cfg.m2, cfg.k, cfg.lam)
    U2_half = bond_unitary(cfg.Nmax, cfg.dt / 2, cfg.k)
    steps = int(cfg.t_scramble / cfg.dt)

    if cfg.second_order_ring_tebd:
        for _ in tqdm(range(steps), desc="  scramble L", leave=False):
            tebd_step_interleaved_ring_side_2nd_order(
                psi, "left", U1_half, U2_half, U2, cfg.N, cfg.Nmax)
    else:
        for _ in tqdm(range(steps), desc="  scramble L", leave=False):
            tebd_step_interleaved_ring_side(psi, "left", U1, U2, cfg.N, cfg.Nmax)

    evolve_with_coupling(
        psi, cfg.N, insert_idx,
        cfg.t_couple, cfg.dt, cfg.Nmax,
        cfg.g, cfg.m2, cfg.k, cfg.lam,
    )

    if cfg.second_order_ring_tebd:
        for _ in tqdm(range(steps), desc="  unscramble R", leave=False):
            tebd_step_interleaved_ring_side_2nd_order(
                psi, "right", U1_half, U2_half, U2, cfg.N, cfg.Nmax)
    else:
        for _ in tqdm(range(steps), desc="  unscramble R", leave=False):
            tebd_step_interleaved_ring_side(psi, "right", U1, U2, cfg.N, cfg.Nmax)

    return psi


# ── Fidelity measurement ──────────────────────────────────────────────────────

def fidelity_vs_site(insert_idx, input_ensemble, cfg, psi_tensor):
    """
    For each site on the right ring, fit a Gaussian channel to the teleportation
    map and return entanglement fidelities.

    Returns
    -------
    fid_symp : list[float]  fidelity with symplectic decoder, one per right-ring site
    fid_flip : list[float]  fidelity with flip decoder
    """
    Vins  = []
    Vouts = [[] for _ in range(cfg.N)]

    log.info("Running ensemble of %d input states ...", len(input_ensemble))
    for idx, (s, theta) in enumerate(tqdm(input_ensemble, desc="ensemble")):
        log.info("  [%d/%d] s=%.3f theta=%.3f", idx + 1, len(input_ensemble), s, theta)
        psi_out = teleportation_protocol(s, theta, insert_idx, cfg, psi_tensor)

        # truncation check — warn if population leaks to Nmax
        for i in range(2 * cfg.N):
            probs = local_number_distribution_from_mps(psi_out, i)
            print(f"site {i}: <n> = {sum(n*p for n,p in enumerate(probs)):.4f}, "f"P(Nmax) = {probs[-1]:.4e}, P(Nmax-1) = {probs[-2]:.4e}")


        Vout = covariance_matrix_from_mps(psi_out, 2 * cfg.N, cfg.Nmax)
        Vins.append(covariance_from_single_mode_state(
            gaussian_mode(Nmax=cfg.Nmax, r=s, theta=theta), cfg.Nmax))
        for i in range(cfg.N):
            Vouts[i].append(extract_subsystem_covariance(
                Vout, [site_right_interleaved(i)]))

    log.info("Fitting Gaussian channels ...")
    fid_symp, fid_flip = [], []
    for i in tqdm(range(cfg.N), desc="channel fit"):
        X, Y = fit_gaussian_channel(Vins, Vouts[i])
        Fs = entanglement_fidelity_gaussian(X, Y, decoder_from_X_symplectic(X), subtract_Y=False)
        Ff = entanglement_fidelity_gaussian(X, Y, decoder_from_X_flip(X),       subtract_Y=False)
        fid_symp.append(Fs)
        fid_flip.append(Ff)
        log.info(
            "  right-ring site %d (phys %d): F_symp=%.4f  F_flip=%.4f",
            i, site_right_interleaved(i), Fs, Ff)

    return fid_symp, fid_flip


# ── Main entry point ──────────────────────────────────────────────────────────

def build_initial_state(cfg):
    """Build the backward-scrambled TFD tensor (interleaved site order for the MPS)."""
    d = cfg.Nmax + 1
    dim_single = d ** cfg.N

    log.info("Building ring Hamiltonian (N=%d, Nmax=%d) ...", cfg.N, cfg.Nmax)
    H_quad = build_ring_H(cfg.N, cfg.Nmax, cfg.m2, cfg.k, cfg.lam)

    if cfg.use_normal_mode_tfd and dim_single <= _PASSIVE_UNITARY_MAX_DIM:
        log.info("Building TFD tensor via normal modes + passive rotation (beta=%.2f) ...", cfg.beta)
        try:
            psi_block = build_tfd_tensor_normal_modes(
                cfg.N, cfg.Nmax, cfg.m2, cfg.k, cfg.beta)
        except Exception as exc:
            log.warning("Normal-mode TFD failed (%s); falling back to dense H spectrum.", exc)
            psi_block = build_tfd_tensor(H_quad, cfg.N, cfg.Nmax, cfg.beta)
    else:
        log.info("Building TFD tensor from dense H spectrum (beta=%.2f) ...", cfg.beta)
        psi_block = build_tfd_tensor(H_quad, cfg.N, cfg.Nmax, cfg.beta)

    log.info("Applying backward modular evolution (t=%.2f) ...", cfg.t_scramble)
    U_back = expm(1j * cfg.t_scramble * H_quad)
    psi_mat = psi_block.reshape(dim_single, dim_single)
    psi_mat = U_back @ psi_mat
    psi_block = psi_mat.reshape([d] * (2 * cfg.N))
    return permute_lr_block_to_interleaved(psi_block, cfg.N)


def main():
    parser = argparse.ArgumentParser(
        description="Wormhole teleportation fidelity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick",  action="store_true",
                        help="Small-scale run for local testing (N=3, Nmax=4, short times)")
    parser.add_argument("--cute-config",  action="store_true",
                        help="Cute configuration (N=3, Nmax=4, t_scramble=2, t_couple=3, n_squeeze=4, n_theta=3)")
    # Override any Config field individually
    parser.add_argument("--N",             type=int,   help="Sites per ring")
    parser.add_argument("--Nmax",          type=int,   help="Bosonic truncation")
    parser.add_argument("--beta",          type=float, help="Inverse temperature")
    parser.add_argument("--m2",            type=float, help="Mass squared")
    parser.add_argument("--k",             type=float, help="Hopping")
    parser.add_argument("--lam",           type=float, help="Quartic coupling")
    parser.add_argument("--g",             type=float, help="Wormhole coupling")
    parser.add_argument("--dt",            type=float, help="Trotter step")
    parser.add_argument("--t-scramble",    type=float, dest="t_scramble")
    parser.add_argument("--t-couple",      type=float, dest="t_couple")
    parser.add_argument("--n-squeeze",     type=int,   dest="n_squeeze")
    parser.add_argument("--n-theta",       type=int,   dest="n_theta")
    parser.add_argument("--squeeze-range", type=float, dest="squeeze_range")
    parser.add_argument("--insert-idx",    type=int,   dest="insert_idx", default=1)
    parser.add_argument("--output",        default="plots/site_vs_fidelity.pdf")
    parser.add_argument("--dense-h-tfd", action="store_true",
                        help="Build initial TFD from dense H spectrum instead of normal modes")
    parser.add_argument("--first-order-tebd", action="store_true",
                        help="Use first-order ring Trotter for scramble/unscramble")
    args = parser.parse_args()

    cfg = QUICK if args.quick else CUTE_CONFFIG if args.cute_config else FULL
    # Apply any explicit overrides
    for field in ["N", "Nmax", "beta", "m2", "k", "lam", "g", "dt",
                  "t_scramble", "t_couple", "n_squeeze", "n_theta", "squeeze_range"]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    if args.dense_h_tfd:
        cfg.use_normal_mode_tfd = False
    if args.first_order_tebd:
        cfg.second_order_ring_tebd = False

    log.info("Config: %s", cfg)

    psi_tensor = build_initial_state(cfg)

    Ss     = np.linspace(-cfg.squeeze_range, cfg.squeeze_range, cfg.n_squeeze)
    Thetas = np.linspace(0, 2 * np.pi, cfg.n_theta, endpoint=False)
    input_ensemble = [(s, th) for s in Ss for th in Thetas]
    log.info("Ensemble size: %d", len(input_ensemble))

    Fs, Ff = fidelity_vs_site(args.insert_idx, input_ensemble, cfg, psi_tensor)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sites = np.array([site_right_interleaved(i) for i in range(cfg.N)])
    plt.figure()
    plt.plot(sites, Fs, marker="o", label="symplectic decoder")
    plt.plot(sites, Ff, marker="s", label="flip decoder")
    plt.xlabel("site")
    plt.ylabel("entanglement fidelity")
    plt.title(f"N={cfg.N}, Nmax={cfg.Nmax}, β={cfg.beta}, g={cfg.g}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    log.info("Saved plot to %s", args.output)
    log.info("done")


if __name__ == "__main__":
    main()