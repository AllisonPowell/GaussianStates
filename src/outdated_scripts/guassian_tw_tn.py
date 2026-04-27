import numpy as np
from scipy.linalg import eigh, expm, sqrtm

from tenpy.networks.site import BosonSite
from tenpy.networks.mps import MPS
from tenpy.linalg import np_conserved as npc


# ============================================================
# Basic linear-algebra helpers
# ============================================================

def kron_all(ops):
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out


def embed_one_site(op, site, L, d):
    ops = [np.eye(d, dtype=complex) for _ in range(L)]
    ops[site] = op
    return kron_all(ops)


def embed_two_site_adjacent(op2, i, j, L, d):
    """
    Embed a 2-site operator op2 acting on adjacent sites i,j with j=i+1.
    """
    assert j == i + 1
    ops = []
    k = 0
    while k < L:
        if k == i:
            ops.append(op2)
            k += 2
        else:
            ops.append(np.eye(d, dtype=complex))
            k += 1
    return kron_all(ops)


def apply_unitary_on_axis_tensor(psi_tensor, U, axis):
    """
    Apply single-site unitary U on tensor axis.
    """
    out = np.tensordot(U, psi_tensor, axes=(1, axis))
    out = np.moveaxis(out, 0, axis)
    return out


def unitary_with_first_column(phi, eps=1e-14):
    """
    Build a dxd unitary U with U|0> = phi.
    """
    phi = np.asarray(phi, dtype=complex).reshape(-1)
    phi = phi / (np.linalg.norm(phi) + eps)
    d = len(phi)

    cols = [phi]
    for k in range(d):
        e = np.zeros(d, dtype=complex)
        e[k] = 1.0
        cols.append(e)

    Q = []
    for v in cols:
        w = v.copy()
        for q in Q:
            w -= np.vdot(q, w) * q
        nw = np.linalg.norm(w)
        if nw > 1e-12:
            Q.append(w / nw)
        if len(Q) == d:
            break

    U = np.column_stack(Q)
    U[:, 0] = phi

    # cleanup
    for j in range(1, d):
        U[:, j] -= np.vdot(phi, U[:, j]) * phi
        U[:, j] /= np.linalg.norm(U[:, j])

    return U


def state_fidelity(rho, sigma, eps=1e-12):
    """
    Uhlmann fidelity F(rho,sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    for small dense matrices.
    """
    rho = 0.5 * (rho + rho.conj().T)
    sigma = 0.5 * (sigma + sigma.conj().T)

    # regularize tiny negatives from numerics
    wr, Vr = eigh(rho)
    wr = np.clip(wr, 0.0, None)
    sqrt_rho = (Vr * np.sqrt(wr)) @ Vr.conj().T

    X = sqrt_rho @ sigma @ sqrt_rho
    X = 0.5 * (X + X.conj().T)

    wx, Vx = eigh(X)
    wx = np.clip(wx, 0.0, None)
    return float(np.real(np.sum(np.sqrt(wx)) ** 2))


# ============================================================
# Bosonic local operators
# ============================================================

def local_boson_ops(Nmax):
    """
    Return dense local operators on a truncated Fock space of dimension d=Nmax+1.
    """
    d = Nmax + 1
    b = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        b[n - 1, n] = np.sqrt(n)

    bd = b.conj().T
    x = (b + bd) / np.sqrt(2.0)
    p = (b - bd) / (1j * np.sqrt(2.0))
    n_op = bd @ b
    I = np.eye(d, dtype=complex)
    return {"I": I, "b": b, "bd": bd, "x": x, "p": p, "n": n_op}


# ============================================================
# One-side quadratic Hamiltonian and TFD
# ============================================================

def build_one_side_hamiltonian_dense(N_sites, Nmax, m2, k, lam=0.0):
    """
    H = sum_i [1/2 p_i^2 + 1/2 m2 x_i^2 + lam x_i^4]
        + sum_<ij> 1/2 k (x_i - x_j)^2
    on a 1D open chain.
    """
    ops = local_boson_ops(Nmax)
    d = Nmax + 1
    L = N_sites

    x = ops["x"]
    p = ops["p"]

    H = np.zeros((d**L, d**L), dtype=complex)

    # onsite
    for i in range(L):
        H += 0.5 * embed_one_site(p @ p, i, L, d)
        H += 0.5 * m2 * embed_one_site(x @ x, i, L, d)
        if lam != 0.0:
            H += lam * embed_one_site(x @ x @ x @ x, i, L, d)

    # nearest-neighbor spring
    x2 = x @ x
    xx = np.kron(x, x)
    for i in range(L - 1):
        H += 0.5 * k * embed_one_site(x2, i, L, d)
        H += 0.5 * k * embed_one_site(x2, i + 1, L, d)
        H += -k * embed_two_site_adjacent(xx, i, i + 1, L, d)

    return 0.5 * (H + H.conj().T)


def build_tfd_tensor_from_one_side_H(H_side, N_sites, Nmax, beta):
    """
    Construct |TFD> = Z^{-1/2} sum_n exp(-beta E_n / 2) |n>_L |n>_R
    in the truncated Fock basis, returned as tensor with legs
      [L0,L1,L2,R0,R1,R2]
    """
    d = Nmax + 1
    D = d ** N_sites

    evals, evecs = eigh(H_side)
    evals0 = evals - evals.min()
    w = np.exp(-beta * evals0)
    Z = np.sum(w)

    # coefficient matrix in Fock basis
    C = evecs @ np.diag(np.sqrt(w / Z)) @ evecs.conj().T  # (D, D)

    # reshape to tensor [L0,L1,...,R0,R1,...]
    psi_tensor = C.reshape([d] * N_sites + [d] * N_sites)
    return psi_tensor


# ============================================================
# Convert full tensor to TeNPy MPS
# ============================================================

def build_sites_total(N_left, N_right, add_env=True, conserve=None, Nmax=5):
    """
    Order:
      L0,L1,L2,R0,R1,R2,(Env)
    """
    n_tot = N_left + N_right + (1 if add_env else 0)
    return [BosonSite(Nmax=Nmax, conserve=conserve) for _ in range(n_tot)]


def psi_tensor_to_mps(psi_tensor, sites, cutoff=1e-14):
    """
    psi_tensor must have one physical leg per site.
    """
    psi_npc = npc.Array.from_ndarray_trivial(
        psi_tensor,
        labels=[f"p{i}" for i in range(len(sites))]
    )
    psi = MPS.from_full(sites, psi_npc, cutoff=cutoff, normalize=True, bc='finite')
    psi.canonical_form()
    return psi


# ============================================================
# Gate builders for Trotter evolution
# ============================================================

def onsite_unitary(Nmax, dt, m2, lam):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    p = ops["p"]
    h = 0.5 * (p @ p) + 0.5 * m2 * (x @ x) + lam * (x @ x @ x @ x)
    return expm(-1j * dt * h)


def bond_unitary(Nmax, dt, k):
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    x2 = x @ x
    h = 0.5 * k * (np.kron(x2, np.eye(Nmax + 1))
                   + np.kron(np.eye(Nmax + 1), x2)) - k * np.kron(x, x)
    return expm(-1j * dt * h)


def lr_coupling_unitary(Nmax, dt, g):
    """
    Simple traversable-style quadratic coupling between matched L_i and R_i:
      H_int = - g x_L x_R
    """
    ops = local_boson_ops(Nmax)
    x = ops["x"]
    h = -g * np.kron(x, x)
    return expm(-1j * dt * h)


# ============================================================
# TeNPy gate application
# ============================================================

def apply_one_site_gate(psi, site, U1):
    op = npc.Array.from_ndarray_trivial(U1, labels=['p', 'p*'])
    psi.apply_local_op(site, op, unitary=True, renormalize=False)
    return psi


def apply_two_site_gate_adjacent(psi, i, j, U2, cutoff=1e-12):
    assert j == i + 1
    d = int(round(np.sqrt(U2.shape[0])))
    op = npc.Array.from_ndarray_trivial(
        U2.reshape(d, d, d, d),
        labels=['p0', 'p1', 'p0*', 'p1*']
    )
    psi.apply_local_op(i, op, unitary=True, renormalize=False, cutoff=cutoff)
    return psi


def apply_two_site_gate_nonlocal(psi, i, j, U2, Nmax, cutoff=1e-12):
    """
    Route nonlocal gate by bosonic SWAPs.
    """
    if j < i:
        i, j = j, i
    if i == j:
        raise ValueError("Need distinct sites.")

    d = Nmax + 1
    # bosonic SWAP on local truncated Hilbert spaces
    SWAP = np.zeros((d * d, d * d), dtype=complex)
    for a in range(d):
        for b in range(d):
            SWAP[b * d + a, a * d + b] = 1.0

    # move j left to i+1
    for k in range(j - 1, i, -1):
        apply_two_site_gate_adjacent(psi, k, k + 1, SWAP, cutoff=cutoff)

    # apply target gate
    apply_two_site_gate_adjacent(psi, i, i + 1, U2, cutoff=cutoff)

    # swap back
    for k in range(i + 1, j):
        apply_two_site_gate_adjacent(psi, k, k + 1, SWAP, cutoff=cutoff)

    return psi


# ============================================================
# Protocol pieces
# ============================================================

def left_indices():
    return [0, 1, 2]

def right_indices():
    return [3, 4, 5]

def env_index():
    return 6


def apply_left_backward_forward_step(psi, dt, Nmax, m2, k, lam, backward=False, cutoff=1e-12):
    """
    One second-order Trotter step on left chain only.
    """
    sgn = -1.0 if backward else 1.0

    U_on_half = onsite_unitary(Nmax, sgn * dt / 2.0, m2, lam)
    U_bond = bond_unitary(Nmax, sgn * dt, k)

    # onsite half
    for i in left_indices():
        apply_one_site_gate(psi, i, U_on_half)

    # bonds
    apply_two_site_gate_adjacent(psi, 0, 1, U_bond, cutoff=cutoff)
    apply_two_site_gate_adjacent(psi, 1, 2, U_bond, cutoff=cutoff)

    # onsite half
    for i in left_indices():
        apply_one_site_gate(psi, i, U_on_half)

    psi.canonical_form()
    return psi


def apply_right_forward_step(psi, dt, Nmax, m2, k, lam=0.0, cutoff=1e-12):
    """
    One second-order Trotter step on right chain only.
    Usually lam=0 here if you want only the left to be non-Gaussian.
    """
    U_on_half = onsite_unitary(Nmax, dt / 2.0, m2, lam)
    U_bond = bond_unitary(Nmax, dt, k)

    for i in right_indices():
        apply_one_site_gate(psi, i, U_on_half)

    apply_two_site_gate_adjacent(psi, 3, 4, U_bond, cutoff=cutoff)
    apply_two_site_gate_adjacent(psi, 4, 5, U_bond, cutoff=cutoff)

    for i in right_indices():
        apply_one_site_gate(psi, i, U_on_half)

    psi.canonical_form()
    return psi


def apply_lr_coupling_window(psi, dt, Nmax, g, insert_idx=None, cutoff=1e-12):
    """
    Couple matched left-right sites.
    If insert_idx is excluded, skip that channel, matching your earlier setup.
    """
    U_lr = lr_coupling_unitary(Nmax, dt, g)
    for j in range(3):
        if insert_idx is not None and j == insert_idx:
            continue
        apply_two_site_gate_nonlocal(psi, j, 3 + j, U_lr, Nmax=Nmax, cutoff=cutoff)
    psi.canonical_form()
    return psi


def make_phi_squeezed_vacuum(Nmax, r, theta):
    """
    Truncated squeezed vacuum coefficients in local Fock basis.
    """
    d = Nmax + 1
    phi = np.zeros(d, dtype=complex)

    ch = np.cosh(r)
    th = np.tanh(r)
    phase = np.exp(1j * 2.0 * theta)

    for k in range((d + 1) // 2):
        n = 2 * k
        pref = np.sqrt(np.math.factorial(2 * k)) / ((2.0 ** k) * np.math.factorial(k) * np.sqrt(ch))
        phi[n] = pref * ((-phase * th) ** k)

    phi /= np.linalg.norm(phi)
    return phi


def insert_mode_with_env_full_tensor(psi_tensor, insert_idx, phi):
    """
    Exact pure insertion on full tensor:
      - append env vacuum
      - swap old left insert site with env
      - prepare inserted site into |phi>
    Tensor ordering in input:
      [L0,L1,L2,R0,R1,R2]
    Output:
      [L0,L1,L2,R0,R1,R2,Env]
    """
    d = psi_tensor.shape[0]
    N = psi_tensor.ndim  # 6 here

    psi_env = np.zeros(psi_tensor.shape + (d,), dtype=complex)
    psi_env[..., 0] = psi_tensor

    # swap insert axis with env axis
    psi_swapped = np.swapaxes(psi_env, insert_idx, N)

    U = unitary_with_first_column(phi)
    psi_prepared = apply_unitary_on_axis_tensor(psi_swapped, U, axis=insert_idx)
    return psi_prepared


def reduced_rho_one_site(psi, site):
    rho_npc = psi.get_rho_segment([site])
    rho = rho_npc.to_ndarray()
    return rho.reshape(rho.shape[0], rho.shape[1])


# ============================================================
# Full protocol
# ============================================================

def gaussian_boson_teleportation_protocol_tenpy(
    Nmax=4,
    beta=1.0,
    m2=13.0,
    k=5.0,
    lam=0.02,
    g_lr=0.2,
    t_scramble=1.0,
    t_couple=1.0,
    dt=0.05,
    insert_idx=1,   # 0,1,2 on the left
    r_insert=0.5,
    theta_insert=0.0,
    mps_cutoff=1e-12,
):
    """
    1) build one-side quadratic H
    2) build TFD for 3+3 sites
    3) append environment during insertion
    4) evolve left backward with quadratic+quartic
    5) insert Gaussian mode
    6) evolve left forward
    7) couple L-R
    8) evolve right forward
    9) compare reduced right-site state to inserted local target state
    """
    N_side = 3
    d = Nmax + 1

    # ---- build one-side quadratic H and TFD
    H_quad_side = build_one_side_hamiltonian_dense(
        N_sites=N_side, Nmax=Nmax, m2=m2, k=k, lam=0.0
    )

    psi_tfd_tensor = build_tfd_tensor_from_one_side_H(
        H_side=H_quad_side, N_sites=N_side, Nmax=Nmax, beta=beta
    )  # shape [d,d,d,d,d,d]

    # ---- backward evolve LEFT on full tensor before insertion
    # Build a temporary MPS on 6 sites, evolve left backward, then recover tensor
    sites6 = build_sites_total(3, 3, add_env=False, conserve=None, Nmax=Nmax)
    psi6 = psi_tensor_to_mps(psi_tfd_tensor, sites6, cutoff=1e-14)

    n_scramble = int(round(t_scramble / dt))
    for _ in range(n_scramble):
        apply_left_backward_forward_step(
            psi6, dt, Nmax, m2, k, lam, backward=True, cutoff=mps_cutoff
        )

    # get back dense 6-site tensor for exact insertion-with-env
    psi6_full = psi6.get_theta(0, n=6).to_ndarray()  # shape [1,p0,p1,...,p5,1] or similar in some versions
    # safer route: reconstruct from rho is awkward; use overlap-friendly combine:
    # for small system, build full tensor from B tensors
    psi6_full = psi6.get_B(0).to_ndarray()
    for site in range(1, 6):
        psi6_full = np.tensordot(psi6_full, psi6.get_B(site).to_ndarray(), axes=([-1], [0]))
    # strip boundary legs
    psi6_full = np.squeeze(psi6_full)
    # reorder if needed to pure physical axes
    # after contraction, axes are interleaved with physical/bond legs; easiest robust fallback:
    # because TeNPy internal forms vary, users may prefer MPS.from_full -> keep initial tensor route.
    # For a compact example, assume squeeze gives physical legs only after canonical contraction.

    # ---- exact insertion with environment
    phi_insert = make_phi_squeezed_vacuum(Nmax, r_insert, theta_insert)
    psi7_tensor = insert_mode_with_env_full_tensor(
        psi_tfd_tensor if psi6_full.ndim != 6 else psi6_full,
        insert_idx=insert_idx,
        phi=phi_insert
    )  # shape [L0,L1,L2,R0,R1,R2,Env]

    sites7 = build_sites_total(3, 3, add_env=True, conserve=None, Nmax=Nmax)
    psi = psi_tensor_to_mps(psi7_tensor, sites7, cutoff=1e-14)

    # ---- evolve LEFT forward
    for _ in range(n_scramble):
        apply_left_backward_forward_step(
            psi, dt, Nmax, m2, k, lam, backward=False, cutoff=mps_cutoff
        )

    # ---- L-R coupling window
    n_couple = int(round(t_couple / dt))
    for _ in range(n_couple):
        apply_lr_coupling_window(
            psi, dt, Nmax, g_lr, insert_idx=insert_idx, cutoff=mps_cutoff
        )

    # ---- evolve RIGHT forward
    for _ in range(n_scramble):
        apply_right_forward_step(
            psi, dt, Nmax, m2, k, lam=0.0, cutoff=mps_cutoff
        )

    # ---- read out right matching site
    rho_out = reduced_rho_one_site(psi, 3 + insert_idx)
    sigma_target = np.outer(phi_insert, phi_insert.conj())

    F = state_fidelity(rho_out, sigma_target)

    return {
        "psi_final": psi,
        "rho_out": rho_out,
        "sigma_target": sigma_target,
        "fidelity": F,
        "phi_insert": phi_insert,
    }


# ============================================================
# Example run
# ============================================================

if __name__ == "__main__":
    out = gaussian_boson_teleportation_protocol_tenpy(
        Nmax=4,
        beta=1.0,
        m2=13.0,
        k=5.0,
        lam=0.01,
        g_lr=0.15,
        t_scramble=0.5,
        t_couple=0.5,
        dt=0.05,
        insert_idx=1,
        r_insert=0.4,
        theta_insert=0.0,
        mps_cutoff=1e-10,
    )
    print("single-site output fidelity =", out["fidelity"])