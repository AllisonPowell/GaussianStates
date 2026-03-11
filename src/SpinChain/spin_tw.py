import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.linalg import np_conserved as npc


# ============================================================
# Basic gates
# ============================================================

def H_gate():
    return (1 / np.sqrt(2)) * np.array([[1, 1],
                                        [1, -1]], dtype=complex)

def H_gate_npc():
    return npc.Array.from_ndarray_trivial(H_gate(), labels=['p', 'p*'])

def X_gate():
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)

def Z_gate():
    return np.array([[1, 0],
                     [0, -1]], dtype=complex)

def S_gate():
    return np.array([[1, 0],
                     [0, 1j]], dtype=complex)

def T_gate():
    return np.array([[1, 0],
                     [0, np.exp(1j * np.pi / 4)]], dtype=complex)

def CNOT_gate():
    U = np.zeros((4, 4), dtype=complex)
    U[0, 0] = 1
    U[1, 1] = 1
    U[2, 3] = 1
    U[3, 2] = 1
    return U

def CZ_gate():
    return np.diag([1, 1, 1, -1]).astype(complex)

def SWAP_gate():
    U = np.zeros((4, 4), dtype=complex)
    U[0, 0] = 1
    U[1, 2] = 1
    U[2, 1] = 1
    U[3, 3] = 1
    return U

def ZZ_coupling_gate(g):
    """
    U = exp(i g Z \otimes Z)
    basis ordering: |00>, |01>, |10>, |11>
    """
    phases = np.array([
        np.exp(1j * g),   # +1 * +1
        np.exp(-1j * g),  # +1 * -1
        np.exp(-1j * g),  # -1 * +1
        np.exp(1j * g),   # -1 * -1
    ], dtype=complex)
    return np.diag(phases)


# ============================================================
# Random Floquet gates
# ============================================================

def kron2(A, B):
    return np.kron(A, B)

def random_single_qubit_clifford():
    I = np.eye(2, dtype=complex)
    X = X_gate()
    Z = Z_gate()
    H = H_gate()
    S = S_gate()
    pool = [I, H, S, H @ S, S @ H, H @ S @ H, X, Z]
    return pool[np.random.randint(len(pool))]

def sample_two_qubit_gate(p_magic):
    """
    Mostly Clifford; with probability p_magic insert a non-Clifford T.
    """
    u = np.random.rand()
    if u > p_magic:
        u1 = random_single_qubit_clifford()
        u2 = random_single_qubit_clifford()
        v1 = random_single_qubit_clifford()
        v2 = random_single_qubit_clifford()
        U = kron2(v1, v2) @ CZ_gate() @ kron2(u1, u2)
    else:
        u1 = random_single_qubit_clifford()
        u2 = random_single_qubit_clifford()
        V = CZ_gate() @ kron2(T_gate(), np.eye(2, dtype=complex))
        U = V @ kron2(u1, u2)
    return U


# ============================================================
# MPS init
# ============================================================

def init_sites_total(N, conserve=None):
    """
    Site ordering:
      0        : reference qubit
      1        : environment qubit
      2..N+1   : left chain
      N+2..2N+1: right chain
    """
    return [SpinHalfSite(conserve=conserve) for _ in range(2 * N + 2)]

def init_product_mps(sites, state0="up"):
    prod_state = [state0] * len(sites)
    return MPS.from_product_state(sites, prod_state, bc='finite')


# ============================================================
# Index helpers
# ============================================================

def idx_ref():
    return 0

def idx_env():
    return 1

def idx_L(i, N):
    """
    i = 0..N-1
    """
    return 2 + i

def idx_R(i, N):
    """
    i = 0..N-1
    """
    return 2 + N + i


# ============================================================
# Gate application
# ============================================================

def apply_one_site_unitary(psi, i, U1):
    op = npc.Array.from_ndarray_trivial(U1, labels=['p', 'p*'])
    psi.apply_local_op(i, op, unitary=True, renormalize=False)
    return psi

def apply_two_site_unitary_adjacent(psi, i, j, U2, chi_max=256, svd_cut=1e-10):
    """
    Adjacent only.
    """
    assert j == i + 1, "Expected adjacent sites."
    op = npc.Array.from_ndarray_trivial(
        U2.reshape(2, 2, 2, 2),
        labels=['p0', 'p1', 'p0*', 'p1*']
    )
    psi.apply_local_op(i, op, unitary=True, renormalize=False, cutoff=svd_cut)
    return psi

def apply_two_site_unitary_nonlocal(psi, i, j, U2, chi_max=256, svd_cut=1e-10):
    """
    Apply a 2-site gate between arbitrary sites i < j by routing with SWAPs.
    Restores original ordering afterwards.
    """
    if i == j:
        raise ValueError("Need distinct sites.")
    if j < i:
        i, j = j, i

    # move site j left until it sits at i+1
    for k in range(j - 1, i, -1):
        apply_two_site_unitary_adjacent(psi, k, k + 1, SWAP_gate(), chi_max, svd_cut)

    # apply desired gate on (i, i+1)
    apply_two_site_unitary_adjacent(psi, i, i + 1, U2, chi_max, svd_cut)

    # swap back to restore original site order
    for k in range(i + 1, j):
        apply_two_site_unitary_adjacent(psi, k, k + 1, SWAP_gate(), chi_max, svd_cut)

    return psi


# ============================================================
# TFD preparation
# ============================================================

def prepare_beta0_tfd(psi, N, chi_max=256, svd_cut=1e-10):
    """
    Prepare the beta=0 TFD:
        product over i of Bell pairs between L_i and R_i
    while leaving ref/env in |0>.
    """
    for i in range(N):
        l = idx_L(i, N)
        r = idx_R(i, N)

        # create Bell pair on (L_i, R_i)
        apply_one_site_unitary(psi, l, H_gate())
        apply_two_site_unitary_nonlocal(psi, l, r, CNOT_gate(), chi_max, svd_cut)

    psi.canonical_form()
    return psi


# ============================================================
# Floquet layers on one side
# ============================================================

def build_side_floquet_layers(N, p_magic, side="L"):
    """
    Logical layers, not yet mapped to physical indices.
    Each entry is (logical_i, logical_j, U2).
    """
    layers = []

    even_layer = []
    for i in range(0, N - 1, 2):
        even_layer.append((i, i + 1, sample_two_qubit_gate(p_magic)))
    layers.append(even_layer)

    odd_layer = []
    for i in range(1, N - 1, 2):
        odd_layer.append((i, i + 1, sample_two_qubit_gate(p_magic)))
    layers.append(odd_layer)

    return layers

def apply_side_layers(psi, N, layers, side, inverse, chi_max, svd_cut):
    """
    Apply one Floquet period to left or right chain.
    If inverse=True, apply reverse order with dagger gates.
    """
    layer_list = layers[::-1] if inverse else layers

    for layer in layer_list:
        gate_list = layer[::-1] if inverse else layer
        for i, j, U2 in gate_list:
            if inverse:
                Uuse = U2.conj().T
            else:
                Uuse = U2

            if side == "L":
                a = idx_L(i, N)
                b = idx_L(j, N)
            elif side == "R":
                a = idx_R(i, N)
                b = idx_R(j, N)
            else:
                raise ValueError("side must be 'L' or 'R'.")

            apply_two_site_unitary_adjacent(psi, a, b, Uuse, chi_max, svd_cut)

        psi.canonical_form()

    return psi


# ============================================================
# Message insertion
# ============================================================

def insert_message_with_env(psi, N, insert_idx, chi_max, svd_cut):
    """
    Exact pure insertion:
      1) swap old L_insert into env
      2) prepare Bell pair between ref and L_insert

    Since env starts in |0>, after SWAP the insert site is reset to |0>.
    """
    ref = idx_ref()
    env = idx_env()
    l_ins = idx_L(insert_idx, N)

    # move old left-site content into env
    apply_two_site_unitary_nonlocal(psi, env, l_ins, SWAP_gate(), chi_max, svd_cut)

    # prepare Bell pair between ref and inserted site
    apply_one_site_unitary(psi, ref, H_gate())
    apply_two_site_unitary_nonlocal(psi, ref, l_ins, CNOT_gate(), chi_max, svd_cut)

    psi.canonical_form()
    return psi


# ============================================================
# Traversable coupling
# ============================================================

def apply_traversable_coupling(psi, N, g, exclude_idx=None, chi_max=256, svd_cut=1e-10):
    """
    Apply U = prod_i exp(i g Z_{L_i} Z_{R_i})
    optionally excluding the insert site.
    """
    Uzz = ZZ_coupling_gate(g)

    for i in range(N):
        if exclude_idx is not None and i == exclude_idx:
            continue
        for j in range(N):
            if exclude_idx is not None and j == exclude_idx:
                continue
            l = idx_L(i, N)
            r = idx_R(j, N)
            apply_two_site_unitary_nonlocal(psi, l, r, Uzz,chi_max,svd_cut)    
        #l = idx_L(i, N)
        #r = idx_R(i, N)
        #apply_two_site_unitary_nonlocal(psi, l, r, Uzz, chi_max, svd_cut)

    psi.canonical_form()
    return psi


# ============================================================
# Reduced density matrices and fidelity
# ============================================================



def bell_phi_plus_density():
    v = np.zeros(4, dtype=complex)
    v[0] = 1 / np.sqrt(2)
    v[3] = 1 / np.sqrt(2)
    return np.outer(v, v.conj())

def rho_sites_matrix(psi, sites):
    rho_npc = psi.get_rho_segment(sites)   # npc.Array
    labels = list(rho_npc.get_leg_labels())
    rho = rho_npc.to_ndarray()

    n = len(sites)

    # collect ket legs first, bra legs second
    ket_axes = [labels.index(f'p{i}') for i in range(n)]
    bra_axes = [labels.index(f'p{i}*') for i in range(n)]

    perm = ket_axes + bra_axes
    rho = np.transpose(rho, perm)

    dim = 2**n
    rho = rho.reshape(dim, dim)
    return rho

def bell_fidelity_two_sites(psi, s1, s2):
    rho = rho_sites_matrix(psi, [s1, s2])
    bell = bell_phi_plus_density()
    return float(np.real(np.trace(rho @ bell)))


# ============================================================
# Full traversable wormhole experiment
# ============================================================

def traversable_wormhole_protocol(
    N,
    insert_idx,
    n_back,
    n_forward,
    n_right,
    p_magic,
    g_couple,
    chi_max,
    svd_cut,
):
    """
    Protocol:
      1) prepare beta=0 TFD on L-R
      2) evolve L backward by inverse Floquet
      3) insert Bell pair at left insert site using ref+env
      4) evolve L forward
      5) apply traversable coupling
      6) evolve R forward
      7) measure Bell fidelity between ref and R_insert
    """

    sites = init_sites_total(N, conserve=None)
    psi = init_product_mps(sites, state0="up")

    # exact beta=0 TFD
    prepare_beta0_tfd(psi, N, chi_max=chi_max, svd_cut=svd_cut)

    # same sampled modular-Floquet layers used on both sides
    left_layers = build_side_floquet_layers(N, p_magic=p_magic, side="L")
    right_layers = left_layers  # keep identical for the doubled system

    # 1) backward evolution on left

    for _ in range(n_back):
        apply_side_layers(psi, N, left_layers, side="L", inverse=True,
                        chi_max=chi_max, svd_cut=svd_cut)

    # 2) insert the message
    insert_message_with_env(psi, N, insert_idx, chi_max=chi_max, svd_cut=svd_cut)

    print("insert L_insert =",n_right,bell_fidelity_two_sites(psi, idx_ref(), idx_L(insert_idx, N)))
    print("insert R_insert =",n_right,bell_fidelity_two_sites(psi, idx_ref(), idx_R(insert_idx, N)))

    # 3) forward evolution on left
    psi0 = psi.copy()
    #while bell_fidelity_two_sites(psi, idx_ref(), idx_L(insert_idx, N)) > .2501:
        #psi = psi0
    for _ in range(n_forward):
        apply_side_layers(psi, N, left_layers, side="L", inverse=False,
                          chi_max=chi_max, svd_cut=svd_cut)
    print("before L_insert =",n_right,bell_fidelity_two_sites(psi, idx_ref(), idx_L(insert_idx, N)))
    print("before R_insert =",n_right,bell_fidelity_two_sites(psi, idx_ref(), idx_R(insert_idx, N)))

    # 4) traversable coupling
    apply_traversable_coupling(psi, N, g_couple, exclude_idx=insert_idx,
                               chi_max=chi_max, svd_cut=svd_cut)

    # 5) forward evolution on right
    for _ in range(n_right):
        apply_side_layers(psi, N, right_layers, side="R", inverse=False,
                          chi_max=chi_max, svd_cut=svd_cut)

    print("after L_insert =",n_right,bell_fidelity_two_sites(psi, idx_ref(), idx_L(insert_idx, N)))
    print("after R_insert =",n_right,bell_fidelity_two_sites(psi, idx_ref(), idx_R(insert_idx, N)))

    for _ in range(2*n_right):
        apply_side_layers(psi, N, right_layers, side="R", inverse=True,
                          chi_max=chi_max, svd_cut=svd_cut)
    print("final L_insert =",n_right,bell_fidelity_two_sites(psi, idx_ref(), idx_L(insert_idx, N)))
    print("final R_insert =",n_right,bell_fidelity_two_sites(psi, idx_ref(), idx_R(insert_idx, N)))


    # 6) evaluate recovery at the matching right site
    F = bell_fidelity_two_sites(psi, idx_ref(), idx_R(insert_idx, N))
    return psi, F


# ============================================================
# Time scan
# ============================================================

def wormhole_time_scan(
    N,
    insert_idx,
    times,
    p_magic,
    g_couple,
    chi_max,
    svd_cut,
):
    """
    Scan the wormhole protocol as a function of backward/forward scrambling time.
    Here we use n_back = n_forward = t and n_right = t.
    """
    out = []

    for t in times:
        _, F = traversable_wormhole_protocol(
            N=N,
            insert_idx=insert_idx,
            n_back=t,
            n_forward=t,
            n_right=t,
            p_magic=p_magic,
            g_couple=g_couple,
            chi_max=chi_max,
            svd_cut=svd_cut,
        )
        out.append((t, F))
        #print(f"t={t}, F={F:.6f}")

    return out

N = 8

data = wormhole_time_scan(
    N,
    insert_idx=1,
    times=range(8, 12),
    p_magic=0.0,
    g_couple=1/N,
    chi_max=128,
    svd_cut=1e-10,
)

