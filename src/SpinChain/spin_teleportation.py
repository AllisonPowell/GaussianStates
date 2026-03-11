import numpy as np
import tenpy
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.linalg import np_conserved as npc
import matplotlib.pyplot as plt


def H_gate_npc():
    H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                     [1, -1]], dtype=complex)
    return npc.Array.from_ndarray_trivial(H, labels=['p', 'p*'])


def inject_bell_pair(psi, A_site, chi_max=200, svd_cut=1e-10):
    R_site = 0
    psi.apply_local_op(R_site, H_gate_npc(), unitary=True, renormalize=False)

    U = CNOT_gate()
    apply_two_site_unitary(psi, R_site, A_site, U, chi_max, svd_cut)
    return psi


def apply_two_site_unitary(psi, i, j, U2, chi_max, svd_cut):
    assert j == i + 1, "This routine expects nearest-neighbor gates."

    op = npc.Array.from_ndarray_trivial(
        U2.reshape(2, 2, 2, 2),
        labels=['p0', 'p1', 'p0*', 'p1*']
    )

    psi.apply_local_op(i, op, unitary=True, renormalize=False, cutoff=svd_cut)
    return psi

def init_sites(L, conserve=None):
    # conserve=None is easiest (no quantum numbers).
    site = SpinHalfSite(conserve=conserve)   # local dim = 2
    #sites = [site] * (L + 1)                 # include R at index 0
    sites = [SpinHalfSite(conserve=conserve) for _ in range(L + 1)]
    return sites

def init_product_mps(sites, state0="up"):
    # SpinHalfSite basis uses "up"/"down" typically
    prod_state = [state0] * len(sites)
    psi = MPS.from_product_state(sites, prod_state, bc='finite')
    return psi

def H_gate():
    return (1/np.sqrt(2))*np.array([[1, 1],
                                   [1,-1]], dtype=complex)

def CNOT_gate():
    # |00>->|00|, |01>->|01|, |10>->|11|, |11>->|10|
    U = np.zeros((4,4), dtype=complex)
    U[0,0]=1; U[1,1]=1; U[2,3]=1; U[3,2]=1
    return U

def apply_two_site_unitary_old(psi, i, j, U2, chi_max, svd_cut):
    """
    Apply a 2-site unitary U2 (4x4) to sites (i,j) of the MPS psi.

    Implements the TEBD two-site update:
        merge -> apply gate -> SVD -> truncate -> write back
    """

    assert j == i + 1, "This routine expects nearest-neighbor gates."

    # bring orthogonality center to bond i
    psi.position(i)

    # get the two-site tensor theta
    # shape: (vL, s_i, s_j, vR)
    theta = psi.get_theta(i, n=2)

    # convert gate to npc tensor
    U = npc.Array.from_ndarray_trivial(
        U2.reshape(2, 2, 2, 2),
        labels=['p0', 'p1', 'p0*', 'p1*']
    )

    # apply gate on physical indices
    theta = npc.tensordot(U, theta, axes=(['p0*','p1*'], ['p0','p1']))

    # reorder indices back
    theta = theta.transpose(['vL','p0','p1','vR'])

    # reshape for SVD
    vL, s0, s1, vR = theta.shape
    theta_mat = theta.reshape((vL * s0, s1 * vR))

    # SVD
    Umat, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

    # truncation
    chi = min(len(S), chi_max)
    mask = S > svd_cut
    chi = min(chi, np.sum(mask))

    Umat = Umat[:, :chi]
    S = S[:chi]
    Vh = Vh[:chi, :]

    # reshape back
    A = Umat.reshape(vL, s0, chi)
    B = (np.diag(S) @ Vh).reshape(chi, s1, vR)

    # convert to npc tensors
    A = npc.Array.from_ndarray_trivial(A, labels=['vL','p','vR'])
    B = npc.Array.from_ndarray_trivial(B, labels=['vL','p','vR'])

    # write back to MPS
    psi.set_B(i, A)
    psi.set_B(j, B)



def T_gate():
    return np.array([[1, 0],
                     [0, np.exp(1j*np.pi/4)]], dtype=complex)

def CZ_gate():
    return np.diag([1, 1, 1, -1]).astype(complex)

def kron2(A,B): return np.kron(A,B)

def random_single_qubit_clifford():
    # minimal practical set (not the full 24, but decent):
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], complex)
    Z = np.array([[1,0],[0,-1]], complex)
    H = H_gate()
    S = np.array([[1,0],[0,1j]], complex)
    pool = [I, H, S, H@S, S@H, H@S@H, X, Z]
    return pool[np.random.randint(len(pool))]

def sample_two_qubit_gate(p_magic):
    u = np.random.rand()
    if u > p_magic:
        # Clifford-ish
        u1 = random_single_qubit_clifford()
        u2 = random_single_qubit_clifford()
        v1 = random_single_qubit_clifford()
        v2 = random_single_qubit_clifford()
        U = kron2(v1,v2) @ CZ_gate() @ kron2(u1,u2)
    else:
        # inject non-Clifford "magic"
        u1 = random_single_qubit_clifford()
        u2 = random_single_qubit_clifford()
        V  = CZ_gate() @ kron2(T_gate(), np.eye(2))   # one-sided T injection
        U = V @ kron2(u1,u2)
    return U

def floquet_period_layers(L_chain, p_magic):
    """
    returns layers = [layer_even, layer_odd]
    each layer is list of (i,j,U2) in *global* indexing with R at 0.
    chain sites are 1..L_chain
    """
    layers = []

    # even bonds: (1,2),(3,4),...
    layer_even = []
    for i in range(1, L_chain, 2):
        layer_even.append((i, i+1, sample_two_qubit_gate(p_magic)))
    layers.append(layer_even)

    # odd bonds: (2,3),(4,5),...
    layer_odd = []
    for i in range(2, L_chain, 2):
        layer_odd.append((i, i+1, sample_two_qubit_gate(p_magic)))
    layers.append(layer_odd)

    return layers

def apply_floquet_period(psi, layers, chi_max=200, svd_cut=1e-10):
    for layer in layers:
        for (i,j,U2) in layer:
            apply_two_site_unitary(psi, i, j, U2, chi_max, svd_cut)
        psi.canonical_form()  # optional stabilization
    return psi

def cut_entropies(psi):
    # returns S across each bond; TeNPy has psi.entanglement_entropy()
    return psi.entanglement_entropy()

def von_neumann_entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = np.clip(w, 0, 1)
    w = w[w > 1e-15]
    return float(-np.sum(w*np.log(w)))

def rho_segment_numpy(psi, i0, i1):
    """
    contiguous region [i0, i1)
    returns a standard 2^n × 2^n numpy density matrix
    """
    rho = psi.get_rho_segment(list(range(i0, i1)))  # npc.Array

    rho = rho.to_ndarray()

    n = i1 - i0
    dim = 2**n

    rho = rho.reshape(dim, dim)

    return rho

def region_entropy(psi, i0, i1):
    rho = rho_segment_numpy(psi, i0, i1)
    return von_neumann_entropy(rho)

def I3_contiguous(psi, A, B, C):
    # A,B,C are (i0,i1) pairs, assumed disjoint and contiguous.
    SA   = region_entropy(psi, *A)
    SB   = region_entropy(psi, *B)
    SC   = region_entropy(psi, *C)
    SAB  = region_entropy(psi, A[0], B[1])  # only if A..B contiguous with no gaps
    SAC  = region_entropy(psi, A[0], C[1])
    SBC  = region_entropy(psi, B[0], C[1])
    SABC = region_entropy(psi, A[0], C[1])

    return SA + SB + SC - SAB - SAC - SBC + SABC

def paulis_1q():
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], complex)
    Y = np.array([[0,-1j],[1j,0]], complex)
    Z = np.array([[1,0],[0,-1]], complex)
    return [I,X,Y,Z]

def kron_all(ops):
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out

def pauli_l1_mana(rho):
    # rho is 2^n x 2^n
    dim = rho.shape[0]
    n = int(np.log2(dim))

    P1 = paulis_1q()
    coeffs_abs_sum = 0.0

    # iterate over 4^n Paulis (OK for n<=6ish)
    # index in base-4 to pick I/X/Y/Z per site
    for k in range(4**n):
        ops = []
        x = k
        for _ in range(n):
            ops.append(P1[x % 4])
            x //= 4
        P = kron_all(ops)
        c = np.trace(rho @ P)  # c_P
        coeffs_abs_sum += abs(c)

    return float(np.log(coeffs_abs_sum + 1e-16))

def mana_block(psi, i0, i1):
    rho = rho_segment_numpy(psi, i0, i1)
    return pauli_l1_mana(rho)

def bell_phi_plus():
    # |Φ+> = (|00>+|11>)/sqrt(2)
    v = np.zeros(4, complex)
    v[0] = 1/np.sqrt(2)
    v[3] = 1/np.sqrt(2)
    rho = np.outer(v, v.conj())
    return rho

def partial_trace_keep_first_two_qubits(rho_RB, nB):
    """
    Keep qubits: R and b1 (first qubit of B).
    Trace out the remaining (nB-1) qubits of B.
    """
    # rho dims: (2*2^nB, 2*2^nB)
    dimB = 2**nB
    dim = 2*dimB

    # reshape into indices: R,B  x  R,B
    rho = rho_RB.reshape(2, dimB, 2, dimB)

    # now split B = b1 * brest
    rho = rho.reshape(2, 2, 2**(nB-1), 2, 2, 2**(nB-1))

    # trace over brest
    out = np.zeros((4,4), complex)
    for k in range(2**(nB-1)):
        out += rho[:,:,k,:,:,k].reshape(4,4)
    return out

def apply_decoder_on_B(rho_RB, V_B):
    """
    rho_RB -> (I_R ⊗ V_B) rho (I_R ⊗ V_B)†
    """
    dimB = V_B.shape[0]
    I2 = np.eye(2, dtype=complex)
    U = np.kron(I2, V_B)
    return U @ rho_RB @ U.conj().T

def teleport_fidelity_from_rhoRB(rho_RB, nB, V_B=None):
    if V_B is not None:
        rho_RB = apply_decoder_on_B(rho_RB, V_B)

    rho_Rb1 = partial_trace_keep_first_two_qubits(rho_RB, nB)
    return float(np.real(np.trace(rho_Rb1 @ bell_phi_plus())))

def apply_two_qubit_to_full(U_full, U2, i, j, n):
    """
    Embed a 2-qubit gate U2 into an n-qubit unitary acting on qubits (i,j).
    """
    I = np.eye(2)

    ops = []
    for k in range(n):
        if k == i:
            ops.append(None)
        elif k == j:
            ops.append(None)
        else:
            ops.append(I)

    # build kron but insert U2
    U = None
    k = 0
    while k < n:
        if k == i:
            op = U2
            k += 2
        else:
            op = ops[k]
            k += 1

        if U is None:
            U = op
        else:
            U = np.kron(U, op)

    return U @ U_full

def candidate_decoders_B(nB):
    # depth-1 CZ layer + local single-qubit Cliffords
    decs = []
    for _ in range(200):  # sample 200 random decoders
        V = np.eye(2**nB, dtype=complex)
        # build random local cliffords
        locals_ = [random_single_qubit_clifford() for _ in range(nB)]
        V = kron_all(locals_) @ V
        # entangle neighboring pairs with CZ
        for i in range(nB-1):
            V = apply_two_qubit_to_full(V, CZ_gate(), i, i+1, nB)  # implement by kron/permutation
        decs.append(V)
    return decs

def best_teleport_fidelity(rho_RB, nB):
    best = 0.0
    for V in candidate_decoders_B(nB):
        F = teleport_fidelity_from_rhoRB(rho_RB, nB, V_B=V)
        best = max(best, F)
    return best


def run_one(L_chain=40, A_site=1, p_magic=0.2, T_periods=20,
            chi_max=256, svd_cut=1e-10,
            mana_region=(10, 14),  # [i0,i1) contiguous
            B_block=(1, 7),       # output block near left for easy rho_RB
            ABC=None):

    sites = init_sites(L_chain, conserve=None)
    psi = init_product_mps(sites, state0="up")

    inject_bell_pair(psi, A_site=A_site, chi_max=chi_max, svd_cut=svd_cut)

    layers = floquet_period_layers(L_chain, p_magic=p_magic)

    data = []
    for t in range(T_periods+1):
        # observables
        obs = {}
        obs["t"] = t
        obs["S_cuts"] = psi.entanglement_entropy()
        obs["mana_block"] = mana_block(psi, *mana_region)  # uses rho_segment + pauli_l1_mana

        # teleport benchmark using rho_RB from contiguous segment [0, B1)
        j0, j1 = B_block
        assert j0 == 1  # so segment [0, j1) is exactly R + B contiguous
        rho_RB = rho_segment_numpy(psi, 0, j1)
        nB = j1 - 1
        obs["teleport_F_best"] = best_teleport_fidelity(rho_RB, nB)

        # I3 on contiguous blocks if you choose them
        if ABC is not None:
            obs["I3"] = I3_contiguous(psi, *ABC)

        data.append(obs)

        # evolve one period
        if t < T_periods:
            apply_floquet_period(psi, layers, chi_max=chi_max, svd_cut=svd_cut)

    return data

data = run_one(L_chain=40, A_site=1, p_magic=0.2, T_periods=20,
            chi_max=256, svd_cut=1e-10,
            mana_region=(10, 14),  # [i0,i1) contiguous
            B_block=(1, 7),       # output block near left for easy rho_RB
            ABC=None)
T_periods  = 20
plt.plot(np.arange(T_periods+1),[ele["teleport_F_best"] for ele in data])
plt.show()


print("done")