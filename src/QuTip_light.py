import numpy as np
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
)
import numpy as np
from qutip import Qobj, sesolve

def vec_from_C(C):
    # column-stacking convention consistent with kron identities
    return C.reshape((-1,), order="F")

def C_from_vec(v, D):
    return np.array(v).reshape((D, D), order="F")


def apply_Hint_step_unitary(C, U_int, D_side, N_modes, N_cutoff):
    v = vec_from_C(C)
    N_total = 2*N_modes
    psi = Qobj(v, dims=[[N_cutoff]*N_total, [1]*N_total])
    psi2 = U_int * psi
    v2 = psi2.full().ravel()
    return C_from_vec(v2, D_side)

def apply_Hint_step_sesolve(C, H_int, dt, N_modes, N_cutoff):
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
    return C_from_vec(v2, D_side)

def build_pair_hint_terms(N_modes, insert_idx, mu_x, mu_p, x_full, p_full):
    terms = []
    for i in range(N_modes):
        if i == insert_idx:
            continue
        Li = i
        Ri = i + N_modes
        Hi = mu_x * x_full[Li] * x_full[Ri] + mu_p * p_full[Li] * p_full[Ri]
        terms.append(Hi)
    return terms

def apply_Hint_commuting_terms(C, H_terms, dt, N_modes, N_cutoff):
    D_side = N_cutoff**N_modes
    v = vec_from_C(C)
    N_total = 2*N_modes
    psi = Qobj(v, dims=[[N_cutoff]*N_total, [1]*N_total])

    for Hi in H_terms:  # commuting: any order
        out = sesolve(Hi, psi, [0, dt], progress_bar=None)
        psi = out.states[-1]

    v2 = psi.full().ravel()
    return C_from_vec(v2, D_side)

def strang_coupling(C, H0_side, H_int, t_couple, n_steps, N_modes, N_cutoff,H_terms):
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
        #C = apply_Hint_commuting_terms(C, H_terms, dt, N_modes, N_cutoff)
        # half local again
        C = apply_left(C, U0_half)
        C = apply_right(C, U0_half)

    return C




def build_TFD_C(evals, evecs, beta, keep_states=None):
    evals = np.asarray(evals, float)
    # stable weights (optional but recommended)
    evals0 = evals - evals.min()
    w = np.exp(-beta * evals0)

    idx = np.argsort(evals)  # low energy first
    if keep_states is not None:
        idx = idx[:keep_states]

    w_kept = w[idx]
    w_kept = w_kept / w_kept.sum()  # <-- renormalize within truncation

    D = evecs[0].shape[0]
    C = np.zeros((D, D), dtype=complex)

    for a, n in enumerate(idx):
        v = evecs[n].full().ravel()
        C += np.sqrt(w_kept[a]) * np.outer(v, v)  # no conjugate here

    return C


def build_TFD_C_fast(evals, evecs, beta, keep_states=None):
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

def apply_full_unitary_to_C(C, U_full, D):
    # If you already have U_full = expm(-i H_full t)
    v = vec_from_C(C)
    v2 = U_full @ v
    return C_from_vec(v2, D)

def apply_coupling_sesolve(C, H_int_full, t_couple, D, dims_full=None):
    v = vec_from_C(C)
    if dims_full is None:
        # bipartite dims
        dims_full = [[D, D], [1]]
    psi = Qobj(v, dims=dims_full)
    out = sesolve(H_int_full, psi, [0, t_couple])
    v2 = out.states[-1].full().ravel()
    return C_from_vec(v2, D)



def build_injection_unitary_side(N_modes, N_cutoff, insert_idx, alpha, r, phi_sq, phi_rot):
    a = destroy(N_cutoff)
    I = qeye(N_cutoff)

    # single-mode ops
    S = squeeze(N_cutoff, r * np.exp(1j*phi_sq))
    D = displace(N_cutoff, alpha)
    R = (1j * phi_rot * a.dag() * a).expm()
    U_local = D * S * R

    # wrap to side space
    ops = [I]*N_modes
    ops[insert_idx] = U_local
    return tensor(ops)

# 1. Configuration
N_modes = 3  # Modes per side (Ring: 0-1, 1-2, 2-0)
N_cutoff = 5  # Lowered for N=3 to keep memory usage safe
beta = 1  # Inverse temperature (Tuning this is critical)
lam = 0.1
chi = 0.15  # Non-Gaussianity (x^3)
g_nn = 1  # Ring coupling strength
g_int = 0.2  # L-R coupling strength
t_scramble = 1.5  # Scrambling time (tune this for peak fidelity)
t_couple = 1
keep_states = 60
insert_idx = 1
k = 5
m_squared = 13
omega0 = np.sqrt(m_squared + 2 * k)
mu_x = omega0 / 2
mu_p = 1 / (2 * omega0)

# 2. Operators
a = destroy(N_cutoff)
I = qeye(N_cutoff)
x_loc = (a + a.dag()) / np.sqrt(2)
p_loc = (a - a.dag()) / (1j * np.sqrt(2))


def wrap(local_op, mode, N_modes, I):
    ops = [I] * (N_modes)
    ops[mode] = local_op
    return tensor(ops)


# build x_i and p_i for all modes
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




# 5. The Protocol

# --- Parameters for the Input State ---
# Step 1: Evolve Left BACKWARD, then Forward
print("left back")

U_back_side = ( 1j * H_scramble_side * t_scramble).expm()
U_back = U_back_side.full()
C = apply_left(C_tfd, U_back)


# Step 2: Inject info
print("inject")
U_inject_side = build_injection_unitary_side(
    N_modes, N_cutoff, insert_idx,
    alpha=0.0, r=0.5, phi_sq=np.pi, phi_rot=np.pi/4
).full()
C = apply_left(C, U_inject_side)

# Step 3: Evolve Left FORWARD (Scrambling)

print("left forwards")
U_fwd_side  = (-1j * H_scramble_side * t_scramble).expm()
U_fwd = U_fwd_side.full()
C = apply_left(C, U_fwd)

# Step D: Apply the "Wormhole" Coupling (L-R Interaction)
# V = exp(i * phi * sum(x_L * x_R))

print("couple")

def wrap_full(local_op, mode, N_total_modes, I):
    ops = [I] * N_total_modes
    ops[mode] = local_op
    return tensor(ops)


N_total = 2 * N_modes

x_full = [wrap_full(x_loc, i, N_total, I) for i in range(N_total)]
p_full = [wrap_full(p_loc, i, N_total, I) for i in range(N_total)]

H_int = 0
for i in range(N_modes):
    if i == insert_idx:
        continue
    Li = i
    Ri = i + N_modes
    H_int += mu_x * x_full[Li] * x_full[Ri]
    H_int += mu_p * p_full[Li] * p_full[Ri]

D_side = N_cutoff**N_modes
H_terms = build_pair_hint_terms(N_modes, insert_idx, mu_x, mu_p, x_full, p_full)

C = strang_coupling(C, H0_side, H_int, t_couple=t_couple, n_steps=50, N_modes=N_modes,N_cutoff=N_cutoff,H_terms=H_terms)

# Step E: Evolve Right FORWARD
print("right forward")
C = apply_right(C, U_fwd)

# 5. Verification: Check Fidelity
# Compare the output on the last mode of the Right side to the input
# rho_in_target = squeeze(N_cutoff, r_sq) * basis(N_cutoff, 0)
# rho_out_actual = psi_final.ptrace(2 * N_modes - 1) # Last mode of Right

fidelity = expect(rho_in_target.proj(), rho_out_actual)
print(f"Teleportation Fidelity: {fidelity:.4f}")
