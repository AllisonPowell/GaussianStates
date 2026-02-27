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
)

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
"""
# ring couplings: right
for i in range(N_modes):
    ii = i + N_modes
    jj = (i+1) % N_modes + N_modes
    H0 += 0.5*k*(x_ops[ii] - x_ops[jj])**2
"""
Hng_side = 0
for i in range(N_modes):  # say: only scramble left
    # Hng += chi * x_ops[i]**3
    Hng_side += lam * x_ops[i] ** 4


# 4. Generate the TFD State
# build TFD from H0_side eigenstates (correct)
evals, evecs = H0_side.eigenstates(eigvals=keep_states)
evals = np.array(evals, float)
w = np.exp(-beta * evals)
w /= w.sum()

psi_tfd = 0
for n in range(len(evals)):
    psi_tfd += np.sqrt(w[n]) * tensor(evecs[n], evecs[n])
psi_tfd = psi_tfd.unit()


# I_side = qeye(N_cutoff**N_modes)
I1 = qeye(N_cutoff)
I_side = tensor([I1] * N_modes)  # dims [[8,8,8],[8,8,8]]

# lift to full L⊗R system
H_L_full = tensor(H0_side, I_side)
H_R_full = tensor(I_side, H0_side)
Hng_L = tensor(Hng_side, I_side)
Hng_R = tensor(I_side, Hng_side)
H_LR_full = H_L_full + H_R_full
H_L_scramble = H_L_full + Hng_L
H_R_scramble = H_R_full + Hng_R

# 5. The Protocol


# --- Parameters for the Input State ---
alpha = 0.0  # Displacement (Set to 0 for mean=0)
r = 0.5  # Squeezing magnitude (r > 0)
phi_sq = np.pi
phi_rot = np.pi / 4  # Squeezing/Rotation angle

# 1. Define the Gaussian Transformation Operators
# We apply these only to Mode 0 of the Left side
S = squeeze(N_cutoff, r * np.exp(1j * phi_sq))  # Squeezing
D = displace(N_cutoff, alpha)  # Displacement
R = (1j * phi_rot * a.dag() * a).expm()  # Rotation

# 2. Construct the Total Injection Operator
# We wrap the local operator to act on the full 2N-mode Hilbert space

# Combine them: Rotate -> Squeeze -> Displace
U_inject = tensor(wrap(D * S * R, insert_idx, N_modes, I), I_side)

# Step 1: Evolve Left BACKWARD, then Forward
# In QuTiP, backward evolution is just mesolve with -H
tlist = np.linspace(0, t_scramble, 100)
res_back = sesolve(-H_L_scramble, psi_tfd, tlist)
psi_back = res_back.states[-1]

# Step 2: Inject info
psi_in = U_inject * psi_back

# Step 3: Evolve Left FORWARD (Scrambling)
res_fwd = sesolve(H_L_scramble, psi_in, tlist)
psi_fwd = res_fwd.states[-1]

# Step D: Apply the "Wormhole" Coupling (L-R Interaction)
# V = exp(i * phi * sum(x_L * x_R))


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

tlist_couple = np.linspace(0, t_couple, 50)

V_int = H_int + H_LR_full
res_coupled = sesolve(V_int, psi_fwd, tlist_couple)
psi_coupled = res_coupled.states[-1]

# Step E: Evolve Right FORWARD
res_final = sesolve(H_R_scramble, psi_coupled, tlist)
psi_out = res_final.states[-1]

# 5. Verification: Check Fidelity
# Compare the output on the last mode of the Right side to the input
# rho_in_target = squeeze(N_cutoff, r_sq) * basis(N_cutoff, 0)
# rho_out_actual = psi_final.ptrace(2 * N_modes - 1) # Last mode of Right

fidelity = expect(rho_in_target.proj(), rho_out_actual)
print(f"Teleportation Fidelity: {fidelity:.4f}")
