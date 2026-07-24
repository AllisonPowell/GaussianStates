import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, block_diag

def symplectic_form(n):
    """Returns the 2n × 2n symplectic form Omega."""
    return np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])

def generate_interacting_tfd(n, omega_0, J, beta, periodic):
    """
    Generates the coupled modular Hamiltonian and covariance matrix
    for a continuous-variable tight-binding chain.
    """
    # 1. Construct
    # the spatial hopping matrix (Hamiltonian h)
    h = omega_0 * np.eye(n)
    for i in range(n - 1):
        h[i, i+1] = -J
        h[i+1, i] = -J
    if periodic == True:
            h[0,n-1] = -J
            h[n-1,0] = -J
        
    # 2. Diagonalize to find collective normal modes
    eigenvalues, V = np.linalg.eigh(h)

    # 3. Calculate mode-dependent squeezing parameters from temperature
    # cosh(2r) = coth(beta * omega / 2)
    cosh_r = np.zeros(n)
    sinh_r = np.zeros(n)
    lambda_diagonal = np.zeros(n)

    for i, omega_i in enumerate(eigenvalues):
        # Prevent division by zero for unphysical modes
        omega_i = max(omega_i, 1e-5)
        # Physical relation: tanh(r) = exp(-beta * omega / 2)
        tanh_ri = np.exp(-beta * omega_i / 2)
        # Avoid pure saturation limits
        tanh_ri = min(tanh_ri, 0.999) 
        
        # Recover squeezing parameter r
        r_i = np.arctanh(tanh_ri)
        
        cosh_r[i] = np.cosh(2 * r_i)
        sinh_r[i] = np.sinh(2 * r_i)
        lambda_diagonal[i] = np.log(1.0 / tanh_ri)

    # 4. Assemble matrices in the normal-mode basis
    C_mat = np.diag(cosh_r)
    S_mat = np.diag(sinh_r)
    zeros_n = np.zeros((n, n))

    Gamma_NM = np.block([
        [C_mat, S_mat, zeros_n, zeros_n],
        [S_mat, C_mat, zeros_n, zeros_n],
        [zeros_n, zeros_n, C_mat, -S_mat],
        [zeros_n, zeros_n, -S_mat, C_mat]
    ])

    KL_NM_block = np.diag(lambda_diagonal)
    KL_NM = np.block([
        [KL_NM_block, zeros_n],
        [zeros_n, KL_NM_block]
    ])

    # 5. Transform back to the physical local spatial basis using V
    V_4n = block_diag(V, V, V, V)
    V_2n = block_diag(V, V)

    Gamma_physical = V_4n @ Gamma_NM @ V_4n.T
    KL_physical = V_2n @ KL_NM @ V_2n.T

    return Gamma_physical, KL_physical



def simulate_and_plot_light_cone(KL_spatial, n, max_time=15):
    """Evolves the first spatial position operator and plots the spread."""
    t_list = np.linspace(0, max_time, 200)
    Omega = symplectic_form(n)
    
    coeffs_x = []
    coeffs_p = []
    
    # Track the evolution of x_0
    r0 = np.zeros(2 * n)
    r0[0] = 1.0 

    for t in t_list:
        S_t = expm(Omega @ KL_spatial * t)
        r_t = S_t @ r0
        coeffs_x.append(np.abs(r_t[:n]))
        coeffs_p.append(np.abs(r_t[n:]))

    # Plot the resulting light cone
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    im1 = axs[0].imshow(np.array(coeffs_x), aspect='auto', cmap='viridis', origin='lower', extent=[0, n-1, 0, max_time])
    axs[0].set_ylabel('Continuous Time (t)')
    axs[0].set_title('Spatial Operator Spreading: Contribution to $x_i(t)$')
    fig.colorbar(im1, ax=axs[0], label='Amplitude')

    im2 = axs[1].imshow(np.array(coeffs_p), aspect='auto', cmap='viridis', origin='lower', extent=[0, n-1, 0, max_time])
    axs[1].set_ylabel('Continuous Time (t)')
    axs[1].set_xlabel('Spatial Lattice Site Index')
    axs[1].set_title('Spatial Operator Spreading: Contribution to $p_i(t)$')
    fig.colorbar(im2, ax=axs[1], label='Amplitude')

    plt.tight_layout()
    plt.show()


def insert_two_mode_state(Gamma_system, insert_idx, Gamma_2mode):
    """
    Inserts a 2-mode state (inserted + observer) into Gamma_system by:
    - Removing the inserted mode from Gamma_system entirely
    - Performing a direct sum with Gamma_insert_2mode
    - Permuting quadratures so inserted mode goes to insert_idx,
      observer goes to the end.

    Parameters:
        Gamma_system: (2n x 2n) real symmetric covariance matrix
        insert_idx: index (0 <= i < n) of mode to be replaced
        Gamma_insert_2mode: (4 x 4) covariance matrix of [inserted, observer]

    Returns:
        Gamma_extended: (2n x 2n) covariance matrix with inserted + observer
    """
    assert Gamma_2mode.shape == (4, 4), "Gamma_insert_2mode must be 4×4"
    n = Gamma_system.shape[0] // 2
    assert Gamma_system.shape == (2*n, 2*n)


    # Permute the rows and columns
    Gamma_direct_sum = np.zeros((2*n+2,2*n+2))
    Gamma_direct_sum[0:2*n,0:2*n] = Gamma_system
    Gamma_permuted = Gamma_direct_sum.copy()
    Gamma_permuted[n+1:2*n+1,:]  = Gamma_direct_sum[n:2*n,:]
    Gamma_permuted[:,n+1:2*n+1] = Gamma_direct_sum[:,n:2*n]
    Gamma_permuted[n+1:2*n+1,n+1:2*n+1] = Gamma_system[n:2*n+1,n:2*n+1]
    Gamma_permuted[insert_idx,:]=0
    Gamma_permuted[:,insert_idx]=0
    Gamma_permuted[n,:]=0
    Gamma_permuted[:,n]=0
    Gamma_permuted[insert_idx+n+1,:]=0
    Gamma_permuted[:,insert_idx+n+1]=0
    Gamma_permuted[insert_idx,insert_idx] = Gamma_2mode[0,0]
    Gamma_permuted[insert_idx,n] = Gamma_2mode[0,1]
    Gamma_permuted[n,insert_idx] = Gamma_2mode[1,0]
    Gamma_permuted[n,n] = Gamma_2mode[1,1]
    Gamma_permuted[insert_idx+n+1,insert_idx+n+1]=Gamma_2mode[2,2]
    Gamma_permuted[insert_idx+n+1,2*n+1] = Gamma_2mode[2,3]
    Gamma_permuted[2*n+1,insert_idx+n+1] = Gamma_2mode[3,2]
    Gamma_permuted[2*n+1,2*n+1]=Gamma_2mode[3,3]

    return(.5*(Gamma_permuted+Gamma_permuted.T))

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

def make_boundary_coupling(n, insert_idx, g):

    coupling_sites_1 = np.arange(0,insert_idx)
    coupling_sites_2 = np.arange(insert_idx+1,n)
    coupling_sites= np.concatenate((coupling_sites_1,coupling_sites_2))


    N = 4*n
    G = np.zeros((N,N))

    for j in coupling_sites:

        # x_L x_R
        G[j, n+j] = g
        G[n+j, j] = g

        # p_L p_R
        G[2*n+j, 3*n+j] = g
        G[3*n+j, 2*n+j] = g

    return G
def extract_subsystem_covariance(Gamma, indices):
    indices = np.array(indices)
    x_idx = indices
    p_idx = indices + Gamma.shape[0] // 2
    full_idx = np.concatenate([x_idx, p_idx])
    return Gamma[np.ix_(full_idx, full_idx)]

def von_neumann_entropy(Gamma):
    n = Gamma.shape[0] // 2
    Omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])
    eigvals = np.linalg.eigvals(1j * Gamma @ Omega)
    nu = np.sort(np.abs(eigvals))[::2]
    nu = np.clip(nu, 0.500001, None)
    return sum((nu + 0.5)*np.log(nu + 0.5) - (nu - 0.5)*np.log(nu - 0.5))
def mutual_information(Gamma, idx_L, idx_R):
    S_L = von_neumann_entropy(extract_subsystem_covariance(Gamma, idx_L))
    S_R = von_neumann_entropy(extract_subsystem_covariance(Gamma, idx_R))
    S_LR = von_neumann_entropy(extract_subsystem_covariance(Gamma, idx_L + idx_R))
    return S_L + S_R - S_LR

def insert_unentangled_mode(Gamma, mode_index, Gamma_insert):
    """
    Replace a single mode (x_i, p_i) in the covariance matrix with a new unentangled mode.

    Parameters:
        Gamma : (2n x 2n) np.array
            Original covariance matrix (x_0, ..., x_{n-1}, p_0, ..., p_{n-1})
        mode_index : int
            The index of the mode (0 <= i < n) to replace
        Gamma_insert : (2x2) np.array (optional)
            Covariance matrix for the inserted mode. If None, defaults to vacuum state.

    Returns:
        Gamma_new : (2n x 2n) np.array
            New covariance matrix with the mode replaced
    """
    n = Gamma.shape[0] // 2
    assert Gamma.shape == (2*n, 2*n), "Gamma must be 2n x 2n"
    assert 0 <= mode_index < n, "Invalid mode index"

    # Default inserted mode: vacuum (ν = 0.5, identity block)
    if Gamma_insert is None:
        Gamma_insert = 0.5 * np.eye(2)

    # Identify row/column indices for mode i
    x_i = mode_index
    p_i = mode_index + n
    idx_remove = [x_i, p_i]

    # Create new Gamma by replacing x_i and p_i rows/cols
    Gamma_new = Gamma.copy()

    # Zero out off-diagonal coupling to/from x_i and p_i
    Gamma_new[idx_remove, :] = 0
    Gamma_new[:, idx_remove] = 0

    # Insert new 2x2 unentangled block
    Gamma_new[np.ix_(idx_remove, idx_remove)] = Gamma_insert

    return Gamma_new

def extract_mode_block(Gamma, mode_index):
    """
    Extract the 2×2 covariance matrix (x, p) block for one mode from full Gamma.
    Assumes Gamma is in (x0, ..., xn, p0, ..., pn) ordering.
    """
    n = Gamma.shape[0] // 2
    x_i = mode_index
    p_i = mode_index + n
    return Gamma[np.ix_([x_i, p_i], [x_i, p_i])]

# --- Execution Parameters ---
n = 5    # Size of the chain
omega_bare = 1.0   # On-site energy
hopping_J = 0.4    # Coupling strength between neighbors
inv_temp = 1     # Inverse temperature beta
t_evolve = 5
t_couple = 1.6
T = 40
dt = t_evolve/T
dt_couple = t_couple/T
insert_idx = 1
# Generate and simulate
Gamma_TFD, HL = generate_interacting_tfd(n, omega_bare, hopping_J, inv_temp,periodic=True)
simulate_and_plot_light_cone(HL, n,max_time=t_evolve)

HL_full = np.zeros((4*n, 4*n))
HL_full[np.ix_(range(n), range(n))] = HL[:n, :n]                     # x-x
HL_full[np.ix_(range(n), range(2*n, 3*n))] = HL[:n, n:]             # x-p
HL_full[np.ix_(range(2*n, 3*n), range(n))] = HL[n:, :n]             # p-x
HL_full[np.ix_(range(2*n, 3*n), range(2*n, 3*n))] = HL[n:, n:]      # p-p



# Symplectic form
Omega = symplectic_form(2*n)

# Evolve backward in time

S_back = expm(-1 * Omega @ HL_full * t_evolve)
Gamma_back = S_back @ Gamma_TFD @ S_back.T


###########
# insert quantum information on one side
###########



#teleported_idx = bdy_len + q # index 0 on right side starts here


n_total = Gamma_TFD.shape[0] // 2

theta = np.pi
s = 1


Rot = np.array([[np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
Squeeze = 0.5 * np.array([[np.exp(-2*s), 0],
                        [0, np.exp(2*s)]])

Gamma_rot_squeezed = Rot @ Squeeze @ Rot.T

Gamma_insert = insert_unentangled_mode(Gamma_back, insert_idx, Gamma_rot_squeezed)

Gamma_2mode = two_mode_squeezed_state(r=1)

Gamma_with_observer = insert_two_mode_state(Gamma_back, insert_idx, Gamma_2mode)

HL_full_padded = pad_matrix_for_observer(HL_full)

#######
# evolve forwards in time
#######
#S_forward_no_insert = expm(Omega @ HL_full * t_evolve)
#Gamma_forward = S_forward_no_insert @ Gamma_insert @ S_forward_no_insert.T

n_total = Gamma_with_observer.shape[0] // 2  # now 2n+1
observer_idx = 2*n

Omega_padded = symplectic_form(n_total)

I_obs_L = []
I_obs_R = []
I_insert = []

Gamma_LR_observer = Gamma_with_observer
Gamma_LR_wigner = Gamma_insert

S_forward_observer = expm(Omega_padded @ HL_full_padded * t_evolve)
Gamma_forward_observer = S_forward_observer @ Gamma_with_observer @ S_forward_observer.T

times_obs_forward = np.linspace(0,t_evolve,T)
S_forward_observer_dt = expm(Omega_padded @ HL_full_padded * dt)
S_forward_wigner_dt = expm(Omega @ HL_full * dt)

for t in enumerate(times_obs_forward):
    Gamma_LR_observer = S_forward_observer_dt @ Gamma_LR_observer @ S_forward_observer_dt.T
    #Gamma_LR_no_insert = S_forward_no_insert @ Gamma_LR_no_insert @ S_forward_no_insert.T
    Gamma_LR_wigner = S_forward_wigner_dt @ Gamma_LR_wigner @ S_forward_wigner_dt.T
    I_L = mutual_information(Gamma_LR_observer, [observer_idx], list(range(n)))
    I_R = mutual_information(Gamma_LR_observer, [observer_idx], list(range(n, 2*n)))
    I_obs_L.append(I_L)
    I_obs_R.append(I_R)
    I_insert.append(mutual_information(Gamma_LR_observer, [observer_idx], [insert_idx]))
    
#######
# couple the two sides
#######

H_coupling = make_boundary_coupling(n, insert_idx, g=1)


#S_coupling = expm(Omega @ H_coupling * t_couple)
#Gamma_coupled = S_coupling @ Gamma_forward @ S_coupling.T

H_coupling_padded = pad_matrix_for_observer(H_coupling)
S_coupling_observer = expm(Omega_padded @ H_coupling_padded * t_couple)
Gamma_coupled_observer = S_coupling_observer @ Gamma_forward_observer @ S_coupling_observer.T

times_obs_coupling = np.linspace(0,t_couple,T)
S_coupling_observer_dt = expm(Omega_padded @ H_coupling_padded * dt_couple)
S_coupling_wigner_dt = expm(Omega @ H_coupling * dt_couple)

for t in enumerate(times_obs_coupling):
    Gamma_LR_observer = S_coupling_observer_dt @ Gamma_LR_observer @ S_coupling_observer_dt.T
    #Gamma_LR_no_insert = S_forward_no_insert @ Gamma_LR_no_insert @ S_forward_no_insert.T
    Gamma_LR_wigner = S_coupling_wigner_dt @ Gamma_LR_wigner @ S_coupling_wigner_dt.T
    I_L = mutual_information(Gamma_LR_observer, [observer_idx], list(range(n)))
    I_R = mutual_information(Gamma_LR_observer, [observer_idx], list(range(n, 2*n)))
    I_obs_L.append(I_L)
    I_obs_R.append(I_R)
    I_insert.append(mutual_information(Gamma_LR_observer, [observer_idx], [insert_idx]))
    



######
# evolve state forwards in time with KR
######


HR_full = np.zeros((4*n, 4*n))
HR_full[np.ix_(range(n, 2*n), range(n, 2*n))] = HL[:n, :n]
HR_full[np.ix_(range(n, 2*n), range(3*n, 4*n))] = HL[:n, n:]
HR_full[np.ix_(range(3*n, 4*n), range(n, 2*n))] = HL[n:, :n]
HR_full[np.ix_(range(3*n, 4*n), range(3*n, 4*n))] = HL[n:, n:]

HR_full_padded = pad_matrix_for_observer(HR_full)



#S_final = expm(Omega @ HR_full * t_evolve)
#Gamma_final = S_final @ Gamma_coupled @ S_final.T

S_final_observer = expm(Omega_padded @ HR_full_padded * t_evolve)
Gamma_final_observer = S_final_observer @ Gamma_coupled_observer @ S_final_observer.T

times_obs_final = np.linspace(0,t_evolve,T)
S_final_observer_dt = expm(Omega_padded @ HR_full_padded * dt)
S_final_wigner_dt = expm(Omega @ HR_full * dt)


for t in enumerate(times_obs_forward):
    Gamma_LR_observer = S_final_observer_dt @ Gamma_LR_observer @ S_final_observer_dt.T
    #Gamma_LR_no_insert = S_forward_no_insert @ Gamma_LR_no_insert @ S_forward_no_insert.T
    Gamma_LR_wigner = S_final_wigner_dt @ Gamma_LR_wigner @ S_final_wigner_dt.T
    I_L = mutual_information(Gamma_LR_observer, [observer_idx], list(range(n)))
    I_R = mutual_information(Gamma_LR_observer, [observer_idx], list(range(n, 2*n)))
    I_obs_L.append(I_L)
    I_obs_R.append(I_R)
    I_insert.append(mutual_information(Gamma_LR_observer, [observer_idx], [insert_idx]))
    
times_obs_coupling = np.array(times_obs_coupling)
times_obs_coupling+=t_evolve

times_obs_final = np.array(times_obs_final)
times_obs_final+=(t_evolve+t_couple)


times = np.concatenate((times_obs_forward,times_obs_coupling,times_obs_final))
plt.plot(times,I_obs_L,"k",label = "mutual info with left")
plt.plot(times,I_obs_R,"r",label = "mutual info with right")
plt.plot(times,I_insert,"green",label = "mutual info with insert")
plt.legend()
plt.show()



Gamma_teleported = extract_mode_block(Gamma_LR_wigner, insert_idx+n)

print(0.5 * np.eye(2))
Gamma_out_real = 0.5 * (Gamma_teleported + Gamma_teleported.conj().T)
print(Gamma_out_real)


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_wigner_ellipse(Gamma_mode, ax, label='', color='blue'):
    from scipy.linalg import eigh
    W = Gamma_mode[:2, :2].real  # just x, p block
    vals, vecs = eigh(W)
    width, height = 2 * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                      edgecolor=color, fc='None', lw=2, label=label)
    ax.add_patch(ellipse)

fig, ax = plt.subplots()
#plot_wigner_ellipse(np.array([[0.5, 0], [0, 0.5]]), ax, label='Vacuum', color='blue')
plot_wigner_ellipse(Gamma_rot_squeezed, ax, label='Input', color='green')
plot_wigner_ellipse(Gamma_out_real, ax, label='Output', color='red')
#plot_wigner_ellipse(Gamma_out_shift, ax, label='No Input', color='orange')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel("Position Quadrature")
ax.set_ylabel("Momentum Quadrature")
ax.set_aspect('equal')
ax.legend()
plt.title("Input vs Output Wigner Ellipses")
plt.grid(True)
plt.show()


print("done")


