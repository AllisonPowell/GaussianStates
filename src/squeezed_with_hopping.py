import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, block_diag

def symplectic_form(n):
    """Returns the 2n × 2n symplectic form Omega."""
    return np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])

def generate_interacting_tfd(n, omega_0, J, beta):
    """
    Generates the coupled modular Hamiltonian and covariance matrix
    for a continuous-variable tight-binding chain.
    """
    # 1. Construct the spatial hopping matrix (Hamiltonian h)
    h = omega_0 * np.eye(n)
    for i in range(n - 1):
        h[i, i+1] = -J
        h[i+1, i] = -J

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



# --- Execution Parameters ---
num_modes = 15     # Size of the chain
omega_bare = 1.0   # On-site energy
hopping_J = 0.6    # Coupling strength between neighbors
inv_temp = 1     # Inverse temperature beta

# Generate and simulate
Gamma_TFD, HL = generate_interacting_tfd(num_modes, omega_bare, hopping_J, inv_temp)
simulate_and_plot_light_cone(HL, num_modes,max_time=30)

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
S_forward_no_insert = expm(Omega @ HL_full * t_evolve)
Gamma_forward = S_forward_no_insert @ Gamma_insert @ S_forward_no_insert.T

n_total = (Gamma_with_observer.shape[0]) // 2  # now 2n+1
Omega_padded = symplectic_form(n_total)
S_forward_observer = expm(Omega_padded @ HL_full_padded * t_evolve)
Gamma_forward_observer = S_forward_observer @ Gamma_with_observer @ S_forward_observer.T

#######
# couple the two sides
#######


S_coupling = expm(Omega @ H_coupling * t_couple)
Gamma_coupled = S_coupling @ Gamma_forward @ S_coupling.T

H_coupling_padded = pad_matrix_for_observer(H_coupling)
S_coupling_observer = expm(Omega_padded @ H_coupling_padded * t_couple)
Gamma_coupled_observer = S_coupling_observer @ Gamma_forward_observer @ S_coupling_observer.T


######
# evolve state forwards in time with KR
######


HR_full = np.zeros((4*n, 4*n))
HR_full[np.ix_(range(n, 2*n), range(n, 2*n))] = HL[:n, :n]
HR_full[np.ix_(range(n, 2*n), range(3*n, 4*n))] = HL[:n, n:]
HR_full[np.ix_(range(3*n, 4*n), range(n, 2*n))] = HL[n:, :n]
HR_full[np.ix_(range(3*n, 4*n), range(3*n, 4*n))] = HL[n:, n:]

HR_full_padded = pad_matrix_for_observer(HR_full)



S_final = expm(Omega @ HR_full * t_evolve)
Gamma_final = S_final @ Gamma_coupled @ S_final.T

S_final_observer = expm(Omega_padded @ HR_full_padded * t_evolve)
Gamma_final_observer = S_final_observer @ Gamma_coupled_observer @ S_final_observer.T





