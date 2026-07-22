import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, sqrtm, block_diag, eigh, expm, sqrtm, schur, det



def symplectic_form(n):
    """Returns the 2n × 2n symplectic form Omega"""
    return np.block([
        [np.zeros((n, n),dtype=np.float64), np.eye(n,dtype=np.float64)],
        [-np.eye(n,dtype=np.float64), np.zeros((n, n),dtype=np.float64)]
    ])

def heisenberg_evolution_operator(H, t, n):
    Omega = symplectic_form(n)
    return expm(Omega @ H * t)

def operator_spread_over_time(H, t_list, op_index=0):
    """
    Computes the Heisenberg evolution of operator r_op_index over time.
    
    Returns:
        coeffs_t: list of arrays of coefficients at each time
    """
    n = H.shape[0] // 2  # number of modes
    coeffs_t = []

    for t in t_list:
        S_t = heisenberg_evolution_operator(H, t, n)
        r0 = np.zeros(2 * n)
        r0[op_index] = 1.0  # evolve x_{op_index}(t)

        evolved = S_t @ r0
        coeffs_t.append(evolved)

    return np.array(coeffs_t)  # shape: (len(t_list), 2n)

def plot_light_cone(coeffs_t, title="Operator Spread"):
    """
    coeffs_t: (T x 2n) array
    """
    T, dim = coeffs_t.shape
    n = dim // 2

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # |x_i| coefficients over time
    im1 = axs[0].imshow(np.abs(coeffs_t[:, :n]), aspect='auto', cmap='inferno', origin='lower')
    axs[0].set_ylabel('Time step')
    axs[0].set_title('Contribution to x_i')

    # |p_i| coefficients over time
    im2 = axs[1].imshow(np.abs(coeffs_t[:, n:]), aspect='auto', cmap='inferno', origin='lower')
    axs[1].set_ylabel('Time step')
    axs[1].set_xlabel('Mode index')
    axs[1].set_title('Contribution to p_i')

    fig.colorbar(im1, ax=axs[0], orientation='vertical', label='|Coefficient|')
    fig.colorbar(im2, ax=axs[1], orientation='vertical', label='|Coefficient|')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()



r = 1
n = 10

cmat = np.cosh(2*r)*np.eye(n)
smat = np.sinh(2*r)*np.eye(n)
mat0 = np.zeros((n,n))

Gamma_TFD = np.block([[cmat,smat,mat0,mat0],
                   [smat,cmat,mat0,mat0],
                   [mat0,mat0,cmat,-smat],
                   [mat0,mat0,-smat,cmat]])

cothmat = np.log(1/np.tanh(r))*np.eye(n)

KL = np.block([[cothmat, mat0],
               [mat0, cothmat]])
KL_full = np.block([[cothmat,mat0,mat0,mat0],
                   [mat0,mat0,mat0,mat0],
                   [mat0,mat0,cothmat,mat0],
                   [mat0,mat0,mat0,mat0]])

omega = symplectic_form(n)
#wrap around?

#t = 10
#S = (omega @ KL_full @ t)



###########
# investigate spreading
###########

t0 = 30

t_list = np.linspace(0, t0, 100)  # 100 time steps from t=0 to t=10
coeffs_t = operator_spread_over_time(KL, t_list, op_index=0)  # evolve x_0(t)
plot_light_cone(coeffs_t, title="Light Cone of $x_0(t)$")


print("done")


