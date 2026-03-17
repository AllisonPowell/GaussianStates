import numpy as np

def _coth(x):
    # stable-ish coth for moderate x
    return 1.0 / np.tanh(x)

def symplectic_form(n):
    return np.block([[np.zeros((n, n)), np.eye(n)],
                     [-np.eye(n), np.zeros((n, n))]])

def symplectic_eigs(Gamma):
    """Symplectic eigenvalues of a 2n×2n covariance matrix in xxpp ordering."""
    Gamma = 0.5 * (Gamma + Gamma.T)
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    vals = np.linalg.eigvals(1j * Omega @ Gamma)
    nu = np.sort(np.abs(vals))[::2]
    return nu.real

def build_ring_potential(N, k, m2):
    """V in H = 1/2 p^T p + 1/2 x^T V x for a periodic ring."""
    V = np.zeros((N, N), dtype=float)
    for i in range(N):
        V[i, i] = m2 + 2.0 * k
        V[i, (i + 1) % N] = -k
        V[i, (i - 1) % N] = -k
    return 0.5 * (V + V.T)

def thermal_cov_one_side_from_modes(O, omega, beta):
    """
    One-side thermal covariance (2N×2N) in xxpp ordering [x1..xN, p1..pN].
    """
    N = len(omega)
    nu = 0.5 * _coth(0.5 * beta * omega)          # symplectic spectrum of each normal mode
    var_x = nu / omega                             # <x^2>
    var_p = nu * omega                             # <p^2>

    Gamma_xx = O @ np.diag(var_x) @ O.T
    Gamma_pp = O @ np.diag(var_p) @ O.T
    Gamma = np.block([[Gamma_xx, np.zeros((N, N))],
                      [np.zeros((N, N)), Gamma_pp]])
    return 0.5 * (Gamma + Gamma.T), nu

def tfd_cov_ring_from_normal_modes(N, k, m2, beta, return_debug=False, eps_omega=1e-15):
    """
    Construct the *pure* TFD covariance matrix for the ring Hamiltonian
        H = 1/2 p^T p + 1/2 x^T V x
    at inverse temperature beta, using the normal-mode diagonalization of V.

    Output ordering (4N×4N) is:
        [x_L(1..N), x_R(1..N), p_L(1..N), p_R(1..N)]   (xxpp with LR split)

    This construction is an *analytic* Gaussian purification mode-by-mode, so in exact arithmetic
    symplectic eigenvalues of the 4N-mode state are exactly 0.5.
    """
    V = build_ring_potential(N, k, m2)

    # V = O diag(omega^2) O^T
    omega2, O = np.linalg.eigh(V)
    omega2 = np.clip(omega2, eps_omega, None)
    omega = np.sqrt(omega2)

    # Thermal invariants per normal mode
    nu = 0.5 * _coth(0.5 * beta * omega)                 # >= 0.5
    alpha = np.sqrt(np.maximum(nu * nu - 0.25, 0.0))     # correlations for purification

    # In normal-mode basis, build 4N×4N covariance for TFD:
    # blocks in xxpp with LR split:
    #   xx: [ diag(nu/ω)     diag(alpha/ω)
    #         diag(alpha/ω)  diag(nu/ω)     ]
    #
    #   pp: [ diag(nu*ω)     diag(-alpha*ω)
    #         diag(-alpha*ω) diag(nu*ω)     ]
    #
    #   xp = px = 0
    Dx  = np.diag(nu / omega)
    Dp  = np.diag(nu * omega)
    Cx  = np.diag(alpha / omega)
    Cp  = np.diag(-alpha * omega)

    xx_nm = np.block([[Dx, Cx],
                      [Cx, Dx]])
    pp_nm = np.block([[Dp, Cp],
                      [Cp, Dp]])

    Gamma_nm = np.block([[xx_nm, np.zeros((2*N, 2*N))],
                         [np.zeros((2*N, 2*N)), pp_nm]])

    # Transform back to site basis on BOTH L and R, for x and p:
    # x_L = O x'_L, x_R = O x'_R, p_L = O p'_L, p_R = O p'_R
    O2 = np.block([[O, np.zeros((N, N))],
                   [np.zeros((N, N)), O]])   # acts on (L,R) index within x-block or p-block
    S = np.block([[O2, np.zeros((2*N, 2*N))],
                  [np.zeros((2*N, 2*N)), O2]])

    Gamma_site = S @ Gamma_nm @ S.T
    Gamma_site = 0.5 * (Gamma_site + Gamma_site.T)

    if not return_debug:
        return Gamma_site

    # Also return the one-side thermal covariance (useful sanity check)
    Gamma_th, nu_check = thermal_cov_one_side_from_modes(O, omega, beta)

    # purity check
    nu_tfd = symplectic_eigs(Gamma_site)   # should be ~0.5 for all 2N modes
    return {
        "Gamma_TFD": Gamma_site,
        "Gamma_th_one_side": Gamma_th,
        "O": O,
        "omega": omega,
        "nu_thermal_per_mode": nu_check,
        "symplectic_eigs_TFD": nu_tfd,
        "min_symplectic_TFD": float(np.min(nu_tfd)),
        "max_symplectic_TFD": float(np.max(nu_tfd)),
    }

# -------------------------
# Example usage:
# -------------------------

N = 3
k = 5.0
m2 = 1
beta = 1.0

out = tfd_cov_ring_from_normal_modes(N, k, m2, beta, return_debug=True)
print("omega:", out["omega"])
print("TFD symplectic eigs range:",
        out["min_symplectic_TFD"], out["max_symplectic_TFD"])
print("stop")