import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, expm, sqrtm, schur, block_diag, eigh, det, polar
from thewalrus.symplectic import xpxp_to_xxpp, sympmat
import random

def symplectic_form(n):
    """Returns the 2n × 2n symplectic form Omega"""
    return np.block([
        [np.zeros((n, n),dtype=np.float64), np.eye(n,dtype=np.float64)],
        [-np.eye(n,dtype=np.float64), np.zeros((n, n),dtype=np.float64)]
    ])


def extract_subsystem_covariance(Gamma, indices):
    indices = np.array(indices)
    x_idx = indices
    p_idx = indices + Gamma.shape[0] // 2
    full_idx = np.concatenate([x_idx, p_idx])
    return Gamma[np.ix_(full_idx, full_idx)]

def trace_out_subsystem(Gamma, keep_indices):
    """
    Return the reduced covariance matrix for a Gaussian state
    by keeping only modes in keep_indices (x and p interleaved).

    keep_indices: list or array of mode indices to keep (0 to n-1)
    Assumes Gamma is in the (x_0,...x_n, p_0,...p_n) basis
    """
    n = Gamma.shape[0] // 2
    x_idx = np.array(keep_indices)
    p_idx = x_idx + n
    full_idx = np.concatenate([x_idx, p_idx])
    return Gamma[np.ix_(full_idx, full_idx)]

def mutual_information(Gamma, idx_A, idx_B):
    S_A = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_A))
    S_B = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_B))
    S_AB = von_neumann_entropy_alt(extract_subsystem_covariance(Gamma, idx_A + idx_B))
    return S_A + S_B - S_AB

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

def insert_probe(Gamma,meas_set,probe_site,Gamma_2mode):
    n = Gamma.shape[0]//2
    nb = (meas_set.shape[0])//2
    Gamma_shift_right = np.zeros((2*n+2,2*n+2))
    Gamma_shift_right[0:2*n,0:nb] = Gamma[:,0:nb]
    Gamma_shift_right[0:2*n,nb+1:n+1+nb]=Gamma[:,nb:n+nb]
    Gamma_shift_right[0:2*n,n+1+nb+1:2*(n+1)]=Gamma[:,n+nb:2*n]
    Gamma_probe = Gamma_shift_right.copy()
    Gamma_probe[nb+1:n+1+nb,:]=Gamma_shift_right[nb:n+nb,:]
    Gamma_probe[n+1+nb+1:2*(n+1),:]=Gamma_shift_right[n+nb:2*n,:]
    Gamma_probe[nb,:]=0
    Gamma_probe[:,nb]=0
    Gamma_probe[nb+n+1,:]=0
    Gamma_probe[:,nb+n+1]=0
    Gamma_probe[probe_site,:]=0
    Gamma_probe[:,probe_site]=0
    Gamma_probe[probe_site+n+1,:]=0
    Gamma_probe[:,probe_site+n+1]=0
    Gamma_probe[probe_site,probe_site] = Gamma_2mode[0,0]
    Gamma_probe[probe_site,nb] = Gamma_2mode[0,1]
    Gamma_probe[nb,probe_site] = Gamma_2mode[1,0]
    Gamma_probe[nb,nb] = Gamma_2mode[1,1]
    Gamma_probe[probe_site+n+1,probe_site+n+1]=Gamma_2mode[2,2]
    Gamma_probe[probe_site+n+1,nb+n+1] = Gamma_2mode[2,3]
    Gamma_probe[nb+n+1,probe_site+n+1] = Gamma_2mode[3,2]
    Gamma_probe[nb+n+1,nb+n+1]=Gamma_2mode[3,3]
    return Gamma_probe

   


def momentum_projection_matrix(m):
    P = np.zeros((2*m, 2*m))
    P[m:, m:] = np.eye(m)
    return P

def momentum_measured_probe(Gamma,un_set,meas_set):
    na = (un_set.shape[0])//2
    nb = (meas_set.shape[0])//2
    n = Gamma.shape[0]//2

    Gamma_AA = np.zeros((4*na+2,4*na+2))
    NA = Gamma_AA.shape[0]//2

    for i in range(4):
        for j in range(4):
            if (i==0 or i==2) and (j==0 or j==2):
                Gamma_AA[(i//2)*NA:na+1+(i//2)*NA,(j//2)*NA:na+1+(j//2)*NA]=Gamma[nb+(i//2)*n:nb+na+1+(i//2)*n,nb+(j//2)*n:nb+na+1+(j//2)*n]
            if (i==1 or i==3) and (j==1 or j==3):
                Gamma_AA[(i//2+1)*NA-na:(i//2+1)*NA,(j//2+1)*NA-na:(j//2+1)*NA]=Gamma[(i//2+1)*n-na:(i//2+1)*n,(j//2+1)*n-na:(j//2+1)*n]
            if (i==0 or i==2) and (j==1 or j==3):
                i2 = i//2
                j2 = (j-1)//2
                Gamma_AA[(i2)*NA:na+1+(i2)*NA,(j2+1)*NA-na:(j2+1)*NA]=Gamma[i2*n+nb:i2*n+nb+na+1,(j2+1)*n-na:(j2+1)*n]
            if (i==1 or i==3) and (j==0 or j==2):
                i2 = (i-1)//2
                j2 = (j-1)//2
                Gamma_AA[(i2+1)*NA-na:(i2+1)*NA,j2*NA:j2*NA+na+1]=Gamma[(i2+1)*n-na:(i2+1)*n,j2*n+nb:j2*n+nb+na+1]
    
    Gamma_BB = np.zeros((4*nb,4*nb))
    NB = Gamma_BB.shape[0]//2        

    for i in range(4):
        for j in range(4):
            if (i==0 or i==2) and (j==0 or j==2):
                i2 = i//2
                j2 = j//2
                Gamma_BB[i2*NB:i2*NB+nb,j2*NB:j2*NB+nb]=Gamma[i2*n:i2*n+nb,j2*n:j2*n+nb]
            if (i==1 or i==3) and (j==1 or j==3):
                i2 = (i-1)//2
                j2 = (j-1)//2
                Gamma_BB[(i2+1)*NB-nb:(i2+1)*NB,(j2+1)*NB-nb:(j2+1)*NB]=Gamma[(i2+1)*n-na-nb:(i2+1)*n-na,(j2+1)*n-na-nb:(j2+1)*n-na]
            if (i==0 or i==2) and (j==1 or j==3):
                i2 = i//2
                j2 = (j-1)//2
                Gamma_BB[i2*NB:i2*NB+nb,(j2+1)*NB-nb:(j2+1)*NB]=Gamma[i2*n:i2*n+nb,(j2+1)*n-na-nb:(j2+1)*n-na]
            if (i==1 or i==3) and (j==0 or j==2):
                i2 = (i-1)//2
                j2 = (j-1)//2
                Gamma_BB[(i2+1)*NB-nb:(i2+1)*NB,j2*NB:j2*NB+nb]=Gamma[(i2+1)*n-na-nb:(i2+1)*n-na,j2*n:j2*n+nb]

    Gamma_AB = np.zeros((4*na+2,4*nb))
    NABi = Gamma_AB.shape[0]//2  
    NABj = Gamma_AB.shape[1]//2      

    for i in range(4):
        for j in range(4):
            if (i==0 or i==2) and (j==0 or j==2):
                i2 = i//2
                j2 = j//2
                Gamma_AB[i2*NABi:i2*NABi+na+1,j2*NABj:j2*NABj+nb]=Gamma[i2*n+nb:i2*n+nb+na+1,j2*n:j2*n+nb]
            if (i==1 or i==3) and (j==1 or j==3):
                i2 = (i-1)//2
                j2 = (j-1)//2
                Gamma_AB[(i2+1)*NABi-na:(i2+1)*NABi,(j2+1)*NABj-nb:(j2+1)*NABj]=Gamma[(i2+1)*n-na:(i2+1)*n,(j2+1)*n-na-nb:(j2+1)*n-na]
            if (i==0 or i==2) and (j==1 or j==3):
                i2 = i//2
                j2 = (j-1)//2
                Gamma_AB[i2*NABi:i2*NABi+na+1,(j2+1)*NABj-nb:(j2+1)*NABj]=Gamma[i2*n+nb:i2*n+nb+na+1,(j2+1)*n-na-nb:(j2+1)*n-na]
            if (i==1 or i==3) and (j==0 or j==2):
                i2 = (i-1)//2
                j2 = (j-1)//2
                Gamma_AB[(i2+1)*NABi-na:(i2+1)*NABi,j2*NABj:j2*NABj+nb]=Gamma[(i2+1)*n-na:(i2+1)*n,j2*n:j2*n+nb]
     

    m = Gamma_BB.shape[0]//2
    P = momentum_projection_matrix(m)
    V_bdy = Gamma_AA - Gamma_AB @ np.linalg.pinv(P @ Gamma_BB @ P) @ Gamma_AB.T

    return V_bdy

def measure_left_side(Gamma,bdy_len):
    n = Gamma.shape[0]//2
    na = bdy_len
    Gamma_AA = np.zeros((2*na+2,2*na+2))
    NA = Gamma_AA.shape[0]//2
    for i in range(4):
        for j in range(4):
            if (i==0 or i==2) and (j==0 or j==2):
                i2 = i//2
                j2 = j//2
                Gamma_AA[i2*NA:i2*NA+1,j2*NA:j2*NA+1]=Gamma[i2*n:i2*n+1,j2*n:j2*n+1]
            if (i==1 or i==3) and (j==1 or j==3):
                i2 = (i-1)//2
                j2 = (j-1)//2
                Gamma_AA[(i2+1)*NA-na:(i2+1)*NA,(j2+1)*NA-na:(j2+1)*NA]=Gamma[(i2+1)*n-na:(i2+1)*n,(j2+1)*n-na:(j2+1)*n]
            if (i==0 or i==2) and (j==1 or j==3):
                i2 = i//2
                j2 = (j-1)//2
                Gamma_AA[i2*NA:i2*NA+1,(j2+1)*NA-na:(j2+1)*NA]=Gamma[i2*n:i2*n+1,(j2+1)*n-na:(j2+1)*n]
            if (i==1 or i==3) and (j==0 or j==2):
                i2 = i//2
                j2 = (j-1)//2
                Gamma_AA[(i2+1)*NA-na:(i2+1)*NA,j2*na:j2*na+1]=Gamma[(i2+1)*n-na:(i2+1)*n,j2*n:j2*n+1]
                
    Gamma_BB = np.zeros((2*na,2*na))
    for i in range(2):
        for j in range(2):
            Gamma_BB[i*na:i*na+na,j*na:j*na+na]=Gamma[i*n+1:i*n+1+na,j*n+1:j*n+1+na]

    Gamma_AB = np.zeros((2*na+2,2*na))
    NABi = Gamma_AB.shape[0]//2
    NABj = Gamma_AB.shape[1]//2
    for i in range(4):
        for j in range(2):
            if i==0 or i==2:
                i2 = i//2
                Gamma_AB[i2*NABi:i2*NABi+1,j*NABj:j*NABj+na]=Gamma[i2*n:i2*n+1,j*n+1:j*n+1+na]
            if i==1 or i==3:
                i2 = (i-1)//2
                Gamma_AB[(i2+1)*NABi-na:(i2+1)*NABi,j*NABj:j*NABj+na]=Gamma[(i2+1)*n-na:(i2+1)*n,j*n+1:j*n+1+na]
   
    m = Gamma_BB.shape[0]//2
    P = momentum_projection_matrix(m)
    V_bdy = Gamma_AA - Gamma_AB @ np.linalg.pinv(P @ Gamma_BB @ P) @ Gamma_AB.T

    return V_bdy

def von_neumann_entropy_alt(Gamma):
    n = Gamma.shape[0] // 2
    Omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])
    eigvals = np.linalg.eigvals(1j * Gamma @ Omega)
    nu = np.sort(np.abs(eigvals))[::2]
    nu = np.clip(nu, 0.500001, None)
    return sum((nu + 0.5)*np.log(nu + 0.5) - (nu - 0.5)*np.log(nu - 0.5))

def symplectic_eigenvalues(Gamma):
    """
    Compute the symplectic eigenvalues ν_i of a covariance matrix Γ.
    """
    n = Gamma.shape[0] // 2
    Omega = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * Gamma @ Omega)
    ν = np.sort(np.abs(eigvals))[::2]  # Take only one of each ν_i pair
    return ν


def momentum_projection_matrix(m):
    P = np.zeros((2*m, 2*m))
    P[m:, m:] = np.eye(m)
    return P


def momentum_measured(Gamma,un_set,meas_set,probe_site):
    na = (un_set.shape[0]-1)//2
    nb = (meas_set.shape[0]+1)//2

    Gamma_AA = np.zeros((4*na+2,4*na+2))
    for i in range(4):
        for j in range(4):
            Gamma_AA[i*na:(i+1)*na,j*na:(j+1)*na] = Gamma[(i+1)*nb+i*na:(i+1)*nb+(i+1)*na,(j+1)*nb+j*na:(j+1)*nb+(j+1)*na]
    
        
    Gamma_BB = np.zeros((4*nb,4*nb))
    for i in range(4):
        for j in range(4):
            Gamma_BB[i*nb:(i+1)*nb,j*nb:(j+1)*nb] = Gamma[i*nb+i*na:(i+1)*nb+i*na,j*nb+j*na:(j+1)*nb+j*na]

    Gamma_AB = np.zeros((4*na,4*nb))
    for i in range(4):
        for j in range(4):
            Gamma_AB[i*na:(i+1)*na,j*nb:(j+1)*nb] = Gamma[(i+1)*nb+i*na:(i+1)*nb+(i+1)*na,j*nb+j*na:(j+1)*nb+j*na]

    m = Gamma_BB.shape[0]//2
    P = momentum_projection_matrix(m)
    V_bdy = Gamma_AA - Gamma_AB @ np.linalg.pinv(P @ Gamma_BB @ P) @ Gamma_AB.T

    return V_bdy

def williamson_strawberry(V):
    tol=1e-11
    r"""Williamson decomposition of positive-definite (real) symmetric matrix.

    See :ref:`williamson`.

    Note that it is assumed that the symplectic form is

    .. math:: \Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the identity matrix and :math:`0` is the zero matrix.

    See https://math.stackexchange.com/questions/1171842/finding-the-symplectic-matrix-in-williamsons-theorem/2682630#2682630

    Args:
        V (array[float]): positive definite symmetric (real) matrix
        tol (float): the tolerance used when checking if the matrix is symmetric: :math:`|V-V^T| \leq` tol

    Returns:
        tuple[array,array]: ``(Db, S)`` where ``Db`` is a diagonal matrix
            and ``S`` is a symplectic matrix such that :math:`V = S^T Db S`
    """
    (n, m) = V.shape

    if n != m:
        raise ValueError("The input matrix is not square")

    diffn = np.linalg.norm(V - np.transpose(V))

    if diffn >= 10**(-5):
        raise ValueError("The input matrix is not symmetric")

    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.eye(n), np.zeros((n, n))]
    ])
    vals = np.linalg.eigvalsh(V)

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = []

    # In what follows I construct a permutation matrix p  so that the Schur matrix has
    # only positive elements above the diagonal
    # Also the Schur matrix uses the x_1,p_1, ..., x_n,p_n  ordering thus I use rotmat to
    # go to the ordering x_1, ..., x_n, p_1, ... , p_n

    for i in range(n):
        if s1[2 * i, 2 * i + 1] > 0:
            seq.append(I)
        else:
            seq.append(X)

    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    dd = xpxp_to_xxpp(s1t)
    perm_indices = xpxp_to_xxpp(np.arange(2 * n))
    Ktt = Kt[:, perm_indices]
    Db = np.diag([1 / dd[i, i + n] for i in range(n)] + [1 / dd[i, i + n] for i in range(n)])
    S = Mm12 @ Ktt @ sqrtm(Db)


    eigvals, U = eigh(sqrtm(V) @ omega @ sqrtm(V))
    v = np.sort(np.abs(eigvals.real))[::2]
    return np.linalg.inv(S).T, Db, v



def construct_modular_hamiltonian_with_pinning(Gamma, epsilon_max=15, tol=1e-6):
    """
    Constructs the modular Hamiltonian K for a mixed Gaussian state Γ,
    assigning very high energy to pure modes (ν ≈ 0.5).
    """
    S, D, V = williamson_strawberry(Gamma)
    delta = 1e-5
    # Modular energies
    epsilons = []
    for v in V:
        if np.abs(v - 0.5) < tol or v < .5:
            epsilons.append(epsilon_max)  # Pin pure modes

        else:
            #eps = np.log((v + 0.5) / (v - 0.5))
            eps = 2*np.arctanh(1/(2*v))

            epsilons.append(eps)
    
    E_diag = np.diag(np.repeat(epsilons, 2))
    # Modular Hamiltonian: K = S^{-T} E S^{-1}
    S_inv = inv(S)
    K = S_inv.T @ E_diag @ S_inv
    return K

def pad_matrix_for_probe(Gamma,meas_set,probe_site):
    n = Gamma.shape[0]//2
    nb = (meas_set.shape[0])//2
    Gamma_shift_right = np.zeros((2*n+2,2*n+2))
    Gamma_shift_right[0:2*n,0:nb] = Gamma[:,0:nb]
    Gamma_shift_right[0:2*n,nb+1:n+1+nb]=Gamma[:,nb:n+nb]
    Gamma_shift_right[0:2*n,n+1+nb+1:2*(n+1)]=Gamma[:,n+nb:2*n]
    Gamma_probe = Gamma_shift_right.copy()
    Gamma_probe[nb+1:n+1+nb,:]=Gamma_shift_right[nb:n+nb,:]
    Gamma_probe[n+1+nb+1:2*(n+1),:]=Gamma_shift_right[n+nb:2*n,:]
    Gamma_probe[nb,:]=0
    Gamma_probe[:,nb]=0
    Gamma_probe[nb,probe_site]=-1
    Gamma_probe[probe_site,nb]=-1
    Gamma_probe[nb,nb]=1
    Gamma_probe[probe_site,probe_site]+=1
    Gamma_probe[nb+n+1,:]=0
    Gamma_probe[:,nb+n+1]=0
    return Gamma_probe


def pad_Atot_for_probe(Gamma,meas_set,probe_site):
    n = Gamma.shape[0]
    nb = (meas_set.shape[0])//2
    Gamma_shift_right = np.zeros((n+1,n+1))
    Gamma_shift_right[0:n,0:nb] = Gamma[:,0:nb]
    Gamma_shift_right[0:n,nb+1:n+1] = Gamma[0:n,nb:n]
    Gamma_probe = Gamma_shift_right.copy()
    Gamma_probe[nb+1:n+1,:]=Gamma_shift_right[nb:n,:]
    Gamma_probe[nb,:]=0
    Gamma_probe[:,nb]=0
    Gamma_probe[nb,probe_site]=1
    Gamma_probe[probe_site,nb]=1
    return Gamma_probe

def teleport(probe_site,n_tube,r):
    # Parameters
    L = 9
    Lh = 5
    g_tube = 1
    mu_s = 1
    t = 10

    # Build the graph
    N = 2**(Lh - 1) * (2**(L - Lh + 1) - 1)
    bdy_len = 2**(L - 1)
    bdy_1 = np.arange(N - bdy_len, N)
    N_tot = 2 * N + n_tube * 2**(Lh - 1)
    bdy_2 = np.arange(N_tot - bdy_len, N_tot)

    # Build base adjacency matrix A
    A = np.zeros((N, N), dtype=np.float64)
    for l1 in range(Lh, L + 1):
        for s1 in range(1, 2**(l1 - 1) + 1):
            for l2 in range(Lh, L + 1):
                for s2 in range(1, 2**(l2 - 1) + 1):
                    prev1 = sum(2**(k - 1) for k in range(Lh, l1))
                    prev2 = sum(2**(k - 1) for k in range(Lh, l2))
                    ind1 = prev1 + s1 - 1
                    ind2 = prev2 + s2 - 1
                    if l1 == l2 and (abs(s1 - s2) == 1 or abs(s1 - s2) == 2**(l1 - 1) - 1):
                        A[ind1, ind2] = mu_s
                    if l2 == l1 + 1 and s2 in [2*s1, 2*s1 - 1]:
                        A[ind1, ind2] = mu_s
                    if l1 == l2 + 1 and s1 in [2*s2, 2*s2 - 1]:
                        A[ind1, ind2] = mu_s

    # Full adjacency with duplicated regions and tube
    A_tot = np.zeros((N_tot, N_tot),dtype=np.float64)
    A_tot[:N, :N] = A
    A_tot[N_tot - N:, N_tot - N:] = A
    hor_1 = np.arange(2**(Lh - 1))
    for ell in range(n_tube + 1):
        offset = N + (ell - 1) * 2**(Lh - 1)
        if ell == 0:
            for i in hor_1:
                A_tot[i, i + N] = A_tot[i + N, i] = g_tube
        elif ell > 0:
            for i in hor_1:
                A_tot[i + offset, i + offset + 2**(Lh - 1)] = g_tube
                A_tot[i + offset + 2**(Lh - 1), i + offset] = g_tube
                # Horizontal connections
                if i < 2**(Lh - 1) - 1:
                    A_tot[i + offset, i + offset + 1] = A_tot[i + offset + 1, i + offset] = g_tube
                else:
                    A_tot[i + offset, i + offset - (2**(Lh - 1) - 1)] = A_tot[i + offset - (2**(Lh - 1) - 1), i + offset] = g_tube

    # Index sets
    un_set = np.concatenate([bdy_1, bdy_2])
    meas_set = np.setdiff1d(np.arange(N_tot), un_set)

    Gamma_0 =.5 * np.eye(2*N_tot,dtype=np.complex128)

    #A_tot_probe = pad_Atot_for_probe(A_tot,meas_set,probe_site)


    # Number of total modes
    n = N_tot

    # Default: mass = 1, so kinetic term is identity
    M = 1* np.eye(n)
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i]=sum(A_tot[i,:])

    # Potential term = adjacency + onsite mass term
    mu_squared = 0  # Choose this to control oscillator frequency
    K = D - A_tot + mu_squared * np.eye(n)

    # Construct full Hamiltonian H (2n x 2n) in (x1..xn, p1..pn) basis
    H = np.block([
        [K,         np.zeros((n, n))],
        [np.zeros((n, n)),   M     ]
    ])

    #insert_probe
    Gamma_2mode = two_mode_squeezed_state(r)
    Gamma_probe = insert_probe(Gamma_0,meas_set,probe_site,Gamma_2mode)


    H_padded = pad_matrix_for_probe(H,meas_set,probe_site)


    n_probe = Gamma_probe.shape[0] // 2
    Omega = symplectic_form(n_probe)
    S_t = expm(Omega @ H_padded * t)
    Gamma_q = S_t @ Gamma_probe @ S_t.T


    Gamma_TFD = momentum_measured_probe(Gamma_q,un_set,meas_set)

    probe_idx = 0 #always


    Gamma_right = measure_left_side(Gamma_TFD,bdy_len)

    mi_final = mutual_information(Gamma_right,[probe_idx],list(range(1,bdy_len+1)))
    mi_left = mutual_information(Gamma_TFD,[probe_idx],list(range(1,bdy_len+1)))
    mi_right = mutual_information(Gamma_TFD,[probe_idx],list(range(bdy_len+1,2*bdy_len+1)))
    return mi_final, mi_left, mi_right

tube_lengths = np.arange(0,11)
nl = 6
final = [[] for _ in range(nl)]
d_left_right = [[] for _ in range(nl)]
rvals = np.linspace(.1,1,5)


for i in range(nl):
    print(i)
    if i==0:
        start_site=0
    if i==1:
        start_site=16
    if i==2:
        start_site=48
    if i==3:
        start_site=112
    if i==4:
        start_site=240
    if i==5:
        start_site=496
    #sites = [random.randint(start_site,start_site+level_sites) for _ in range(7)]
    level_sites = 2**(i+4) 
    sites = [start_site,start_site+1, start_site+2,start_site+3,level_sites//2,3*level_sites//4+1]
    #sites = [start_site]
    for j in range(len(tube_lengths)):
        print(j)
        mi_fin_vals = []
        mi_left_vals = []
        mi_right_vals = []
        for r in range(len(rvals)):
            mi_fin_level_vals = []
            mi_left_level_vals = []
            mi_right_level_vals = []
            for site in sites:     
                mi_fin, mi_left, mi_right = teleport(site,j,rvals[r])
                mi_fin_level_vals.append(mi_fin)
                mi_left_level_vals.append(mi_left)
                mi_right_level_vals.append(mi_right)
            mi_fin_vals.append(sum(mi_fin_level_vals)/len(sites))
            mi_left_vals.append(sum(mi_left_level_vals)/len(sites))
            mi_right_vals.append(sum(mi_right_level_vals)/len(sites))
        final[i].append(sum(mi_fin_vals)/len(rvals))
        d_left_right[i].append(sum(mi_left_vals)/len(rvals)-sum(mi_right_vals)/len(rvals))
        

for i in range(nl):
    plt.plot(tube_lengths,final[i],label=f"layer {i}")
plt.xlabel("tube length")
plt.ylabel("final mutual information")
plt.legend()
plt.savefig("plots/tube_vs_mut_right.pdf")
plt.close()
#plt.show()

for i in range(nl):
    plt.plot(tube_lengths,d_left_right[i],label=f"layer {i}")
plt.xlabel("tube length")
plt.ylabel("mutual information left - right")
plt.legend()
plt.savefig("plots/tube_vs_mut_left_right.pdf")
plt.close()
#plt.show()

"""
print("n_tube=",n_tube)
print("probe_site=",probe_site)
print(mutual_information(Gamma_TFD,[probe_idx],list(range(1,bdy_len+1))))
print(mutual_information(Gamma_TFD,[probe_idx],list(range(bdy_len+1,2*bdy_len+1))))

left_side = []
right_side = []
for i in range(bdy_len):
    left_side.append(mutual_information(Gamma_TFD,[probe_idx],[i+1]))
    right_side.append(mutual_information(Gamma_TFD,[probe_idx],[i+bdy_len+1]))

sites=np.arange(bdy_len)
plt.plot(sites,left_side,label="left")
plt.plot(sites,right_side,label="right")
plt.legend()
plt.show()




print(mutual_information(Gamma_right,[probe_idx],list(range(1,bdy_len+1))))
right_after_meas = []
for i in range(bdy_len):
    right_after_meas.append(mutual_information(Gamma_right,[probe_idx],[i+1]))

plt.plot(sites,right_after_meas,label="right after measurement")
plt.legend()
plt.show()
"""

print("done")