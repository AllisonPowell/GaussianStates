import numpy as np
from scipy.linalg import eigh, expm

from tenpy.networks.site import BosonSite
from tenpy.networks.mps import MPS
from tenpy.linalg import np_conserved as npc


# ============================================================
# Bosonic local operators
# ============================================================

def local_boson_ops(Nmax):

    d = Nmax + 1

    b = np.zeros((d,d),dtype=complex)

    for n in range(1,d):
        b[n-1,n] = np.sqrt(n)

    bd = b.conj().T

    x = (b + bd)/np.sqrt(2)
    p = (b - bd)/(1j*np.sqrt(2))

    I = np.eye(d)

    return dict(x=x,p=p,b=b,bd=bd,I=I)


# ============================================================
# Kronecker helpers
# ============================================================

def kron_all(ops):

    out = ops[0]

    for A in ops[1:]:
        out = np.kron(out,A)

    return out


def embed_one_site(op,site,L,d):

    ops = [np.eye(d) for _ in range(L)]
    ops[site] = op

    return kron_all(ops)


def embed_two_site(op, i, j, L, d):
    """
    Embed a 2-site operator acting on sites i,j into an L-site Hilbert space.

    Works for arbitrary i,j (not necessarily adjacent).
    """

    if i > j:
        i, j = j, i

    # reshape op → (d,d,d,d)
    op = op.reshape(d, d, d, d)

    # build full operator tensor
    dims = [d]*L

    O = np.zeros(dims + dims, dtype=complex)

    for a in range(d):
        for b in range(d):
            for c in range(d):
                for e in range(d):

                    idx_in  = [slice(None)]*L
                    idx_out = [slice(None)]*L

                    idx_in[i] = a
                    idx_in[j] = b

                    idx_out[i] = c
                    idx_out[j] = e

                    O[tuple(idx_out + idx_in)] = op[c,e,a,b]

    return O.reshape(d**L, d**L)

# ============================================================
# Ring Hamiltonian
# ============================================================

def build_ring_H(Nsites,Nmax,m2,k,lam):

    ops = local_boson_ops(Nmax)

    x = ops["x"]
    p = ops["p"]

    d = Nmax+1
    L = Nsites

    H = np.zeros((d**L,d**L),dtype=complex)

    # onsite terms

    for i in range(L):

        H += 0.5*embed_one_site(p@p,i,L,d)
        H += 0.5*m2*embed_one_site(x@x,i,L,d)

        if lam>0:
            H += lam*embed_one_site(x@x@x@x,i,L,d)

    # ring nearest neighbour

    x2 = x@x
    xx = np.kron(x,x)

    for i in range(L):

        j = (i+1)%L

        H += 0.5*k*embed_one_site(x2,i,L,d)
        H += 0.5*k*embed_one_site(x2,j,L,d)
        H += -k*embed_two_site(xx,i,j,L,d)

    return 0.5*(H+H.conj().T)


# ============================================================
# TFD construction
# ============================================================

def build_tfd_tensor(H_side,Nsites,Nmax,beta):

    d = Nmax+1

    evals,evecs = eigh(H_side)

    evals -= evals.min()

    w = np.exp(-beta*evals)

    Z = np.sum(w)

    C = evecs @ np.diag(np.sqrt(w/Z)) @ evecs.conj().T

    psi = C.reshape([d]*Nsites + [d]*Nsites)

    return psi


# ============================================================
# Insert message with environment
# ============================================================

def insert_with_env(psi_tensor,insert_idx,phi):

    d = psi_tensor.shape[0]

    psi_env = np.zeros(psi_tensor.shape+(d,),dtype=complex)

    psi_env[...,0] = psi_tensor

    psi_swapped = np.swapaxes(psi_env,insert_idx,psi_env.ndim-1)

    U = np.eye(d)

    U[:,0] = phi

    psi_out = np.tensordot(U,psi_swapped,axes=(1,insert_idx))
    psi_out = np.moveaxis(psi_out,0,insert_idx)

    return psi_out


# ============================================================
# Convert tensor → MPS
# ============================================================

def tensor_to_mps(psi_tensor,Nmax):

    nsites = psi_tensor.ndim

    sites = [BosonSite(Nmax=Nmax) for _ in range(nsites)]

    psi_npc = npc.Array.from_ndarray_trivial(
        psi_tensor,
        labels=[f'p{i}' for i in range(nsites)]
    )

    psi = MPS.from_full(sites,psi_npc,normalize=True)

    psi.canonical_form()

    return psi


# ============================================================
# Trotter gates
# ============================================================

def onsite_unitary(Nmax,dt,m2,lam):

    ops = local_boson_ops(Nmax)

    x = ops["x"]
    p = ops["p"]

    h = 0.5*p@p + 0.5*m2*x@x + lam*x@x@x@x

    return expm(-1j*dt*h)


def bond_unitary(Nmax,dt,k):

    ops = local_boson_ops(Nmax)

    x = ops["x"]

    d = Nmax+1

    h = -k*np.kron(x,x)

    return expm(-1j*dt*h)


# ============================================================
# Apply gates
# ============================================================

def apply_one_site(psi,i,U):

    op = npc.Array.from_ndarray_trivial(U,labels=['p','p*'])

    psi.apply_local_op(i,op,unitary=True)

    return psi


def apply_two_site(psi,i,U):

    d = int(np.sqrt(U.shape[0]))

    op = npc.Array.from_ndarray_trivial(
        U.reshape(d,d,d,d),
        labels=['p0','p1','p0*','p1*']
    )

    psi.apply_local_op(i,op,unitary=True)

    return psi

from scipy.linalg import expm

def gaussian_mode(Nmax, r=0.5, theta=0.0):

    ops = local_boson_ops(Nmax)

    b = ops["b"]
    bd = ops["bd"]

    d = Nmax+1

    vacuum = np.zeros(d)
    vacuum[0] = 1

    # squeezing
    Hs = 0.5*(b@b - bd@bd)
    S = expm(r*Hs)

    # rotation
    n = bd@b
    R = expm(-1j*theta*n)

    U = R @ S

    phi = U @ vacuum

    return phi / np.linalg.norm(phi)

# ============================================================
# Protocol
# ============================================================

def teleportation_protocol():
    N=3

    Nmax = 6

    beta = 1.0

    m2 = 13
    k = 5
    lam = 0.02

    g = 0.2

    dt = 0.05
    t_scramble = 1.0

    insert_idx = 1

    # build ring Hamiltonian

    H_quad = build_ring_H(N,Nmax,m2,k,lam=0)

    # build TFD

    psi_tensor = build_tfd_tensor(H_quad,N,Nmax,beta)

    # backward evolve left (dense)

    H_left = build_ring_H(N,Nmax,m2,k,lam)

    U_back = expm(1j*t_scramble*H_left)

    psi_left = psi_tensor.reshape((Nmax+1)**N,-1)

    psi_left = U_back @ psi_left

    psi_tensor = psi_left.reshape([Nmax+1]*(2*N))

    # insert gaussian mode

    phi = np.zeros(Nmax+1)
    phi[0]=1

    psi_tensor = insert_with_env(psi_tensor,insert_idx,phi)

    # convert to MPS

    psi = tensor_to_mps(psi_tensor,Nmax)

    # forward evolve left

    U1 = onsite_unitary(Nmax,dt,m2,lam)
    U2 = bond_unitary(Nmax,dt,k)

    steps = int(t_scramble/dt)

    for _ in range(steps):

        for i in range(N):
            apply_one_site(psi,i,U1)
        for i in range(N):
            apply_two_site(psi,i,U2) # ring bond

    # coupling

    Uc = bond_unitary(Nmax,dt,g)

    for i in range(N):

        apply_two_site(psi,i,Uc)

    # evolve right

    for _ in range(steps):

        for i in range(N,2*N):
            apply_one_site(psi,i,U1)
        for i in range(N,2*N):
            apply_two_site(psi,i,U2)
            
    return psi


if __name__=="__main__":

    psi = teleportation_protocol()

    print("Simulation finished.")