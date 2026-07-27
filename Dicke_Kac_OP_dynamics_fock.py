import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import os

print("Current working directory:", os.getcwd())

############
# parameters
############

N_arr = np.array([10, 50, 100, 150, 200, 250, 300, 350, 400])
ω = 1.0
ω0 = 1.0
g = 1.0
tlist = np.linspace(0, (10/g) , 100)

# Create data folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Filename with parameters
def make_filename():
    return f"data_N{N_arr[0]}-{N_arr[-1]}_step{N_arr[1]-N_arr[0]}_w{ω}_w0{ω0}_g{g}.npz"

# Full path
filename = os.path.join(data_folder, "Dicke_Kac_OP_dynamics_fock.npz")

print("Data file:", filename)

######################
# collective operators
######################

def collective_ops(N):

    Sx = qt.jmat(N/2, 'x')
    Sy = qt.jmat(N/2, 'y')
    Sz = qt.jmat(N/2, 'z')

    Sp = qt.jmat(N/2, '+')
    Sm = Sp.dag()

    HB = Sz + (N/2)*qt.qeye(N+1)

    return Sx, Sy, Sz, Sp, Sm, HB

############################
# Dicke Hamiltonian
############################

def dicke_fun(N, nmax, ω, ω0, g):
    
    a = qt.destroy(nmax)
    Ic = qt.qeye(nmax)

    Sx, Sy, Sz, Sp, Sm, HB = collective_ops(N)
    IB = qt.qeye(N + 1)
    
    H = ω  * qt.tensor(a.dag() * a, IB) + ω0 * qt.tensor(Ic, HB) + (g / np.sqrt(N)) * qt.tensor(a + a.dag(), Sp + Sm)

    return H, HB, Sx, Sy, Sz

################
# Initial state
################

def initial_state(N, nmax, state):
    if state=="coherent":
        psiA = qt.coherent(nmax, np.sqrt(N))
    elif state=="fock":
        psiA = qt.basis(nmax, N)
    elif state == "squeezed":
        psiA = qt.squeeze(nmax, np.arcsinh(np.sqrt(N))) * qt.basis(nmax, 0)
    psiB = qt.basis(N+1, N)  
    return qt.tensor(psiA, psiB)


###########################################
# Dynamics of battery energy & ergotropy
###########################################

def compute_dynamics(N, tlist):

    nmax = N + 1

    H, HB, _, _, _ = dicke_fun(N, nmax, ω, ω0, g)

    psi0 = initial_state(N, nmax, "fock")      

    opts = {
        "atol":1e-16,
        "rtol":1e-14,
        "nsteps":1000000
    }

    res = qt.sesolve(H, psi0, tlist, options=opts)

    EB = np.zeros(len(tlist))
    Eerg = np.zeros(len(tlist))

    e_vals = np.arange(N + 1) * ω0

    for i, state in enumerate(res.states):

        rho_b = state.ptrace(1)

        # Battery energy
        E_B = qt.expect(HB, rho_b)

        # Same ergotropy calculation as your .py file
        r_vals, r_vecs = rho_b.eigenstates()

        r_vals = np.maximum(r_vals, 0)
        r_vals = r_vals / np.sum(r_vals)

        idx = np.argsort(r_vals)[::-1]
        r_vals = r_vals[idx]
        r_vecs = [r_vecs[j] for j in idx]

        E_erg = E_B - sum(r_vals[j] * e_vals[j]
                          for j in range(len(r_vals)))

        EB[i] = E_B / N
        Eerg[i] = E_erg / N

    return EB, Eerg

results = Parallel(n_jobs=-1)(delayed(compute_dynamics)(N, tlist) for i, N in enumerate(tqdm(N_arr, desc="Running simulation for dynamics")))

############################
# LOAD or RUN
############################

if os.path.exists(filename):
    print("Loading data...")
    data = np.load(filename)

    N_arr = data["N"]
    tlist = data["tlist"]
    E_B_arr = data["Eb"]
    E_ergo = data["Eerg"]

else:
    print("Running simulation...")

    results = Parallel(n_jobs=-1)(
        delayed(compute_dynamics)(N, tlist)
        for N in tqdm(N_arr, desc="Dynamics")
    )

    E_B_arr, E_ergo = zip(*results)

    E_B_arr = np.array(E_B_arr)
    E_ergo = np.array(E_ergo)

    np.savez_compressed(
        filename,
        N=N_arr,
        tlist=tlist,
        Eb=E_B_arr,
        Eerg=E_ergo
    )

    print(f"Saved results to {filename}")
    print("Simulation completed successfully.")