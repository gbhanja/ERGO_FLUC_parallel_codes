import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from math import comb
from tqdm import tqdm
from joblib import Parallel, delayed
import os

print("Current working directory:", os.getcwd())

############
# parameters
############

N_arr = np.array([10, 50, 100, 150, 200, 300, 400])
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
filename = os.path.join(data_folder, "TC_Kac_OP_dynamics_fock_full.npz")

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
# Tavis-Cummings Hamiltonian
############################

def tavis_cummings(N, nmax, ω, ω0, g):
    a = qt.destroy(nmax)
    Sx, Sy, Sz, Sp, Sm, HB = collective_ops(N)
    H = (ω * qt.tensor(a.dag()*a, qt.qeye(N+1)) + ω0 * qt.tensor(qt.qeye(nmax), HB) + (g / np.sqrt(N)) * (qt.tensor(a, Sp) + qt.tensor(a.dag(), Sm)))
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

################################
# Passive state moments
################################

def passive_moments(r_vals, omega0):

    r = np.sort(np.maximum(r_vals, 0))[::-1]
    r /= r.sum()

    N = len(r) - 1

    E_pass = 0.0
    E2_pass = 0.0

    i = 0

    for k in range(N + 1):

        E = k * omega0

        for _ in range(min(comb(N, k), len(r) - i)):
            E_pass += r[i] * E
            E2_pass += r[i] * E**2
            i += 1

        if i == len(r):
            break

    return E_pass, E2_pass

###########################################
# Dynamics of battery energy & ergotropy
###########################################

def compute_dynamics(N, tlist):

    nmax =  N + 400

    H, HB, _, _, _  = tavis_cummings(N, nmax, ω, ω0, g)
    psi0 = initial_state(N, nmax, "fock")

    opts = {
        "atol": 1e-8,
        "rtol": 1e-6,
        "nsteps": 500000
    }

    res = qt.sesolve(H, psi0, tlist, options=opts)

    EB = np.zeros(len(tlist))
    Eerg = np.zeros(len(tlist))

    for i, state in enumerate(res.states):

        rho_b = state.ptrace(1)

        E_B = qt.expect(HB, rho_b)

        r_vals, _ = rho_b.eigenstates()

        r_vals = np.maximum(r_vals, 0)
        r_vals /= r_vals.sum()

        # Passive-state energy in the full 2^N Hilbert space
        E_pass, _ = passive_moments(r_vals, ω0)

        EB[i] = E_B / N
        Eerg[i] = (E_B - E_pass) / N

    return EB, Eerg

results = Parallel(n_jobs=64)(delayed(compute_dynamics)(N, tlist) for i, N in enumerate(tqdm(N_arr, desc="Running simulation for dynamics")))

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