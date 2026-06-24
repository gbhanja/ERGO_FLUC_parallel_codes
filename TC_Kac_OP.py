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

N_arr = np.arange(2, 402, 2)
nmax = int(N_arr[-1] + 8*np.sqrt(N_arr[-1]))      # Fock space dimension, adjusted for larger N
ω = 1.0
ω0 = 1.0
g = 1.0


# Create data folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Filename with parameters
def make_filename():
    return f"data_N{N_arr[0]}-{N_arr[-1]}_step{N_arr[1]-N_arr[0]}_nmax{nmax}_w{ω}_w0{ω0}_g{g}.npz"

# Full path
filename = os.path.join(data_folder, "TC_Kac_OP.npz")

print("Data file:", filename)

######################
# collective operators
######################

def collective_ops(N):
    Sp = qt.jmat(N/2, '+')
    Sm = Sp.dag()
    Sz = qt.jmat(N/2, 'z')
    HB = Sz + (N/2)*qt.qeye(N+1)
    return Sp, Sm, HB

############################
# Tavis-Cummings Hamiltonian
############################

def tavis_cummings(N, nmax, ω, ω0, g):
    a = qt.destroy(nmax)
    Sp, Sm, HB = collective_ops(N)
    H = ω * qt.tensor(a.dag()*a, qt.qeye(N+1)) + ω0 * qt.tensor(qt.qeye(nmax), HB) + (g / np.sqrt(N)) * (qt.tensor(a, Sp) + qt.tensor(a.dag(), Sm))
    return H, HB

################
# Initial state
################

def initial_state(N, nmax, state):
    if state=="coherent":
        psiA = qt.coherent(nmax, np.sqrt(N))
    elif state=="fock":
        psiA = qt.basis(nmax, N)
    psiB = qt.basis(N+1, N)  
    return qt.tensor(psiA, psiB)

#########################################
# optimal charging time τ (maximum power)
#########################################

def compute_tau(N):
    H, HB = tavis_cummings(N, nmax, ω, ω0, g)
    HB_full = qt.tensor(qt.qeye(nmax), HB)
    psi0 = initial_state(N, nmax, "coherent")


    t_max =  1.16556 /  g 
    tlist_local = np.linspace(t_max * 0.001, t_max, 1000)

    opts = {
        "atol":1e-16, 
        "rtol":1e-14,
        "nsteps":100000}            ## ODE solver options

    res = qt.sesolve(H, psi0, tlist_local, e_ops=HB_full, options=opts)

    EB = np.array(res.expect[0])

    power = EB / tlist_local

    τ = tlist_local[np.argmax(power)]

    return τ

τ_list = Parallel(n_jobs=-1)(delayed(compute_tau)(N) for N in tqdm(N_arr, desc="Running simulation 1"))


###########################################
# Calculation of ergotropy and fluctuations
###########################################

def compute_ergotropy(i, N):

    τ = τ_list[i]
    
    H, HB = tavis_cummings(N, nmax, ω, ω0, g)
    
    psi0 = initial_state(N, nmax, "coherent")

    opts = {
        "atol":1e-16, 
        "rtol":1e-14,
        "nsteps":100000}            ## ODE solver options
    
    res = qt.sesolve(H, psi0, [0, τ], options=opts)
    
    rho_b = res.states[-1].ptrace(1)

    E_B = qt.expect(HB, rho_b)
    
    r_vals, r_vecs = rho_b.eigenstates()

    # Clip numerical noise to prevent negative probabilities
    r_vals = np.maximum(r_vals, 0) 
    # Renormalize 
    r_vals = r_vals / np.sum(r_vals)
        
    idx = np.argsort(r_vals)[::-1]
    r_vals = r_vals[idx]
    r_vecs = [r_vecs[i] for i in idx]

    # assign degenerate energies 
    # e_vals = np.array([0] + [ω0]*N)
    
    e_vals = np.arange(N + 1) * ω0
    
    E_B = qt.expect(HB, rho_b)
    
    E_erg = E_B - sum(r_vals[j]*e_vals[j] for j in range(len(r_vals)))

    W_2 = (qt.expect(HB**2, rho_b) + sum(r_vals[j]*(e_vals[j])**2 for j in range(len(r_vals)))) - 2 * sum(e_vals[i]*r_vals[i]*qt.expect(HB, r_vecs[i]) for i in range(len(r_vals)))
    
    ΔE2 = W_2 - (E_erg)**2
    
    ΔE = np.sqrt(ΔE2)

    Ratio = E_erg / E_B
    
    return N, τ, E_B, E_erg, Ratio, ΔE2
    
results = Parallel(n_jobs=-1)(delayed(compute_ergotropy)(i, N) for i, N in enumerate(tqdm(N_arr, desc="Running simulation 2")))

############################
# LOAD or RUN
############################
if os.path.exists(filename):
    print("Loading data...")
    data = np.load(filename)

    N_arr = data["N"]
    τ_list = data["tau"]
    E_B_arr = data["Eb"]
    E_ergo = data["Eerg"]
    E_ratio = data["ratio"]
    E_var = data["variance"]

else:
    print("Running simulation...")

    τ_list = Parallel(n_jobs=-1)(
        delayed(compute_tau)(N)
        for N in tqdm(N_arr, desc="τ computation")
    )

    results = Parallel(n_jobs=-1)(
        delayed(compute_ergotropy)(i, N)
        for i, N in enumerate(tqdm(N_arr, desc="Ergotropy"))
    )

    N_out, tau_out, Eb_out, Eerg_out, ratio_out, var_out = zip(*results)

    N_arr = np.array(N_out)
    τ_list = np.array(tau_out)
    E_B_arr = np.array(Eb_out)
    E_ergo = np.array(Eerg_out)
    E_ratio = np.array(ratio_out)
    E_var = np.array(var_out)

    np.savez_compressed(
        filename,
        N=N_arr,
        tau=τ_list,
        Eb=E_B_arr,
        Eerg=E_ergo,
        ratio=E_ratio,
        variance=E_var
    )

    print(f"Saved results to {filename}")
    print("Simulation completed successfully.")
