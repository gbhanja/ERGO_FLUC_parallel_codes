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

N_arr = np.array([2,10,50,100,200]) 
nmax = int(N_arr[-1] + 8*np.sqrt(N_arr[-1]))      # Fock space dimension, adjusted for larger N
ω = 1.0
ω0 = 1.0
g = 1.0


# Create data folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Filename with parameters
def make_filename():
    return f"data_N{N_arr[0]}-{N_arr[-1]}_step{N_arr[1]-N_arr[0]}_w{ω}_w0{ω0}_g{g}.npz"

# Full path
filename = os.path.join(data_folder, "Dicke_Kac_OP_coherent_bosonic_tau_evals.npz")

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

    return Sx, Sy, Sz, Sp, Sm

############################
# Dicke Hamiltonian
############################

def dicke_fun(N, nmax, ω, ω0, g):
    
    a = qt.destroy(nmax)
    Ic = qt.qeye(nmax)

    Sx, Sy, Sz, Sp, Sm = collective_ops(N)
    IB = qt.qeye(N + 1)
    HB = ω0 * (Sz + (N/2)*qt.qeye(N+1))
    
    H = ω  * qt.tensor(a.dag() * a, IB) + qt.tensor(Ic, HB) + (g / np.sqrt(N)) * qt.tensor(a + a.dag(), Sp + Sm)

    return H, HB, Sx, Sy, Sz

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

###########################
# Passive state moments
###########################

def passive_moments(r_vals, ω0):

    r = np.sort(np.maximum(r_vals,0))[::-1]
    r /= r.sum()

    E = np.arange(len(r)) * ω0

    E_pass = np.sum(r * E)
    E2_pass = np.sum(r * E**2)

    return E_pass, E2_pass

#########################################
# optimal charging time τ (fixed)
#########################################

τ_fixed = 2.0


###########################################
# Calculation of ergotropy and fluctuations
###########################################

def compute_ergotropy(N):

    τ = τ_fixed
    
    H, HB, Sx, Sy, Sz = dicke_fun(N, nmax, ω, ω0, g)
    
    psi0 = initial_state(N, nmax, "coherent")

    opts = {
        "atol":1e-16, 
        "rtol":1e-14,
        "nsteps":500000}            ## ODE solver options

    res = qt.sesolve(H, psi0, [0, τ], options=opts)
    
    rho_b = res.states[-1].ptrace(1)

    eigvals = np.real(rho_b.eigenenergies())
    eigvals = np.maximum(eigvals, 0)
    eigvals /= eigvals.sum()
    eigvals = np.sort(eigvals)[::-1]
    
    r_vals, r_vecs = rho_b.eigenstates()

    # Clip numerical noise to prevent negative probabilities
    r_vals = np.maximum(r_vals, 0) 
    # Renormalize 
    r_vals = r_vals / np.sum(r_vals)
        
    idx = np.argsort(r_vals)[::-1]
    r_vals = r_vals[idx]
    r_vecs = [r_vecs[i] for i in idx]

    E_B = qt.expect(HB, rho_b)

    # Passive-state moments in the symmetric subspace
    E_pass, E2_pass = passive_moments(r_vals, ω0)

    E_erg = E_B - E_pass

    # Cross term
    cross = 0.0

    for k in range(N + 1):
        E = k * ω0
        cross += E * r_vals[k] * qt.expect(HB, r_vecs[k])

    W_2 = qt.expect(HB**2, rho_b) + E2_pass - 2 * cross

    ΔE2 = np.real_if_close(W_2 - E_erg**2)
    ΔE = np.sqrt(max(ΔE2, 0.0))

    Ratio = E_erg / E_B
    
    return N, τ, E_B, E_erg, Ratio, ΔE2, W_2, eigvals
    
results = Parallel(n_jobs=-1)(delayed(compute_ergotropy)(N) for N in tqdm(N_arr, desc="Running simulation 1"))

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
    E_W2 = data["W2"]
    Eigvals = data["eigvals"]
    

else:
    print("Running simulation...")


    results = Parallel(n_jobs=-1)(
        delayed(compute_ergotropy)(N)
        for N in tqdm(N_arr, desc="Ergotropy"))
    

    N_out, tau_out, Eb_out, Eerg_out, ratio_out, var_out, W2_out, eigvals_out = zip(*results)

    N_arr = np.array(N_out)
    τ_list = np.array(tau_out)
    E_B_arr = np.array(Eb_out)
    E_ergo = np.array(Eerg_out)
    E_ratio = np.array(ratio_out)
    E_var = np.array(var_out)
    E_W2 = np.array(W2_out)
    

    Eigvals = np.full((len(N_arr), N_arr.max() + 1), np.nan)

    for i, ev in enumerate(eigvals_out):
        Eigvals[i, :len(ev)] = ev

    
    np.savez_compressed(
        filename,
        N=N_arr,
        tau=τ_list,
        Eb=E_B_arr,
        Eerg=E_ergo,
        ratio=E_ratio,
        variance=E_var,
        W2=E_W2,
        eigvals=Eigvals
    )

    print(f"Saved results to {filename}")
    print("Simulation completed successfully.")
