import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import os

print("Current working directory:", os.getcwd())


############
# parameters
############

N_arr = np.arange(2, 203, 4)
nmax = 300
ω = 1.0
ω0 = 1.0
g = 1.0
tlist = np.arange(0.01, (2*np.pi), 0.0001)

# Create data folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Filename with parameters
def make_filename():
    return f"data_N{N_arr[0]}-{N_arr[-1]}_step{N_arr[1]-N_arr[0]}_nmax{nmax}_w{ω}_w0{ω0}_g{g}.npz"

# Full path
filename = os.path.join(data_folder, "Fluc_erg_Dicke_WS.npz")

print("Data file:", filename)

######################
# collective operators
######################

def collective_ops(N):
    Sp = qt.jmat(N/2,'+')
    Sm = Sp.dag()
    Sz = qt.jmat(N/2,'z')
    HB = Sz + (N/2)*qt.qeye(N+1)
    return Sp, Sm, HB

#############################
# Dicke Hamiltonian
#############################

def dicke_fun(N, nmax, ω, ω0, g):
    
    a = qt.destroy(nmax)

    Ic = qt.qeye(nmax)

    Jp, Jm, HB = collective_ops(N)

    IB = qt.qeye(N + 1)
    
    H = ω  * qt.tensor(a.dag() * a, IB) + ω0 * qt.tensor(Ic, HB) + g * (1.0 / np.sqrt(N) * qt.tensor(a + a.dag(), Jp + Jm))
    
    return H, HB

################
# Passive state
################

def passive_state(rho, H):
    r_val, r_vec = rho.eigenstates()
    
    # Keep formalism intact from the notebook
    r_val = np.maximum(r_val, 0)
    r_val = r_val / np.sum(r_val)

    
    idx = np.argsort(r_val)[::-1]          # descending order
    r_val = r_val[idx]


    e_val, e_vec = H.eigenstates()         # ascending order

    return np.sum(r_val[i] * e_vec[i] * e_vec[i].dag()
               for i in range(len(r_val)))

########################
# ergotropy calculation
########################

def ergotropy(ρ, H):
    ρ_p = passive_state(ρ, H)
    return qt.expect(H, ρ) - qt.expect(H, ρ_p)

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

##############
# Pnm matrix
##############

def pnm_matrix(rho, H):
    r_val, r_vec = rho.eigenstates()
    
   
    # Clip numerical noise to prevent negative probabilities
    r_val = np.maximum(r_val, 0) 

    # Renormalize 
    r_val = r_val / np.sum(r_val)
    
    idx = np.argsort(r_val)[::-1]
    r_val = r_val[idx]
    r_vec = [r_vec[i] for i in idx]

    e_val, e_vec = H.eigenstates()

    pnm = np.zeros((len(e_val), len(e_val)))
    for n in range(len(e_val)):
        for m in range(len(e_val)):
            pnm[n, m] = r_val[m]*abs(e_vec[n].overlap(r_vec[m]))**2

    return pnm, r_val, e_val

############################
# ergotropy using pnm matrix
############################

def ergotropy_pnm(pnm, r_val, e_val):
    deltaE = e_val[:,None] - e_val[None,:]
    return np.sum(pnm * deltaE)

############################
# variance using pnm matrix
############################

def variance_pnm(pnm, r_val, e_val):
    deltaE2 = (e_val[:,None] - e_val[None,:])**2
    mean2 = np.sum(pnm * deltaE2)
    mean = ergotropy_pnm(pnm, r_val, e_val)
    return mean2 - mean**2

###########################
# Extracting Coefficients
###########################

def extract_Cn_dicke(rho_b, N):
    dim = rho_b.shape[0]
    diag = np.real(rho_b.diag())

    Cn = np.zeros(N+1)

    for idx, state in enumerate(product([0,1], repeat=N)):
        n_exc = sum(state)  
        Cn[n_exc] += diag[idx]

    return Cn

##############################################
# Plot Battery energy <Eb> as function of time 
# and pick time τ for which <Eb> is maximum
##############################################

def compute_tau(N):
    H, HB = dicke_fun(N, nmax, ω, ω0, g)
    HB_full = qt.tensor(qt.qeye(nmax), HB)
    psi0 = initial_state(N, nmax, "coherent")

    opts = qt.Options(atol=1e-16, rtol=1e-14)         ## ODE solver options
    res = qt.sesolve(H, psi0, tlist, e_ops=HB_full, options=opts)
    EB = np.array(res.expect[0])
    power = EB / tlist

    τ = tlist[np.argmax(power)]

    return τ

τ_list = Parallel(n_jobs=-1)(delayed(compute_tau)(N) for N in tqdm(N_arr, desc="Running simulation 1"))


###########################################
# Calculation of ergotropy and fluctuations
###########################################

def compute_ergotropy(i, N):

    τ = τ_list[i]

    # Rebuild Hamiltonian
    H, HB = dicke_fun(N, nmax, ω, ω0, g)

    # Initial state
    ψ0 = initial_state(N, nmax, "coherent")

    opts = {"atol": 1e-16, "rtol": 1e-14, "nsteps": 1000000}

    # Evolve until τ
    result = qt.sesolve(H, ψ0, [0, τ], options=opts)

    # Full density matrix at τ
    ρ_full = result.states[-1].proj()

    # Partial trace over cavity
    ρb = ρ_full.ptrace(1)

    Eb = qt.expect(HB, ρb)

    # TPM probability matrix
    pnm, r_val, e_val = pnm_matrix(ρb, HB)

    erg = 0
    E_second = 0

    # TPM ergotropy and variance
    for n in range(len(e_val)):

        En = e_val[n]

        for m in range(len(e_val)):

            Em_pass = 0 if m == 0 else ω0

            ΔE = En - Em_pass

            erg += pnm[n, m] * ΔE

            E_second += pnm[n, m] * (ΔE**2)

    var = E_second - erg**2

    ratio = erg / Eb

    return N, τ, Eb, erg, ratio, var

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
