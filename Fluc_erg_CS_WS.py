
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
nc = 300
ω = 1.0
ω0 = 1.0
g = 1.0
tlist = np.arange(0.01, (2*np.pi), 0.0001)

# Create data folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Filename with parameters
def make_filename():
    return f"data_N{N_arr[0]}-{N_arr[-1]}_step{N_arr[1]-N_arr[0]}_nc{nc}_w{ω}_w0{ω0}_g{g}.npz"

# Full path
filename = os.path.join(data_folder, "Fluc_erg_CS_WS.npz")

print("Data file:", filename)

#############################
# Central-Spin Hamiltonian
#############################

def central_spin_chain(N, nc, ω, ω0, g):
   
# ======== Collective Battery operators ============

    JpB = qt.jmat(N / 2, '+')
    JmB = qt.jmat(N / 2, '-')
    JzB = qt.jmat(N / 2, 'z')

# ======== Collective Charger operators ============

    JpC = qt.jmat(nc / 2, '+')
    JmC = qt.jmat(nc / 2, '-')
    JzC = qt.jmat(nc / 2, 'z')
    
# ======== Identities =================

    IC = qt.qeye(nc+1)
    IB = qt.qeye(N+1)
    I_full = qt.tensor(IB, IC)

# ======== Embedded operators into full space =============

# ====== Battery operators ==========

    JpB_full = qt.tensor(JpB, IC)
    JmB_full = qt.tensor(JmB, IC)
    JzB_full = qt.tensor(JzB, IC)

# ===== Charger operators ===========

    JpC_full = qt.tensor(IB, JpC)
    JmC_full = qt.tensor(IB, JmC)
    JzC_full = qt.tensor(IB, JzC)

# ======== Battery Hamiltonian =============

    HB = ω0 * (JzB_full + 0.5 * N * I_full)
    
# ======== Charger Hamiltonian =============

    HC = ω * (JzC_full + 0.5 * nc * I_full)

# ======== Interaction Hamiltonian ===========

    Hint = g * (JpB_full * JmC_full + JmB_full * JpC_full)
    
# ======== Total Hamiltonian =====================

    H = HB + HC + Hint

# ======= Battery-only Hamiltonian ==============


    HB_local = ω0 * (JzB + 0.5 * N * IB)

    return H, HB, HB_local

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

###################
# Initial state
###################

def initial_state(N, nc):

# ========= Battery state: all spins DOWN (ground) ================
    
    psiB = qt.basis(N+1, N)

# ========= Charger state: all spins UP (excited) ==================
    
    psiC = qt.basis(nc+1, 0)

    return qt.tensor(psiB, psiC)

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
    H, HB, HB_local = central_spin_chain(N, nc, ω, ω0, g)
  
    psi0 = initial_state(N, nc)

    opts = qt.Options(atol=1e-16, rtol=1e-14)         ## ODE solver options
    res = qt.sesolve(H, psi0, tlist, e_ops=HB, options=opts)
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
    H, HB, HB_local = central_spin_chain(N, nc, ω, ω0, g)

    # Initial state
    ψ0 = initial_state(N, nc)

    opts = {"atol": 1e-16, "rtol": 1e-14, "nsteps": 1000000}

    # Evolve until τ
    result = qt.sesolve(H, ψ0, [0, τ], options=opts)

    # Full density matrix at τ
    ρ_full = result.states[-1].proj()

    # Partial trace over cavity
    ρb = ρ_full.ptrace(0)

    Eb = qt.expect(HB_local, ρb)

    # TPM probability matrix
    pnm, r_val, e_val = pnm_matrix(ρb, HB_local)

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
