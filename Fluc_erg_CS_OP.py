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

N_arr = np.arange(2, 203, 4)
# nc = N_arr                           # charger size equal to battery size
ω = 1.0
ω0 = 1.0
g = 1.0
tlist = np.arange(0.01, (2*np.pi), 0.0001)

# Create data folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Filename with parameters
def make_filename():
    return f"data_N{N_arr[0]}-{N_arr[-1]}_step{N_arr[1]-N_arr[0]}_w{ω}_w0{ω0}_g{g}.npz"

# Full path
filename = os.path.join(data_folder, "Fluc_erg_CS_OP.npz")

print("Data file:", filename)

#############################
# Central-Spin Hamiltonian
#############################

def central_spin_chain(N, nc, ω, ω0, g):

    N = int(N)
    nc = int(nc)
   
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


###################
# Initial state
###################

def initial_state(N, nc):

    N = int(N)
    nc = int(nc)

# ========= Battery state: all spins DOWN (ground) ================
    
    psiB = qt.basis(N+1, N)

# ========= Charger state: all spins UP (excited) ==================
    
    psiC = qt.basis(nc+1, 0)

    return qt.tensor(psiB, psiC)


# In[44]:


#########################################
# optimal charging time τ (maximum power)
#########################################

def compute_tau(N):

    nc = N

    H, HB, HB_local = central_spin_chain(N, nc, ω, ω0, g)
    
    psi0 = initial_state(N, nc)

    opts = qt.Options(atol=1e-16, rtol=1e-14, nsteps=100000)        ## ODE solver options
    
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

    nc = N

    τ = τ_list[i]
    
    H, HB, HB_local = central_spin_chain(N, nc, ω, ω0, g)
    
    psi0 = initial_state(N, nc)

    opts = qt.Options(atol=1e-16, rtol=1e-14, nsteps=100000)        ## ODE solver options
    
    res = qt.sesolve(H, psi0, [0, τ], options=opts)
    
    rho_b = res.states[-1].ptrace(0)
    
    r_vals, r_vecs = rho_b.eigenstates()

    # Clip numerical noise to prevent negative probabilities
    r_vals = np.maximum(r_vals, 0) 
    
    # Renormalize 
    r_vals = r_vals / np.sum(r_vals)
    
    idx = np.argsort(r_vals)[::-1]
    r_vals = r_vals[idx]
    r_vecs = [r_vecs[i] for i in idx]
    
     # assign degenerate energies 
    e_vals = np.array([0] + [ω0]*N)
    
    E_B = qt.expect(HB_local, rho_b)
    
    E_erg = E_B - sum(r_vals[j]*e_vals[j] for j in range(len(r_vals)))

    W_2 = (qt.expect(HB_local**2, rho_b) + sum(r_vals[j]*(e_vals[j])**2 for j in range(len(r_vals)))) - 2 * sum(e_vals[i]*r_vals[i]*qt.expect(HB_local, r_vecs[i]) for i in range(len(r_vals)))
    
    ΔE2 = W_2 - (E_erg)**2
    
    ΔE = np.sqrt(ΔE2)

    Ratio = E_erg / E_B
    
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