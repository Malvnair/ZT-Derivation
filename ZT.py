import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Constants
k_B = 1.380649e-23        # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19 # 1 eV in J
e = 1.602176634e-19       # Elementary charge, C
T = 300.0                 # Temperature, K
C = 1.0                   # Conductivity prefactor (normalized)
gamma_3_2 = mp.gamma(1.5) # Γ(3/2)

# Empirical constants
D = mp.e**(-mp.pi**2 * k_B / (3 * e * 116))     # from Lorenz exponent fileciteturn2file1
G = (mp.pi**4 / 9) * (k_B**2 / e**2)             # PF-to-ZT constant fileciteturn2file4

# Lattice thermal conductivity (set to your value; units consistent with κe)
kappa_l = 1e-3  # placeholder value

# Chemical potential range (ϕ ≡ μ) in eV
mu_eV = np.linspace(-0.5, 0.5, 200)

ZT_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li_half  = mp.polylog(0.5, z).real
    Li_3half = mp.polylog(1.5, z).real
    R = abs(Li_half / Li_3half)
    
    # Numerator: PF × T
    numerator = G * (Li_half**2 / Li_3half) * T
    
    # Denominator: κe + κl term fileciteturn2file2
    kappa_e = -C * gamma_3_2 * Li_3half * (1.5 + D**R) * 1e-8 * T
    denominator = kappa_e + kappa_l
    
    ZT = numerator / denominator
    ZT_vals.append(float(ZT))

ZT_vals = np.array(ZT_vals)

# Plotting
plt.plot(mu_eV, ZT_vals, linewidth=2)
plt.xlabel("μ (eV)")
plt.ylabel("ZT")
plt.title("Thermoelectric Figure of Merit ZT versus Chemical Potential at 300 K")
plt.grid(True)
plt.tight_layout()
plt.show()
