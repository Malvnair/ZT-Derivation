import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Constants
k_B = 1.380649e-23        # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19 # 1 eV in J
e = 1.602176634e-19       # Elementary charge, C
T = 300.0                 # Temperature, K
C = 1.0                   # Prefactor (normalized conductivity)
gamma_3_2 = mp.gamma(1.5) # Γ(3/2)

# Chemical potential range in eV
mu_eV = np.linspace(-0.5, 0.5, 200)

# Compute Power Factor for each mu
PF_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li_half  = mp.polylog(0.5, z).real
    Li_3half = mp.polylog(1.5, z).real
    PF = -C * gamma_3_2 * (mp.pi**4 / 9) * (k_B**2 / e**2) * (Li_half**2 / Li_3half)
    PF_vals.append(float(PF))

PF_vals = np.array(PF_vals)

# Plotting
plt.plot(mu_eV, PF_vals, linewidth=2)
plt.xlabel("μ (eV)")
plt.ylabel("PF (V²/K²)")
plt.title("Power Factor Versus Chemical Potential at 300 K")
plt.grid(True)
plt.tight_layout()
plt.show()
