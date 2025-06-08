import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Constants
k_B = 1.380649e-23        # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19 # 1 eV in J
e = 1.602176634e-19       # Elementary charge, C
T = 300.0                 # Temperature, K

# Chemical potential range in eV
mu_eV = np.linspace(-0.5, 0.5, 200)

# Compute Seebeck coefficient for each mu
S_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li_half  = mp.polylog(0.5, z).real
    Li_3half = mp.polylog(1.5, z).real
    S = - (mp.pi**2 / 3) * (k_B / e) * (Li_half / Li_3half)
    S_vals.append(float(S))

S_vals = np.array(S_vals)

# Plotting
plt.plot(mu_eV, S_vals, linewidth=2)
plt.xlabel("μ (eV)")
plt.ylabel("S (V/K)")
plt.title("Seebeck Coefficient Versus Chemical Potential at 300 K")
plt.grid(True)
plt.tight_layout()
plt.show()
