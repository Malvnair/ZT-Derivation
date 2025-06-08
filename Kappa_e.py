import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Constants
k_B = 1.380649e-23        # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19 # 1 eV in J
e = 1.602176634e-19       # Elementary charge, C
T = 300.0                 # Temperature, K
C = 1.0                   # Prefactor from conductivity
gamma_3_2 = mp.gamma(1.5) # Γ(3/2)

# Constant D: D = exp[-π^2 k_B / (3 e * 116)]
D = mp.e**(-mp.pi**2 * k_B / (3 * e * 116))

# Chemical potential range in eV
mu_eV = np.linspace(-0.5, 0.5, 200)

# Compute electronic thermal conductivity for each μ
kappa_e_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li_3half = mp.polylog(1.5, z).real
    Li_half = mp.polylog(0.5, z).real
    R = abs(Li_half / Li_3half)
    L_dimless = 1.5 + D**R              # dimensionless Lorenz number
    kappa_e = -C * gamma_3_2 * Li_3half * L_dimless * 1e-8 * T
    kappa_e_vals.append(float(kappa_e))

kappa_e_vals = np.array(kappa_e_vals)

# Plotting
plt.plot(mu_eV, kappa_e_vals, linewidth=2)
plt.xlabel("μ (eV)")
plt.ylabel("κₑ (W·Ω·K⁻¹)")
plt.title("Electronic Thermal Conductivity Versus Chemical Potential at 300 K")
plt.grid(True)
plt.tight_layout()
plt.show()
