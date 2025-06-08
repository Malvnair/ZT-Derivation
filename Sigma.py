import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

k_B = 1.380649e-23
eV_to_J = 1.602176634e-19
gamma_3_2 = mp.gamma(1.5)
T = 300.0
C = 1.0

mu_eV = np.linspace(-0.5, 0.5, 200)
sigma_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li = mp.polylog(1.5, z)
    sigma_vals.append(float(-C * gamma_3_2 * Li.real))
sigma_vals = np.array(sigma_vals)

plt.plot(mu_eV, sigma_vals, linewidth=2)
plt.xlabel("μ (eV)")
plt.ylabel("σ (normalized)")
plt.title("Electrical Conductivity Versus Chemical Potential at 300 K")
plt.grid(True)
plt.tight_layout()
plt.show()
