import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Constants
k_B = 1.380649e-23        # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19 # 1 eV in J
e = 1.602176634e-19       # Elementary charge, C
T = 300.0                 # Temperature, K

# Empirical constant D with correct 116e-6 factor
D = mp.e**(-mp.pi**2 * k_B / (3 * e * 116e-6))

# Chemical potential range matching six-panel plot
mu_eV = np.linspace(-0.15, 0.15, 200)

# Compute dimensionless Lorenz number (units of 10^-8 W·Ω·K^-2)
L_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li_half  = float(mp.polylog(0.5, z).real)
    Li_3half = float(mp.polylog(1.5, z).real)
    R = abs(Li_half / Li_3half)
    L_vals.append(1.5 + D**R)

L_vals = np.array(L_vals)

plt.style.use('default')
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)

ax.plot(mu_eV, L_vals, color='#2ca02c', linewidth=2.5)
ax.set_xlabel('Chemical Potential μ (eV)', fontsize=11)
ax.set_ylabel('L (10⁻⁸ W·Ω·K⁻²)', fontsize=11)
ax.set_title('Lorenz Number', fontsize=12, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(mu_eV[0], mu_eV[-1])
ax.set_ylim(1.55, 2.05) 
ax.tick_params(labelsize=10)

plt.tight_layout(pad=2.0)
plt.savefig("Lorentz")
plt.show()
