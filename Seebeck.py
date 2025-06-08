import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Constants
k_B = 1.380649e-23        # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19 # 1 eV in J
e = 1.602176634e-19       # Elementary charge, C
T = 300.0                 # Temperature, K

# Chemical potential range matching six‐panel layout
mu_eV = np.linspace(-0.15, 0.15, 200)

# Compute Seebeck coefficient values in μV/K
S_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li_half  = float(mp.polylog(0.5, z).real)
    Li_3half = float(mp.polylog(1.5, z).real)
    # Seebeck S in V/K, then convert to μV/K
    S = - (mp.pi**2 / 3) * (k_B / e) * (Li_half / Li_3half)
    S_vals.append(float(S) * 1e6)

S_vals = np.array(S_vals)

plt.style.use('default')
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)

ax.plot(mu_eV, S_vals, color='#1f77b4', linewidth=2.5)
ax.set_xlabel('Chemical Potential μ (eV)', fontsize=11)
ax.set_ylabel('S (μV/K)', fontsize=11)
ax.set_title('Seebeck Coefficient', fontsize=12, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(mu_eV[0], mu_eV[-1])
ax.tick_params(labelsize=10)

plt.tight_layout(pad=2.0)
plt.savefig("Seebeck")
plt.show()
