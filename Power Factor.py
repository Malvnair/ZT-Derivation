import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Physical constants
k_B = 1.380649e-23        # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19 # 1 eV in J
e = 1.602176634e-19       # Elementary charge, C
h = 6.62607015e-34        # Planck constant, J·s
hbar = h / (2 * np.pi)    # Reduced Planck constant, J·s
m_e = 9.1093837015e-31    # Electron mass, kg

# Material parameters for Bi₂Te₃ at 300 K
T = 300.0                 # Temperature, K
m_star = 1.06 * m_e       # Effective mass, kg
mu_mobility = 0.02        # Mobility, m²/V·s (200 cm²/V·s)
g3D_prefactor = (1/(2*np.pi**2)) * ((2*m_star/hbar**2)**1.5)
C = e * mu_mobility * g3D_prefactor * (k_B * T)**1.5  # Conductivity prefactor

# Gamma function
gamma_3_2 = float(mp.gamma(1.5))  # Γ(3/2)

# Chemical potential range matching six-panel layout
mu_eV = np.linspace(-0.15, 0.15, 200)

# Compute Power Factor PF in W/m·K²
PF_vals = []
for mu in mu_eV:
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    z = -mp.e**(eta)
    Li_half  = float(mp.polylog(0.5, z).real)
    Li_3half = float(mp.polylog(1.5, z).real)
    
    # Electrical conductivity σ (S/m)
    sigma = -C * gamma_3_2 * Li_3half
    
    # Seebeck coefficient S (V/K)
    S = - (np.pi**2 / 3) * (k_B / e) * (Li_half / Li_3half)
    
    # Power Factor PF = S^2 * σ
    PF_vals.append((S**2) * abs(sigma))

PF_vals = np.array(PF_vals)

# Convert to mW/m·K² for plotting
PF_vals_mW = PF_vals * 1e3

plt.style.use('default')
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)

ax.plot(mu_eV, PF_vals_mW, color='#9467bd', linewidth=2.5)
ax.set_xlabel('Chemical Potential μ (eV)', fontsize=11)
ax.set_ylabel('PF (mW/m·K²)', fontsize=11)
ax.set_title('Power Factor', fontsize=12, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(mu_eV[0], mu_eV[-1])
ax.tick_params(labelsize=10)

plt.tight_layout(pad=2.0)
plt.show()
