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

# Material parameters for Bi₂Te₃ at 300K
T = 300.0                 # Temperature, K
m_star = 1.06 * m_e       # Effective mass, kg
mu_mobility = 0.02        # Mobility, m²/V·s (200 cm²/V·s)
kappa_l = 1.5             # Lattice thermal conductivity, W/m·K

# Calculate proper conductivity prefactor C
# C = (e * mu * g_3D) where g_3D is 3D density of states prefactor
# g_3D = (1/2π²) * (2m*/ℏ²)^(3/2) * (k_B*T)^(3/2)
g_3D_prefactor = (1/(2*np.pi**2)) * ((2*m_star/hbar**2)**(3/2))
C = e * mu_mobility * g_3D_prefactor * (k_B * T)**(3/2)
# correct empirical constant D (includes the “×10⁻⁶” in the denominator)
D = np.e**(-np.pi**2 * k_B / (3 * e * 116e-6))

print(f"Conductivity prefactor C = {C:.2e} S/m")

# Other constants
gamma_3_2 = float(mp.gamma(1.5))  # Γ(3/2) = √π/2

# Corrected empirical constants for Lorenz number
def lorentz_factor(R_val):
    """Empirical Lorenz number formula"""
    return 1.5 + np.exp(-R_val / 116e-6)

# Power factor constant
G = (np.pi**4 / 9) * (k_B**2 / e**2)

# Chemical potential range (more realistic for degenerate semiconductor)
# Scan around the band edge where real thermoelectric operation occurs
mu_eV = np.linspace(-0.15, 0.15, 200)

# More conservative clipping to avoid numerical issues
ETA_MIN = -20
ETA_MAX = 20

def safe_polylog(s, z):
    """Safely compute polylog with error handling"""
    try:
        if abs(z) > 1e6:  # Near convergence boundary
            z = 1e6 * np.sign(z.real) if z.real != 0 else 0.99j * np.sign(z.imag)
        result = mp.polylog(s, z)
        return complex(result).real
    except:
        return 0.0

def _li_values(mu):
    """Return polylog(1/2) and polylog(3/2) for chemical potential mu (eV)"""
    mu_J = mu * eV_to_J
    eta = mu_J / (k_B * T)
    # Clamp eta to keep z in a stable range
    eta = max(min(eta, ETA_MAX), ETA_MIN)
    z = -np.exp(eta)
    
    Li_half = safe_polylog(0.5, z)
    Li_3half = safe_polylog(1.5, z)
    
    return Li_half, Li_3half

def sigma(mu):
    """Electrical conductivity σ (S/m)"""
    _, Li_3half = _li_values(mu)
    return -C * gamma_3_2 * Li_3half

def seebeck(mu):
    """Seebeck coefficient S (V/K)"""
    Li_half, Li_3half = _li_values(mu)
    if abs(Li_3half) < 1e-50:  # Avoid division by zero
        return 0.0
    return -(np.pi**2 / 3) * (k_B / e) * (Li_half / Li_3half)



def lorentz(mu):
    """Lorenz number L(μ) = (1.5 + D**R)·1e-8, where R = |Li₁/₂/Li₃/₂|."""
    Li_half, Li_3half = _li_values(mu)
    R = abs(Li_half / Li_3half)
    return (1.5 + D**R) * 1e-8


def kappa_e(mu):
    """Electronic thermal conductivity κₑ (W/m·K)"""
    return lorentz(mu) * abs(sigma(mu)) * T

def power_factor(mu):
    """Power factor PF (W/m·K²)"""
    S = seebeck(mu)
    sig = sigma(mu)
    return S**2 * abs(sig)

def ZT(mu):
    """Thermoelectric figure of merit ZT"""
    PF = power_factor(mu)
    k_e = kappa_e(mu)
    return (PF * T) / (k_e + kappa_l)

# Compute values for each chemical potential
sigma_vals = np.array([sigma(mu) for mu in mu_eV])
S_vals = np.array([seebeck(mu) * 1e6 for mu in mu_eV])  # Convert to μV/K
L_vals = np.array([lorentz(mu) * 1e8 for mu in mu_eV])  # Convert to 10⁻⁸ W·Ω·K⁻²
kappa_e_vals = np.array([kappa_e(mu) for mu in mu_eV])
PF_vals = np.array([power_factor(mu) for mu in mu_eV])
ZT_vals = np.array([ZT(mu) for mu in mu_eV])

# Print some diagnostics
max_zt_idx = np.argmax(np.abs(ZT_vals))
print(f"\nDiagnostics at μ = {mu_eV[max_zt_idx]:.3f} eV:")
print(f"σ = {sigma_vals[max_zt_idx]:.2e} S/m")
print(f"S = {S_vals[max_zt_idx]:.1f} μV/K")
print(f"L = {L_vals[max_zt_idx]:.2f} × 10⁻⁸ W·Ω·K⁻²")
print(f"κₑ = {kappa_e_vals[max_zt_idx]:.3f} W/m·K")
print(f"PF = {PF_vals[max_zt_idx]:.2e} W/m·K²")
print(f"ZT = {ZT_vals[max_zt_idx]:.3f}")

# Plotting with corrected units and scale
plt.style.use('default')
fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=120)
axes = axes.flatten()

# Define colors
colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

plots = [
    (sigma_vals/1e4, 'σ (10⁴ S/m)', 'Electrical Conductivity', colors[0]),
    (S_vals, 'S (μV/K)', 'Seebeck Coefficient', colors[1]),
    (L_vals, 'L (10⁻⁸ W·Ω·K⁻²)', 'Lorenz Number', colors[2]),
    (kappa_e_vals, 'κₑ (W/m·K)', 'Electronic Thermal Conductivity', colors[3]),
    (PF_vals*1e3, 'PF (mW/m·K²)', 'Power Factor', colors[4]),
    (ZT_vals, 'ZT', 'Thermoelectric Figure of Merit', colors[5]),
]

for idx, (vals, ylabel, title, color) in enumerate(plots):
    ax = axes[idx]
    ax.plot(mu_eV, vals, color=color, linewidth=2.5)
    ax.set_xlabel('Chemical Potential μ (eV)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(mu_eV[0], mu_eV[-1])

# Add some formatting
for ax in axes:
    ax.tick_params(labelsize=10)
    
plt.tight_layout(pad=2.0)
plt.savefig("thermoelectric_properties.png", dpi=150, bbox_inches='tight')
plt.show()

# Additional analysis: Find optimal operating point
optimal_idx = np.argmax(ZT_vals)
print(f"\nOptimal operating point:")
print(f"Chemical potential: {mu_eV[optimal_idx]:.3f} eV")
print(f"Maximum ZT: {ZT_vals[optimal_idx]:.3f}")
print(f"At this point:")
print(f"  σ = {sigma_vals[optimal_idx]/1e4:.1f} × 10⁴ S/m")
print(f"  S = {S_vals[optimal_idx]:.0f} μV/K")
print(f"  PF = {PF_vals[optimal_idx]*1e3:.1f} mW/m·K²")