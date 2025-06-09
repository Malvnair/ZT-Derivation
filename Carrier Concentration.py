import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant, J/K
eV_to_J = 1.602176634e-19  # 1 eV in J

def lambert_w_extremum(phi, T):
    """
    Calculate the extremum energy E* using Lambert W function
    
    E* = k_B*T * [1/2 + W(1/2 * exp(φ/(k_B*T) - 1/2))]
    
    Parameters:
    phi: chemical potential (J)
    T: temperature (K)
    
    Returns:
    E*: extremum energy (J)
    """
    # Calculate the argument of Lambert W
    arg = 0.5 * np.exp(phi / (k_B * T) - 0.5)
    
    # Use Lambert W function (principal branch)
    w_val = lambertw(arg, k=0).real
    
    # Calculate E*
    E_star = k_B * T * (0.5 + w_val)
    
    return E_star

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
plt.style.use('seaborn-v0_8')

# Plot 1: E* vs Chemical Potential at different temperatures
phi_range = np.linspace(-0.2, 0.2, 200) * eV_to_J  # -0.2 to 0.2 eV
temperatures = [200, 300, 400, 500]  # K

for T in temperatures:
    E_star_vals = [lambert_w_extremum(phi, T) for phi in phi_range]
    E_star_eV = np.array(E_star_vals) / eV_to_J  # Convert to eV
    phi_eV = phi_range / eV_to_J
    
    ax1.plot(phi_eV, E_star_eV, linewidth=2.5, label=f'T = {T} K')

ax1.set_xlabel('Chemical Potential φ (eV)', fontsize=12)
ax1.set_ylabel('Extremum Energy E* (eV)', fontsize=12)
ax1.set_title('(a) E* vs Chemical Potential at Different Temperatures', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: E* vs Temperature at different chemical potentials
T_range = np.linspace(100, 600, 200)  # K
phi_values = [-0.1, -0.05, 0, 0.05, 0.1]  # eV

for phi_eV in phi_values:
    phi = phi_eV * eV_to_J
    E_star_vals = [lambert_w_extremum(phi, T) for T in T_range]
    E_star_eV = np.array(E_star_vals) / eV_to_J
    
    ax2.plot(T_range, E_star_eV, linewidth=2.5, label=f'φ = {phi_eV} eV')

ax2.set_xlabel('Temperature T (K)', fontsize=12)
ax2.set_ylabel('Extremum Energy E* (eV)', fontsize=12)
ax2.set_title('(b) E* vs Temperature at Different Chemical Potentials', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: 3D surface plot
T_3d = np.linspace(200, 500, 50)
phi_3d = np.linspace(-0.15, 0.15, 50) * eV_to_J
T_mesh, phi_mesh = np.meshgrid(T_3d, phi_3d)

E_star_mesh = np.zeros_like(T_mesh)
for i in range(len(phi_3d)):
    for j in range(len(T_3d)):
        E_star_mesh[i, j] = lambert_w_extremum(phi_mesh[i, j], T_mesh[i, j]) / eV_to_J

contour = ax3.contourf(T_mesh, phi_mesh / eV_to_J, E_star_mesh, levels=20, cmap='viridis')
ax3.set_xlabel('Temperature T (K)', fontsize=12)
ax3.set_ylabel('Chemical Potential φ (eV)', fontsize=12)
ax3.set_title('(c) E* Contour Map', fontsize=14, fontweight='bold')
plt.colorbar(contour, ax=ax3, label='E* (eV)')

# Plot 4: Lambert W argument behavior
phi_range_fine = np.linspace(-0.3, 0.3, 300) * eV_to_J
T_ref = 300  # K

# Calculate Lambert W argument
w_args = [0.5 * np.exp(phi / (k_B * T_ref) - 0.5) for phi in phi_range_fine]
w_vals = [lambertw(arg, k=0).real for arg in w_args]

ax4_twin = ax4.twinx()
line1 = ax4.plot(phi_range_fine / eV_to_J, w_args, 'b-', linewidth=2.5, label='W argument')
line2 = ax4_twin.plot(phi_range_fine / eV_to_J, w_vals, 'r-', linewidth=2.5, label='W value')

ax4.set_xlabel('Chemical Potential φ (eV)', fontsize=12)
ax4.set_ylabel('Lambert W Argument', fontsize=12, color='b')
ax4_twin.set_ylabel('Lambert W Value', fontsize=12, color='r')
ax4.set_title(f'(d) Lambert W Function Components (T = {T_ref} K)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='y', labelcolor='b')
ax4_twin.tick_params(axis='y', labelcolor='r')

# Add legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout(pad=3.0)

# Add main title
fig.suptitle('Lambert W Extremum Energy Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

plt.subplots_adjust(top=0.92)
plt.savefig("Carrier Concentration")
plt.show()

# Print some sample values
print("Sample calculations:")
print("=" * 50)
T_sample = 300  # K
phi_samples = [-0.1, 0, 0.1]  # eV

for phi_eV in phi_samples:
    phi = phi_eV * eV_to_J
    E_star = lambert_w_extremum(phi, T_sample)
    E_star_eV = E_star / eV_to_J
    
    # Also calculate the W argument and value
    w_arg = 0.5 * np.exp(phi / (k_B * T_sample) - 0.5)
    w_val = lambertw(w_arg, k=0).real
    
    print(f"φ = {phi_eV:6.2f} eV, T = {T_sample} K:")
    print(f"  W argument = {w_arg:.6f}")
    print(f"  W value    = {w_val:.6f}")
    print(f"  E*         = {E_star_eV:.6f} eV")
    print()