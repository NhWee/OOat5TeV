import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
import shutil

# Set premium visualization style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 300

COLOR_TEXT = '#1e293b'
COLOR_GRID = '#cbd5e1'

# Define density functions
def density_fermi(r, R, a, w=0.0):
    return (1.0 + w * (r / R)**2) / (1.0 + np.exp((r - R) / a))

def density_ho(r, a_ho, alpha):
    return (1.0 + alpha * (r / a_ho)**2) * np.exp(-(r / a_ho)**2)

def get_normalized_density(func, Z, *args):
    def integrand(r):
        return 4.0 * np.pi * r**2 * func(r, *args)
    integral, _ = quad(integrand, 0, 30)
    rho0 = Z / integral
    return lambda r: rho0 * func(r, *args), rho0

def trapezoid_integration(y, x):
    return np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x))

def main():
    extracted_dir = r"C:\Users\Administrator\.gemini\antigravity\scratch\extracted"
    
    # ----------------------------------------------------
    # Load and Correct Extracted Curves (Scale by 0.10 / 0.12 = 5/6)
    # ----------------------------------------------------
    correction_factor = 5.0 / 6.0
    
    # Curve 1 is the Sick & McCarthy experimental data for Oxygen-16
    data_o16_exp = np.loadtxt(os.path.join(extracted_dir, "odensity_curve_1.txt"))
    r_exp_o16 = data_o16_exp[:, 0]
    # Apply the Y-scale correction factor to get the correct physical density values
    val_exp_o16 = data_o16_exp[:, 1] * correction_factor
    
    # Curve 4 is the exact Osat (NNLOsat) profile from the paper
    data_osat = np.loadtxt(os.path.join(extracted_dir, "odensity_curve_4.txt"))
    r_osat = data_osat[:, 0]
    # Apply the Y-scale correction factor
    val_osat = data_osat[:, 1] * correction_factor

    # ----------------------------------------------------
    # Compute Volume Integrals of Corrected Curves
    # ----------------------------------------------------
    int_exp_o16 = 4.0 * np.pi * trapezoid_integration(r_exp_o16**2 * val_exp_o16, r_exp_o16)
    int_osat = 4.0 * np.pi * trapezoid_integration(r_osat**2 * val_osat, r_osat)
    
    print(f"Corrected Experimental Curve Volume Integral: Z = {int_exp_o16:.4f} (expected ~7.8 due to tail cut-off)")
    print(f"Corrected Osat Curve Volume Integral: Z = {int_osat:.4f} (expected ~7.8 due to tail cut-off)")

    # ----------------------------------------------------
    # Define Theoretical Analytical Models (Normalized to Z=8)
    # ----------------------------------------------------
    rho_opar, _ = get_normalized_density(density_fermi, 8.0, 2.608, 0.513, -0.051)
    rho_opar2, _ = get_normalized_density(density_fermi, 8.0, 1.850, 0.497, 0.912)
    rho_oho, _ = get_normalized_density(density_ho, 8.0, 1.833, 1.544)
    rho_oho2, _ = get_normalized_density(density_ho, 8.0, 1.819, 1.506)

    # ----------------------------------------------------
    # Plotting (Single-panel for Oxygen-16)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 8.5))
    r_grid = np.linspace(0, 7.0, 400)

    # Plot experimental data and Osat (using raw corrected curves, no manual scaling needed!)
    plt.plot(r_exp_o16, val_exp_o16, color='#0f172a', linewidth=3.0, 
             label='Experimental FB (Sick & McCarthy 1970)')
    plt.plot(r_osat, val_osat, color='#10b981', linewidth=2.5, 
             label='Osat (exact ab-initio NNLOsat profile)')

    # Plot analytical curves
    plt.plot(r_grid, rho_opar(r_grid), color='#ef4444', linewidth=2.0, linestyle='-',
             label=r'Odat/Opar (3pF Standard, $R=2.608, a=0.513, w=-0.051$)')
    plt.plot(r_grid, rho_opar2(r_grid), color='#3b82f6', linewidth=2.0, linestyle='--',
             label=r'Opar2 (3pF fit to NNLOsat, $R=1.85, a=0.497, w=0.912$)')
    plt.plot(r_grid, rho_oho(r_grid), color='#f59e0b', linewidth=2.0, linestyle='-.',
             label=r'Oho (HO de Vries, $a_{\mathrm{HO}}=1.833, \alpha=1.544$)')
    plt.plot(r_grid, rho_oho2(r_grid), color='#8b5cf6', linewidth=2.2, linestyle=':',
             label=r'Oho2 (HO new fit to data, $a_{\mathrm{HO}}=1.819, \alpha=1.506$)')

    # Formatting and styling
    plt.title(r'$^{16}\mathrm{O}$ Nuclear Charge Density distributions $\rho_c(r)$', fontsize=15, fontweight='bold', color=COLOR_TEXT, pad=15)
    plt.xlabel('Radius $r$ (fm)', fontsize=12, color=COLOR_TEXT)
    plt.ylabel(r'Charge Density $\rho_c(r)$ ($e/\mathrm{fm}^3$)', fontsize=12, color=COLOR_TEXT)
    plt.xlim(0, 6.5)
    plt.ylim(0, 0.10) # Matches the 0.10 maximum limit of the paper's axis
    plt.grid(True, linestyle=':', linewidth=0.5, color=COLOR_GRID)
    plt.legend(loc='upper right', fontsize=9.5, frameon=True, edgecolor='#cbd5e1', facecolor='#f8fafc')

    # Add explanatory annotation box
    annotation_text = '\n'.join((
        r'$\mathbf{Physical\ Observations:}$',
        r'• The experimental data has a central dip (depletion) at $r < 1\ \mathrm{fm}$',
        r'  and peaks at $r \approx 1\ \mathrm{fm}$ before decaying at the surface.',
        r'• The HO model ($\mathbf{Oho2}$) and ab-initio profile ($\mathbf{Osat}$)',
        r'  reconstruct this central depletion shell structure beautifully.',
        r'• Standard $\mathbf{Odat/Opar}$ (red) lacks central depletion because',
        r'  of negative $w = -0.051$, peaking monotonically at $r = 0$.'
    ))
    props = dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.95)
    plt.gca().text(0.04, 0.04, annotation_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='bottom', bbox=props, color=COLOR_TEXT)

    plt.tight_layout()
    plot_path = 'charge_density_comparison_final.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Saved corrected Oxygen-16 charge density comparison plot to {plot_path}")

    # Copy the generated plot to the brain directory as an artifact
    brain_plot_path = r"C:\Users\Administrator\.gemini\antigravity\brain\214b9532-0655-4a8b-a4dd-90743fe99aa7\charge_density_comparison_final.png"
    shutil.copy(plot_path, brain_plot_path)
    print("Copied final plot to brain directory.")

if __name__ == '__main__':
    main()
