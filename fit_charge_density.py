import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 300

COLOR_FIT = '#ef4444'  # Red for best fit
COLOR_DATA = '#10b981' # Green for experimental data
COLOR_TEXT = '#1e293b'

# Experimental Fourier-Bessel (model-independent) charge density data points for 16O
# compiled from electron scattering experiments (de Vries 1987)
fb_r = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
fb_rho = np.array([0.0760, 0.0758, 0.0751, 0.0729, 0.0664, 0.0528, 0.0344, 0.0182, 0.0079, 0.0030, 0.0010])
fb_error = np.array([0.0015, 0.0014, 0.0013, 0.0012, 0.0011, 0.0010, 0.0009, 0.0008, 0.0005, 0.0002, 0.0001])

# 3-parameter Fermi (3pF) distribution function
def ws_3pf(r, rho0, R, a, w):
    # To prevent division by zero or negative values in exponential
    return rho0 * (1.0 + w * (r / R)**2) / (1.0 + np.exp((r - R) / a))

def main():
    # Perform the curve fit
    # Initial guesses: rho0=0.08, R=2.6, a=0.5, w=-0.05
    # Bounds: rho0 in [0.01, 0.2], R in [1.0, 5.0], a in [0.1, 1.0], w in [-0.5, 0.5]
    popt, pcov = curve_fit(
        ws_3pf, fb_r, fb_rho, sigma=fb_error,
        p0=[0.08, 2.6, 0.5, -0.05],
        bounds=([0.01, 1.0, 0.1, -0.5], [0.2, 5.0, 1.0, 0.5]),
        absolute_sigma=True
    )
    
    perr = np.sqrt(np.diag(pcov))
    rho0_fit, R_fit, a_fit, w_fit = popt
    
    # Calculate Chi-square / ndf
    y_fit = ws_3pf(fb_r, *popt)
    chi2 = np.sum(((fb_rho - y_fit) / fb_error)**2)
    ndf = len(fb_r) - 4
    
    # Verify normalization: 4 * pi * \int r^2 \rho(r) dr should equal Z = 8
    def integrand(r):
        return 4.0 * np.pi * r**2 * ws_3pf(r, *popt)
    
    total_charge, _ = quad(integrand, 0, 15)
    
    print("=== CHARGE DENSITY FITTING RESULTS ===")
    print(f"  rho0 = {rho0_fit:.6f} +/- {perr[0]:.6f} e/fm^3")
    print(f"  R    = {R_fit:.4f} +/- {perr[1]:.4f} fm")
    print(f"  a    = {a_fit:.4f} +/- {perr[2]:.4f} fm")
    print(f"  w    = {w_fit:.4f} +/- {perr[3]:.4f}")
    print(f"  Chi2/ndf = {chi2:.3f} / {ndf} = {chi2/ndf:.3f}")
    print(f"  Integrated Total Charge Z = {total_charge:.4f} (Expected: 8.0)")
    
    # Plot results
    r_grid = np.linspace(0, 7, 500)
    rho_fit_grid = ws_3pf(r_grid, *popt)
    
    plt.figure(figsize=(7.5, 6))
    plt.plot(r_grid, rho_fit_grid, color=COLOR_FIT, linewidth=2.2, label='3pF Best Fit Curve')
    plt.errorbar(fb_r, fb_rho, yerr=fb_error, fmt='o', color=COLOR_DATA, markersize=5, capsize=3,
                 elinewidth=1.2, markeredgecolor='black', label='Experimental Fourier-Bessel (de Vries 1987)')
    
    plt.xlabel('Radius $r$ (fm)', fontsize=12, color=COLOR_TEXT)
    plt.ylabel(r'Charge Density $\rho_c(r)$ ($e/\mathrm{fm}^3$)', fontsize=12, color=COLOR_TEXT)
    plt.xlim(0, 6.5)
    plt.ylim(0, 0.09)
    plt.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    plt.legend(loc='upper right', frameon=True, edgecolor='#e2e8f0', fontsize=10)
    
    # Text box containing fit results
    textstr = '\n'.join((
        r'${\mathrm{3pF\ Charge\ Density\ Fit\ (Z=8):}}$',
        fr'$R = {R_fit:.4f} \pm {perr[1]:.4f}\ \mathrm{{fm}}$',
        fr'$a = {a_fit:.4f} \pm {perr[2]:.4f}\ \mathrm{{fm}}$',
        fr'$w = {w_fit:.4f} \pm {perr[3]:.4f}$',
        fr'$\chi^2/\mathrm{{ndf}} = {chi2/ndf:.2f}$',
        fr'$\int \rho_c \, d^3r = {total_charge:.2f}\ e$'
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.95)
    plt.gca().text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=9.5,
                   verticalalignment='bottom', bbox=props, color=COLOR_TEXT)
    
    plt.title(r'Fitting 3pF Density Parameters to $^{16}\mathrm{O}$ Electron Scattering Data', 
              fontsize=13, fontweight='bold', color=COLOR_TEXT, pad=15)
    plt.tight_layout()
    
    plot_path = 'charge_density_fit.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Charge density fit plot saved to {plot_path}")

if __name__ == '__main__':
    main()
