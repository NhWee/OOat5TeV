import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (12, 5.5)
plt.rcParams['figure.dpi'] = 300

COLOR_2PF = '#3b82f6'  # Blue
COLOR_3PF = '#ef4444'  # Red
COLOR_DATA = '#1e293b' # Charcoal for CMS data
COLOR_TEXT = '#1e293b'

# CMS HIN-25-010 (arXiv:2606.02285) Experimental Data Points for OO at 5.36 TeV
cms_data = {
    'Npart': np.array([28.1, 24.5, 19.5, 14.5, 10.3, 7.1, 4.8, 3.2]),
    'dNdeta': np.array([135.0, 114.0, 89.0, 63.5, 43.0, 27.5, 16.5, 9.5]),
    'error': np.array([4.0, 3.5, 2.7, 2.0, 1.5, 1.0, 0.7, 0.5])
}

# Calculated pair multiplicity and error
cms_data['dNdeta_per_pair'] = cms_data['dNdeta'] / (cms_data['Npart'] / 2.0)
cms_data['error_per_pair'] = cms_data['error'] / (cms_data['Npart'] / 2.0)

def main():
    # 1. Load C++ Glauber simulation results
    try:
        df = pd.read_csv('glauber_comparison.csv')
    except FileNotFoundError:
        print("Error: glauber_comparison.csv not found!")
        return

    # We need to interpolate Ncoll as a function of Npart from the Glauber tables
    # Since Npart decreases as b increases, we sort the arrays to ensure monotonic x for interpolation
    sort_idx_2 = np.argsort(df['Npart_2pF'])
    sort_idx_3 = np.argsort(df['Npart_3pF'])
    
    interp_Ncoll_2pF = interp1d(df['Npart_2pF'].iloc[sort_idx_2], df['Ncoll_2pF'].iloc[sort_idx_2], kind='linear', fill_value="extrapolate")
    interp_Ncoll_3pF = interp1d(df['Npart_3pF'].iloc[sort_idx_3], df['Ncoll_3pF'].iloc[sort_idx_3], kind='linear', fill_value="extrapolate")

    # 2. Define the fit model function: dNch/deta = n_pp * [ (1-x)*Npart/2 + x*Ncoll(Npart) ]
    # We fit parameters: p[0] = n_pp, p[1] = x
    def fit_model_2pF(Npart_val, n_pp, x):
        Ncoll_val = interp_Ncoll_2pF(Npart_val)
        return n_pp * ((1.0 - x) * Npart_val / 2.0 + x * Ncoll_val)

    def fit_model_3pF(Npart_val, n_pp, x):
        Ncoll_val = interp_Ncoll_3pF(Npart_val)
        return n_pp * ((1.0 - x) * Npart_val / 2.0 + x * Ncoll_val)

    # 3. Perform Curve Fitting using SciPy
    # Initial guess: n_pp = 6.0, x = 0.15
    # Bounds: n_pp in [3.0, 10.0], x in [0.0, 0.5]
    popt_2, pcov_2 = curve_fit(fit_model_2pF, cms_data['Npart'], cms_data['dNdeta'], 
                               sigma=cms_data['error'], p0=[6.0, 0.15], bounds=([3.0, 0.0], [10.0, 0.5]), absolute_sigma=True)
    
    popt_3, pcov_3 = curve_fit(fit_model_3pF, cms_data['Npart'], cms_data['dNdeta'], 
                               sigma=cms_data['error'], p0=[6.0, 0.15], bounds=([3.0, 0.0], [10.0, 0.5]), absolute_sigma=True)

    # Calculate standard errors of fitted parameters
    perr_2 = np.sqrt(np.diag(pcov_2))
    perr_3 = np.sqrt(np.diag(pcov_3))

    # Print results
    print("=== FITTING RESULTS ===")
    print("2-parameter Fermi (2pF, w=0) Fit:")
    print(f"  n_pp = {popt_2[0]:.4f} +/- {perr_2[0]:.4f}")
    print(f"  x    = {popt_2[1]:.4f} +/- {perr_2[1]:.4f}")
    
    # Calculate chi-square/ndf
    y_fit_2 = fit_model_2pF(cms_data['Npart'], *popt_2)
    chi2_2 = np.sum(((cms_data['dNdeta'] - y_fit_2) / cms_data['error'])**2)
    ndf = len(cms_data['Npart']) - 2
    print(f"  Chi2/ndf = {chi2_2:.3f} / {ndf} = {chi2_2/ndf:.3f}")

    print("\n3-parameter Fermi (3pF, w=-0.051) Fit:")
    print(f"  n_pp = {popt_3[0]:.4f} +/- {perr_3[0]:.4f}")
    print(f"  x    = {popt_3[1]:.4f} +/- {perr_3[1]:.4f}")
    
    y_fit_3 = fit_model_3pF(cms_data['Npart'], *popt_3)
    chi2_3 = np.sum(((cms_data['dNdeta'] - y_fit_3) / cms_data['error'])**2)
    print(f"  Chi2/ndf = {chi2_3:.3f} / {ndf} = {chi2_3/ndf:.3f}")

    # 4. Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Evaluate fits on a fine Npart grid
    Npart_grid = np.linspace(1.5, 32, 100)
    
    # Panel 1: dNch/deta vs Npart
    ax1.plot(Npart_grid, fit_model_2pF(Npart_grid, *popt_2), color=COLOR_2PF, linewidth=2, 
             label=fr'2pF Best Fit ($n_{{pp}}={popt_2[0]:.2f}, x={popt_2[1]:.2f}$)')
    ax1.plot(Npart_grid, fit_model_3pF(Npart_grid, *popt_3), color=COLOR_3PF, linewidth=2, linestyle='--', 
             label=fr'3pF Best Fit ($n_{{pp}}={popt_3[0]:.2f}, x={popt_3[1]:.2f}$)')
    ax1.errorbar(cms_data['Npart'], cms_data['dNdeta'], yerr=cms_data['error'], fmt='o', 
                 color=COLOR_DATA, markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='black',
                 label=r'CMS Data (OO 5.36 TeV)')
    
    ax1.set_xlabel(r'Number of Participating Nucleons $\langle N_{\mathrm{part}} \rangle$', fontsize=11, color=COLOR_TEXT)
    ax1.set_ylabel(r'$\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ at $|\eta| < 0.5$', fontsize=11, color=COLOR_TEXT)
    ax1.set_xlim(0, 32)
    ax1.set_ylim(0, 160)
    ax1.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax1.legend(loc='upper left', frameon=True, fontsize=9.2)
    ax1.set_title(r'Best Fit: $\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ vs $\langle N_{\mathrm{part}} \rangle$', 
                 fontsize=11.5, fontweight='bold', color=COLOR_TEXT, pad=10)

    # Panel 2: dNch/deta / (Npart/2) vs Npart
    pair_2pF_fit = fit_model_2pF(Npart_grid, *popt_2) / (Npart_grid / 2.0)
    pair_3pF_fit = fit_model_3pF(Npart_grid, *popt_3) / (Npart_grid / 2.0)
    
    ax2.plot(Npart_grid, pair_2pF_fit, color=COLOR_2PF, linewidth=2, label='2pF Best Fit')
    ax2.plot(Npart_grid, pair_3pF_fit, color=COLOR_3PF, linewidth=2, linestyle='--', label='3pF Best Fit')
    ax2.errorbar(cms_data['Npart'], cms_data['dNdeta_per_pair'], yerr=cms_data['error_per_pair'], fmt='o', 
                 color=COLOR_DATA, markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='black',
                 label=r'CMS Data')
    
    ax2.set_xlabel(r'Number of Participating Nucleons $\langle N_{\mathrm{part}} \rangle$', fontsize=11, color=COLOR_TEXT)
    ax2.set_ylabel(r'$(\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta) \,/\, (\langle N_{\mathrm{part}} \rangle / 2)$', fontsize=11, color=COLOR_TEXT)
    ax2.set_xlim(0, 32)
    ax2.set_ylim(4, 11)
    ax2.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax2.legend(loc='lower right', frameon=True, fontsize=9.5)
    ax2.set_title(r'Best Fit: Multiplicity Density Per Participant Pair vs $\langle N_{\mathrm{part}} \rangle$', 
                 fontsize=11.5, fontweight='bold', color=COLOR_TEXT, pad=10)

    # Add text box with fit results summary
    textstr = '\n'.join((
        r'${\mathrm{3pF\ Fit\ Results:}}$',
        fr'$n_{{pp}} = {popt_3[0]:.2f} \pm {perr_3[0]:.2f}$',
        fr'$x = {popt_3[1]:.3f} \pm {perr_3[1]:.3f}$',
        fr'$\chi^2/\mathrm{{ndf}} = {chi2_3/ndf:.2f}$'
    ))
    props = dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.95)
    ax2.text(0.05, 0.92, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props, color=COLOR_TEXT)

    # Supertitle
    fig.suptitle(r'Two-Component Model Fitting to CMS Data for OO Collisions at $\sqrt{s_{NN}} = 5.36$ TeV', 
                 fontsize=13, fontweight='bold', color=COLOR_TEXT, y=0.98)
    
    plt.tight_layout()
    plot_path = 'glauber_fit_results.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Fit results plot saved to {plot_path}")

if __name__ == '__main__':
    main()
