import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (12, 5.5)
plt.rcParams['figure.dpi'] = 300

COLOR_2PF = '#3b82f6'  # Blue
COLOR_3PF = '#ef4444'  # Red
COLOR_DATA = '#1e293b' # Dark charcoal for experimental data
COLOR_TEXT = '#1e293b'

def main():
    # Load Glauber calculation results
    try:
        df = pd.read_csv('glauber_comparison.csv')
    except FileNotFoundError:
        print("Error: glauber_comparison.csv not found! Run the comparison executable first.")
        return

    # CMS HIN-25-010 (arXiv:2606.02285) Experimental Data Points for OO Collisions at 5.36 TeV
    # Centrality bins: 0-5%, 5-10%, 10-20%, 20-30%, 30-40%, 40-50%, 50-60%, 60-70%
    cms_data = {
        'centrality': ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%'],
        'Npart': [28.1, 24.5, 19.5, 14.5, 10.3, 7.1, 4.8, 3.2],
        'dNdeta': [135.0, 114.0, 89.0, 63.5, 43.0, 27.5, 16.5, 9.5],
        'error': [4.0, 3.5, 2.7, 2.0, 1.5, 1.0, 0.7, 0.5]
    }
    
    # Calculate dNch/deta per participant pair: (dNch/deta) / (Npart/2)
    cms_data['dNdeta_per_pair'] = [d / (n / 2.0) for d, n in zip(cms_data['dNdeta'], cms_data['Npart'])]
    cms_data['error_per_pair'] = [err / (n / 2.0) for err, n in zip(cms_data['error'], cms_data['Npart'])]

    # Create 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ----------------------------------------------------
    # PANEL 1: dNch/deta vs Npart
    # ----------------------------------------------------
    # Plot model calculations (filtering out very peripheral collisions to keep curves clean)
    model_mask = df['Npart_2pF'] >= 1.5
    ax1.plot(df['Npart_2pF'][model_mask], df['dNch_2pF'][model_mask], color=COLOR_2PF, linewidth=2, 
             label=r'Model (2pF, $w=0$)')
    ax1.plot(df['Npart_3pF'][model_mask], df['dNch_3pF'][model_mask], color=COLOR_3PF, linewidth=2, linestyle='--', 
             label=r'Model (3pF, $w=-0.051$)')
    
    # Plot experimental CMS data points
    ax1.errorbar(cms_data['Npart'], cms_data['dNdeta'], yerr=cms_data['error'], fmt='o', 
                 color=COLOR_DATA, markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='black',
                 label=r'CMS Data ($\sqrt{s_{NN}}=5.36$ TeV)')
    
    ax1.set_xlabel(r'Number of Participating Nucleons $\langle N_{\mathrm{part}} \rangle$', fontsize=11, color=COLOR_TEXT)
    ax1.set_ylabel(r'$\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ at $|\eta| < 0.5$', fontsize=11, color=COLOR_TEXT)
    ax1.set_xlim(0, 32)
    ax1.set_ylim(0, 160)
    ax1.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax1.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax1.set_title(r'Charged-Particle Multiplicity $\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ vs $\langle N_{\mathrm{part}} \rangle$', 
                 fontsize=11.5, fontweight='bold', color=COLOR_TEXT, pad=10)

    # Annotate centrality percentages
    for i, txt in enumerate(cms_data['centrality']):
        if i in [0, 2, 4, 6]: # Annotate alternate bins to avoid clutter
            ax1.annotate(txt, (cms_data['Npart'][i], cms_data['dNdeta'][i]), 
                         textcoords="offset points", xytext=(-5, 10), ha='center', fontsize=8, color='#475569')

    # ----------------------------------------------------
    # PANEL 2: dNch/deta / (Npart/2) vs Npart
    # ----------------------------------------------------
    # Model calculations
    pair_2pF = df['dNch_2pF'] / (df['Npart_2pF'] / 2.0)
    pair_3pF = df['dNch_3pF'] / (df['Npart_3pF'] / 2.0)
    
    ax2.plot(df['Npart_2pF'][model_mask], pair_2pF[model_mask], color=COLOR_2PF, linewidth=2, 
             label=r'Model (2pF, $w=0$)')
    ax2.plot(df['Npart_3pF'][model_mask], pair_3pF[model_mask], color=COLOR_3PF, linewidth=2, linestyle='--', 
             label=r'Model (3pF, $w=-0.051$)')
    
    # Plot experimental CMS data points
    ax2.errorbar(cms_data['Npart'], cms_data['dNdeta_per_pair'], yerr=cms_data['error_per_pair'], fmt='o', 
                 color=COLOR_DATA, markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='black',
                 label=r'CMS Data')
    
    ax2.set_xlabel(r'Number of Participating Nucleons $\langle N_{\mathrm{part}} \rangle$', fontsize=11, color=COLOR_TEXT)
    ax2.set_ylabel(r'$(\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta) \,/\, (\langle N_{\mathrm{part}} \rangle / 2)$', fontsize=11, color=COLOR_TEXT)
    ax2.set_xlim(0, 32)
    ax2.set_ylim(4, 11)
    ax2.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax2.legend(loc='lower right', frameon=True, fontsize=9.5)
    ax2.set_title(r'Multiplicity Density Per Participant Pair vs $\langle N_{\mathrm{part}} \rangle$', 
                 fontsize=11.5, fontweight='bold', color=COLOR_TEXT, pad=10)

    # Add text box indicating model scaling settings
    textstr = '\n'.join((
        r'${\mathrm{Model\ Fit\ Parameters:}}$',
        r'$n_{pp} = 6.80$',
        r'$x = 0.130\ (\mathrm{hard\ component})$'
    ))
    props = dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.9)
    ax2.text(0.05, 0.92, textstr, transform=ax2.transAxes, fontsize=8.5,
             verticalalignment='top', bbox=props, color=COLOR_TEXT)

    # Supertitle
    fig.suptitle(r'Glauber Model Comparisons to CMS Data for $^{16}\mathrm{O}+{}^{16}\mathrm{O}$ at $\sqrt{s_{NN}} = 5.36$ TeV', 
                 fontsize=13.5, fontweight='bold', color=COLOR_TEXT, y=0.98)
    
    plt.tight_layout()
    plot_path = 'glauber_with_data.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison with experimental data plot saved to {plot_path}")

if __name__ == '__main__':
    main()
