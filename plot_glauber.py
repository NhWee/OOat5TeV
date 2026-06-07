import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set premium styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 300

# Color palette
COLOR_NPART = '#2b7bba' # Deep blue
COLOR_NCOLL = '#e05a47' # Coral red
COLOR_MULT = '#10b981'  # Emerald green
COLOR_TEXT = '#2c3e50'

def main():
    # 1. Read data from CSV
    try:
        df = pd.read_csv('glauber_results.csv')
    except FileNotFoundError:
        print("Error: glauber_results.csv not found! Run the C++ executable first.")
        return

    # 2. Create a 2-panel figure
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6.5))

    # --- PANEL 1: N_part and N_coll ---
    # Plot N_part on the left y-axis of panel 1
    line1, = ax1.plot(df['b'], df['Npart'], color=COLOR_NPART, linewidth=2.5, 
                      label=r'Participants $\langle N_{\mathrm{part}} \rangle$')
    ax1.set_xlabel('Impact Parameter $b$ (fm)', color=COLOR_TEXT, fontsize=11, labelpad=10)
    ax1.set_ylabel(r'$\langle N_{\mathrm{part}} \rangle$', color=COLOR_NPART, fontsize=12, labelpad=10)
    ax1.tick_params(axis='y', labelcolor=COLOR_NPART, labelsize=10)
    ax1.tick_params(axis='x', labelcolor=COLOR_TEXT, labelsize=10)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 32)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5, color='#cbd5e1')

    # Plot N_coll on the right y-axis of panel 1
    ax2 = ax1.twinx()  
    line2, = ax2.plot(df['b'], df['Ncoll'], color=COLOR_NCOLL, linewidth=2.5, linestyle='--',
                      label=r'Binary Collisions $\langle N_{\mathrm{coll}} \rangle$')
    ax2.set_ylabel(r'$\langle N_{\mathrm{coll}} \rangle$', color=COLOR_NCOLL, fontsize=12, labelpad=10)
    ax2.tick_params(axis='y', labelcolor=COLOR_NCOLL, labelsize=10)
    ax2.set_ylim(0, 60)
    ax2.grid(False)

    # Combine legends for panel 1
    lines_p1 = [line1, line2]
    labels_p1 = [l.get_label() for l in lines_p1]
    ax1.legend(lines_p1, labels_p1, loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9)
    ax1.set_title('Nuclear Overlap Geometry', fontsize=12, color=COLOR_TEXT, fontweight='bold', pad=12)

    # Annotate central collision parameters
    npart_central = df.iloc[0]["Npart"]
    ncoll_central = df.iloc[0]["Ncoll"]
    mult_central = df.iloc[0]["dNch_deta"]
    
    ax1.annotate(f'b=0:\n'
                 fr'$\langle N_{{\mathrm{{part}}}} \rangle \approx {npart_central:.1f}$' + '\n'
                 fr'$\langle N_{{\mathrm{{coll}}}} \rangle \approx {ncoll_central:.1f}$', 
                 xy=(0.1, npart_central), 
                 xytext=(1.5, 25),
                 arrowprops=dict(facecolor='#64748b', shrink=0.08, width=0.5, headwidth=5, headlength=5),
                 fontsize=8.5, color=COLOR_TEXT, 
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#e2e8f0", lw=1))

    # --- PANEL 2: Charged Particle Multiplicity dNch/deta ---
    ax3.plot(df['b'], df['dNch_deta'], color=COLOR_MULT, linewidth=2.8, 
             label=r'Multiplicity $\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$')
    ax3.set_xlabel('Impact Parameter $b$ (fm)', color=COLOR_TEXT, fontsize=11, labelpad=10)
    ax3.set_ylabel(r'$\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ at $|\eta| < 0.5$', color=COLOR_TEXT, fontsize=12, labelpad=10)
    ax3.tick_params(axis='both', labelcolor=COLOR_TEXT, labelsize=10)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 150)
    ax3.grid(True, which='both', linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax3.set_title('Charged-Particle Production', fontsize=12, color=COLOR_TEXT, fontweight='bold', pad=12)

    # Annotate experimental match (CMS data at b=0)
    ax3.annotate(f'Central OO (b=0):\n'
                 fr'$\mathrm{{d}}N_{{\mathrm{{ch}}}}/\mathrm{{d}}\eta \approx {mult_central:.1f}$' + '\n'
                 '(Tuned to CMS 5.36 TeV)', 
                 xy=(0.15, mult_central), 
                 xytext=(1.5, 115),
                 arrowprops=dict(facecolor='#64748b', shrink=0.08, width=0.5, headwidth=5, headlength=5),
                 fontsize=8.5, color=COLOR_TEXT, 
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#e2e8f0", lw=1))

    # Add text box for physics constants used in the two-component model
    textstr = '\n'.join((
        r'${\mathrm{Two\text{-}Component\ Model:}}$',
        r'$\frac{\mathrm{d}N_{\mathrm{ch}}}{\mathrm{d}\eta} = n_{pp} \left[ (1 - x) \frac{\langle N_{\mathrm{part}} \rangle}{2} + x \langle N_{\mathrm{coll}} \rangle \right]$',
        r'$n_{pp} = 6.80\ (\text{pp Multiplicity})$',
        r'$x = 0.130\ (\text{Hard fraction})$',
        r'$\sigma_{\mathrm{in}}^{NN} = 70.0\ \mathrm{mb}$'
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.9)
    ax3.text(0.35, 0.05, textstr, transform=ax3.transAxes, fontsize=8.5,
             verticalalignment='bottom', bbox=props, color=COLOR_TEXT)

    # Overall Figure Title
    fig.suptitle('Glauber Model & Particle Production for $^{16}\\mathrm{O} + {}^{16}\\mathrm{O}$ at $\\sqrt{s_{NN}} = 5.36$ TeV', 
                 fontsize=15, color=COLOR_TEXT, fontweight='bold', y=0.98)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = 'glauber_plot.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot successfully saved to {plot_path}")

if __name__ == '__main__':
    main()
