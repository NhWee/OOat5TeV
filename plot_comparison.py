import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (16, 5)
plt.rcParams['figure.dpi'] = 300

COLOR_2PF = '#3b82f6'  # Blue
COLOR_3PF = '#ef4444'  # Red
COLOR_TEXT = '#1e293b'

def main():
    # Read data
    df_dens = pd.read_csv('density_comparison.csv')
    df_glaub = pd.read_csv('glauber_comparison.csv')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.2))
    
    # ----------------------------------------------------
    # PANEL 1: Density Profile \rho(r)
    # ----------------------------------------------------
    ax1.plot(df_dens['r'], df_dens['rho_2pF'], color=COLOR_2PF, linewidth=2.5, label='2-parameter Fermi (2pF, $w=0$)')
    ax1.plot(df_dens['r'], df_dens['rho_3pF'], color=COLOR_3PF, linewidth=2.5, linestyle='--', label='3-parameter Fermi (3pF, $w=-0.051$)')
    ax1.set_xlabel('Radius $r$ (fm)', fontsize=11, color=COLOR_TEXT)
    ax1.set_ylabel(r'Nucleon Density $\rho(r)$ ($\mathrm{fm}^{-3}$)', fontsize=11, color=COLOR_TEXT)
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 0.17)
    ax1.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    ax1.set_title('Nuclear Density Distribution', fontsize=12, fontweight='bold', color=COLOR_TEXT, pad=10)
    
    # Add annotation for central depletion
    ax1.annotate('Central depletion\ndue to $w < 0$', xy=(0.2, 0.148), xytext=(1.8, 0.12),
                 arrowprops=dict(facecolor='#64748b', shrink=0.08, width=0.5, headwidth=4, headlength=4),
                 fontsize=8.5, color=COLOR_TEXT)

    # ----------------------------------------------------
    # PANEL 2: Overlap Function T_{AB}(b)
    # ----------------------------------------------------
    ax2.plot(df_glaub['b'], df_glaub['Tab_2pF'], color=COLOR_2PF, linewidth=2.5, label='2pF')
    ax2.plot(df_glaub['b'], df_glaub['Tab_3pF'], color=COLOR_3PF, linewidth=2.5, linestyle='--', label='3pF')
    ax2.set_xlabel('Impact Parameter $b$ (fm)', fontsize=11, color=COLOR_TEXT)
    ax2.set_ylabel(r'Overlap Function $T_{AB}(b)$ ($\mathrm{fm}^{-2}$)', fontsize=11, color=COLOR_TEXT)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8)
    ax2.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax2.set_title('Nuclear Overlap Function', fontsize=12, fontweight='bold', color=COLOR_TEXT, pad=10)
    
    # Inset or twin axis to show relative difference: (3pF - 2pF)/2pF in %
    ax2_diff = ax2.twinx()
    diff_tab = 100.0 * (df_glaub['Tab_3pF'] - df_glaub['Tab_2pF']) / df_glaub['Tab_2pF']
    # Filter out division by near zero at large b
    mask = df_glaub['b'] <= 6.0
    ax2_diff.plot(df_glaub['b'][mask], diff_tab[mask], color='#6b7280', linewidth=1.5, linestyle=':', label='Relative Diff')
    ax2_diff.set_ylabel('Relative Difference (%)', color='#6b7280', fontsize=10, labelpad=10)
    ax2_diff.tick_params(axis='y', labelcolor='#6b7280')
    ax2_diff.set_ylim(-10, 2)
    ax2_diff.grid(False)
    
    # ----------------------------------------------------
    # PANEL 3: N_part & dNch/deta
    # ----------------------------------------------------
    line1, = ax3.plot(df_glaub['b'], df_glaub['Npart_2pF'], color=COLOR_2PF, linewidth=2.5, label=r'$N_{\mathrm{part}}$ (2pF)')
    line2, = ax3.plot(df_glaub['b'], df_glaub['Npart_3pF'], color=COLOR_2PF, linewidth=2.5, linestyle='--', label=r'$N_{\mathrm{part}}$ (3pF)')
    
    ax3_mult = ax3.twinx()
    line3, = ax3_mult.plot(df_glaub['b'], df_glaub['dNch_2pF'], color=COLOR_3PF, linewidth=2.5, label=r'$\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ (2pF)')
    line4, = ax3_mult.plot(df_glaub['b'], df_glaub['dNch_3pF'], color=COLOR_3PF, linewidth=2.5, linestyle='--', label=r'$\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ (3pF)')
    
    ax3.set_xlabel('Impact Parameter $b$ (fm)', fontsize=11, color=COLOR_TEXT)
    ax3.set_ylabel(r'Participant Nucleons $\langle N_{\mathrm{part}} \rangle$', fontsize=11, color=COLOR_2PF)
    ax3.tick_params(axis='y', labelcolor=COLOR_2PF)
    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 32)
    
    ax3_mult.set_ylabel(r'Multiplicity $\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$', fontsize=11, color=COLOR_3PF, labelpad=10)
    ax3_mult.tick_params(axis='y', labelcolor=COLOR_3PF)
    ax3_mult.set_ylim(0, 150)
    
    ax3.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax3_mult.grid(False)
    ax3.set_title('Participants & Particle Multiplicity', fontsize=12, fontweight='bold', color=COLOR_TEXT, pad=10)
    
    # Legend for Panel 3
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=8.5)
    
    # Title
    fig.suptitle(r'Comparison of 2pF vs 3pF Woods-Saxon for $^{16}\mathrm{O}+{}^{16}\mathrm{O}$ at $\sqrt{s_{NN}} = 5.36$ TeV', 
                 fontsize=14, fontweight='bold', color=COLOR_TEXT, y=0.98)
    
    plt.tight_layout()
    plot_path = 'glauber_comparison.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot successfully saved to {plot_path}")

if __name__ == '__main__':
    main()
