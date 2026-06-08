import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (16, 5.5)
plt.rcParams['figure.dpi'] = 300

COLOR_2PF = '#3b82f6'  # Blue
COLOR_3PF = '#ef4444'  # Red
COLOR_HO = '#8b5cf6'   # Purple
COLOR_TEXT = '#1e293b'

def main():
    # Read data
    df_dens = pd.read_csv('density_comparison.csv')
    df_glaub = pd.read_csv('glauber_comparison.csv')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.6))
    
    # ----------------------------------------------------
    # PANEL 1: Density Profile \rho(r)
    # ----------------------------------------------------
    ax1.plot(df_dens['r'], df_dens['rho_user'], color=COLOR_2PF, linewidth=2.5, 
             label=r'2pF UserWS ($R=2.82, a=0.546, w=0$)')
    ax1.plot(df_dens['r'], df_dens['rho_opar'], color=COLOR_3PF, linewidth=2.5, linestyle='--', 
             label=r'3pF Opar ($R=2.608, a=0.513, w=-0.051$)')
    ax1.plot(df_dens['r'], df_dens['rho_oho2'], color=COLOR_HO, linewidth=2.5, linestyle=':', 
             label=r'HO Oho2 ($a_{\mathrm{HO}}=1.819, \alpha=1.506$)')
    
    ax1.set_xlabel('Radius $r$ (fm)', fontsize=11, color=COLOR_TEXT)
    ax1.set_ylabel(r'Nucleon Density $\rho(r)$ ($\mathrm{fm}^{-3}$)', fontsize=11, color=COLOR_TEXT)
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 0.18)
    ax1.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax1.legend(loc='upper right', frameon=True, fontsize=8.5, facecolor='#f8fafc', edgecolor='#cbd5e1')
    ax1.set_title('Nuclear Density Distribution', fontsize=12, fontweight='bold', color=COLOR_TEXT, pad=10)
    
    # Add annotations
    ax1.annotate('Central depletion\nin HO Oho2', xy=(0.3, 0.147), xytext=(1.8, 0.155),
                 arrowprops=dict(facecolor='#64748b', shrink=0.08, width=0.5, headwidth=4, headlength=4),
                 fontsize=8.5, color=COLOR_TEXT)
    ax1.annotate('Peak due to negative $w$\nin 3pF Opar', xy=(0.05, 0.165), xytext=(1.2, 0.11),
                 arrowprops=dict(facecolor='#64748b', shrink=0.08, width=0.5, headwidth=4, headlength=4),
                 fontsize=8.5, color=COLOR_TEXT)

    # ----------------------------------------------------
    # PANEL 2: Overlap Function T_{AB}(b)
    # ----------------------------------------------------
    ax2.plot(df_glaub['b'], df_glaub['Tab_user'], color=COLOR_2PF, linewidth=2.5, label='2pF UserWS')
    ax2.plot(df_glaub['b'], df_glaub['Tab_opar'], color=COLOR_3PF, linewidth=2.5, linestyle='--', label='3pF Opar')
    ax2.plot(df_glaub['b'], df_glaub['Tab_oho2'], color=COLOR_HO, linewidth=2.5, linestyle=':', label='HO Oho2')
    
    ax2.set_xlabel('Impact Parameter $b$ (fm)', fontsize=11, color=COLOR_TEXT)
    ax2.set_ylabel(r'Overlap Function $T_{AB}(b)$ ($\mathrm{fm}^{-2}$)', fontsize=11, color=COLOR_TEXT)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8.5)
    ax2.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax2.legend(loc='upper right', frameon=True, fontsize=8.5, facecolor='#f8fafc', edgecolor='#cbd5e1')
    ax2.set_title('Nuclear Overlap Function', fontsize=12, fontweight='bold', color=COLOR_TEXT, pad=10)
    
    # Twin axis to show relative difference: (Model - 2pF_UserWS)/2pF_UserWS in %
    ax2_diff = ax2.twinx()
    diff_opar = 100.0 * (df_glaub['Tab_opar'] - df_glaub['Tab_user']) / df_glaub['Tab_user']
    diff_oho2 = 100.0 * (df_glaub['Tab_oho2'] - df_glaub['Tab_user']) / df_glaub['Tab_user']
    
    # Filter out division by near zero at large b
    mask = df_glaub['b'] <= 6.0
    ax2_diff.plot(df_glaub['b'][mask], diff_opar[mask], color=COLOR_3PF, linewidth=1.2, linestyle='--', alpha=0.6)
    ax2_diff.plot(df_glaub['b'][mask], diff_oho2[mask], color=COLOR_HO, linewidth=1.2, linestyle=':', alpha=0.6)
    ax2_diff.set_ylabel('Diff vs 2pF (%)', color='#64748b', fontsize=10, labelpad=10)
    ax2_diff.tick_params(axis='y', labelcolor='#64748b')
    ax2_diff.set_ylim(-15, 5)
    ax2_diff.grid(False)
    
    # ----------------------------------------------------
    # PANEL 3: N_part & dNch/deta
    # ----------------------------------------------------
    # Plot N_part on left axis
    line1, = ax3.plot(df_glaub['b'], df_glaub['Npart_user'], color=COLOR_2PF, linewidth=2.2, label=r'$N_{\mathrm{part}}$ (2pF)')
    line2, = ax3.plot(df_glaub['b'], df_glaub['Npart_opar'], color=COLOR_2PF, linewidth=2.2, linestyle='--', label=r'$N_{\mathrm{part}}$ (3pF)')
    line3, = ax3.plot(df_glaub['b'], df_glaub['Npart_oho2'], color=COLOR_2PF, linewidth=2.2, linestyle=':', label=r'$N_{\mathrm{part}}$ (HO)')
    
    # Plot multiplicity on right axis
    ax3_mult = ax3.twinx()
    line4, = ax3_mult.plot(df_glaub['b'], df_glaub['dNch_user'], color=COLOR_3PF, linewidth=2.2, label=r'$\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ (2pF)')
    line5, = ax3_mult.plot(df_glaub['b'], df_glaub['dNch_opar'], color=COLOR_3PF, linewidth=2.2, linestyle='--', label=r'$\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ (3pF)')
    line6, = ax3_mult.plot(df_glaub['b'], df_glaub['dNch_oho2'], color=COLOR_3PF, linewidth=2.2, linestyle=':', label=r'$\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$ (HO)')
    
    ax3.set_xlabel('Impact Parameter $b$ (fm)', fontsize=11, color=COLOR_TEXT)
    ax3.set_ylabel(r'Participant Nucleons $\langle N_{\mathrm{part}} \rangle$', fontsize=11, color=COLOR_2PF)
    ax3.tick_params(axis='y', labelcolor=COLOR_2PF)
    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 34)
    
    ax3_mult.set_ylabel(r'Multiplicity $\mathrm{d}N_{\mathrm{ch}}/\mathrm{d}\eta$', fontsize=11, color=COLOR_3PF, labelpad=10)
    ax3_mult.tick_params(axis='y', labelcolor=COLOR_3PF)
    ax3_mult.set_ylim(0, 160)
    
    ax3.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax3_mult.grid(False)
    ax3.set_title('Participants & Particle Multiplicity', fontsize=12, fontweight='bold', color=COLOR_TEXT, pad=10)
    
    # Legend for Panel 3
    lines = [line1, line2, line3, line4, line5, line6]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=8)
    
    # Title
    fig.suptitle(r'Comparison of 2pF, 3pF, and HO Profiles for $^{16}\mathrm{O}+{}^{16}\mathrm{O}$ Glauber Simulation at $\sqrt{s_{NN}} = 5.36$ TeV', 
                 fontsize=14, fontweight='bold', color=COLOR_TEXT, y=0.98)
    
    plt.tight_layout()
    plot_path = 'glauber_comparison.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot successfully saved to {plot_path}")
    
    # Copy the plot to the brain directory as an artifact
    brain_plot_path = r"C:\Users\Administrator\.gemini\antigravity\brain\214b9532-0655-4a8b-a4dd-90743fe99aa7\glauber_comparison.png"
    shutil.copy(plot_path, brain_plot_path)
    print("Copied Glauber comparison plot to brain directory.")

if __name__ == '__main__':
    main()
