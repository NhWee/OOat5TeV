import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set premium styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300

# Color palette (modern scientific dark-on-light theme)
COLOR_NPART = '#2b7bba' # Deep blue
COLOR_NCOLL = '#e05a47' # Coral red
COLOR_TEXT = '#2c3e50'
COLOR_GRID = '#ecf0f1'

def main():
    # 1. Read data from CSV
    try:
        df = pd.read_csv('glauber_results.csv')
    except FileNotFoundError:
        print("Error: glauber_results.csv not found! Run the C++ executable first.")
        return

    # 2. Create the plot
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot N_part on the left y-axis
    line1, = ax1.plot(df['b'], df['Npart'], color=COLOR_NPART, linewidth=2.5, 
                      label=r'Number of Participants $\langle N_{\mathrm{part}} \rangle$')
    ax1.set_xlabel('Impact Parameter $b$ (fm)', color=COLOR_TEXT, fontsize=12, labelpad=10)
    ax1.set_ylabel(r'$\langle N_{\mathrm{part}} \rangle$', color=COLOR_NPART, fontsize=12, labelpad=10)
    ax1.tick_params(axis='y', labelcolor=COLOR_NPART, labelsize=10)
    ax1.tick_params(axis='x', labelcolor=COLOR_TEXT, labelsize=10)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 32) # Oxygen + Oxygen max = 32

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    line2, = ax2.plot(df['b'], df['Ncoll'], color=COLOR_NCOLL, linewidth=2.5, linestyle='--',
                      label=r'Number of Binary Collisions $\langle N_{\mathrm{coll}} \rangle$')
    ax2.set_ylabel(r'$\langle N_{\mathrm{coll}} \rangle$', color=COLOR_NCOLL, fontsize=12, labelpad=10)
    ax2.tick_params(axis='y', labelcolor=COLOR_NCOLL, labelsize=10)
    ax2.set_ylim(0, 60) # Central OO collisions have ~53.3 binary collisions

    # Title & Subtitle
    plt.title('Glauber Model for $^{16}\\mathrm{O} + {}^{16}\\mathrm{O}$ Collisions at $\\sqrt{s_{NN}} = 5.36$ TeV', 
              fontsize=14, color=COLOR_TEXT, fontweight='bold', pad=20)

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)

    # Grid styling
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5, color='#cbd5e1')
    ax2.grid(False) # Turn off grid on secondary y-axis to prevent overlapping grid lines

    # Annotations to highlight key parameters
    npart_central = df.iloc[0]["Npart"]
    ncoll_central = df.iloc[0]["Ncoll"]
    annotation_text = (
        f"Central Collisions (b=0):\n"
        fr"$\langle N_{{\mathrm{{part}}}} \rangle \approx {npart_central:.1f}$\n"
        fr"$\langle N_{{\mathrm{{coll}}}} \rangle \approx {ncoll_central:.1f}$"
    )
    ax1.annotate(annotation_text, 
                 xy=(0.1, npart_central), 
                 xytext=(1.5, 27),
                 arrowprops=dict(facecolor='#64748b', shrink=0.08, width=1, headwidth=6, headlength=6),
                 fontsize=9, color=COLOR_TEXT, 
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#e2e8f0", lw=1))

    # Add text box for physics constants used
    textstr = '\n'.join((
        r'${\mathrm{Nucleus:}}\ ^{16}\mathrm{O}\ (A=16)$',
        r'$R = 2.608\ \mathrm{fm},\ a = 0.513\ \mathrm{fm}$',
        r'$\sigma_{\mathrm{in}}^{NN} = 70.0\ \mathrm{mb}$'
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.9)
    ax1.text(0.05, 0.08, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props, color=COLOR_TEXT)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = 'glauber_plot.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot successfully saved to {plot_path}")

if __name__ == '__main__':
    main()
