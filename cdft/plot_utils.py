import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import matplotlib.cm as cm

# Set up the color cycle
num_colors = 20
colors = cm.RdPu(np.linspace(0, 1, num_colors))
color_cycle = np.tile(colors, (int(np.ceil(1000000 / num_colors)), 1))[:1000000]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_cycle)

# Update matplotlib parameters
params = {"axes.labelsize": 14, "axes.titlesize": 15}
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams.update(params)
plt.rcParams['figure.dpi'] = 300



def configure_plot_charge(zbins):
    fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    ax[0,0].set_xlim(zbins[0], zbins[-1])
    ax[0,1].set_ylim(-30, 30)

    ax[0,0].set_ylabel(r'$\rho(z)$ [$\mathrm{\AA}^{-3}$]')
    ax[1,0].set_ylabel(r'$n(z)$ [$e\mathrm{\AA}^{-3}$]')
    ax[0,1].set_ylabel(r'$\beta[\mu - V_{\mathrm{ext}}(z)],   \beta e\phi(z)$')
    ax[1,1].set_ylabel(r'$\beta e \phi_{\mathrm{R}}(z)$')
    ax[1,0].set_xlabel(r'$z$ [$\mathrm{\AA}$]')
    ax[1,1].set_xlabel(r'$z$ [$\mathrm{\AA}$]')
    ax[0,1].yaxis.set_label_position("right")
    ax[0,1].yaxis.tick_right()
    ax[1,1].yaxis.set_label_position("right")
    ax[1,1].yaxis.tick_right()
    

    ax[0,0].grid(which="major", ls="dashed", dashes=(1, 3), lw=0.5, zorder=0)
    ax[0,1].grid(which="major", ls="dashed", dashes=(1, 3), lw=0.5, zorder=0)
    ax[1,0].grid(which="major", ls="dashed", dashes=(1, 3), lw=0.5, zorder=0)
    ax[1,1].grid(which="major", ls="dashed", dashes=(1, 3), lw=0.5, zorder=0)
    ax[0,0].tick_params(direction="in", which="major", length=5, labelsize=13)
    ax[0,1].tick_params(direction="in", which="major", length=5, labelsize=13)
    ax[1,0].tick_params(direction="in", which="major", length=5, labelsize=13)
    ax[1,1].tick_params(direction="in", which="major", length=5, labelsize=13)
    plt.tight_layout()
    return fig, ax


def plot_interactive_density_charge(fig, ax, zbins, rho, n, muloc, elec, lmf, color_count):
    display.clear_output(wait=True)
    ax[0,0].plot(zbins, rho, color=color_cycle[color_count])
    ax[1,0].plot(zbins, n, color=color_cycle[color_count])
    
    ax[0,1].plot(zbins, muloc, color=color_cycle[color_count])
    ax[0,1].plot(zbins, elec, color=color_cycle[color_count], ls='--')
    ax[1,1].plot(zbins, lmf, color=color_cycle[color_count])
    display.display(fig)
    
def plot_end_density_charge(zbins, rho, n, muloc, elec, lmf, ax):
    display.clear_output(wait=False)
    ax[0,0].plot(zbins, rho, color='black', lw=2)
    ax[1,0].plot(zbins, n, color='black', lw=2)
    ax[0,1].plot(zbins, muloc, color='black', lw=2)
    ax[0,1].plot(zbins, elec, ls='--', color='black', lw=2)
    ax[1,1].plot(zbins, lmf, color='black', lw=2)


