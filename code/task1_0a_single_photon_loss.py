import dynamiqs as dq
import jax
import math
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
FIG = Path(__file__).resolve().parent.parent / "figures" / "task1_0"
FIG.mkdir(parents=True, exist_ok=True)

dq.set_device('cpu')
dq.set_precision('double')

class HarmonicOscillator:
    def __init__(self):
        self.N = 30
        self.two_pi = 2 * math.pi
        self.Delta = self.two_pi * 1.0       # rad/us  (Delta/2pi = 1 MHz)
        self.kappa = self.two_pi * 1.0     # rad/us  (kappa/2pi = 1 MHz, exaggerated)
        self.alpha0 = 2.0
        self.T = 1.0                         # us

        self.a = dq.destroy(self.N)
        self.ad = dq.dag(self.a)
        self.H = self.Delta * self.ad @ self.a
        self.jump_ops = [jnp.sqrt(self.kappa) * self.a]
        self.psi0 = dq.coherent(self.N, self.alpha0)
        self.tsave = jnp.linspace(0.0, self.T, 200)

    def printDevices(self):
        print(jax.devices())

    def photonLoss(self):
        result = dq.mesolve(self.H, self.jump_ops, self.psi0, self.tsave)

        n_sim = dq.expect(self.ad @ self.a, result.states).real
        n_th = self.alpha0 ** 2 * jnp.exp(-self.kappa * self.tsave)

        fig, ax = plt.subplots()
        ax.plot(self.tsave, n_sim, 'o', markersize=3, label='simulation (mesolve)')
        ax.plot(self.tsave, n_th, '-', label=r'$|\alpha_0|^2 e^{-\kappa t}$')
        ax.set_xlabel(r'$t$ ($\mu$s)')
        ax.set_ylabel(r'$\langle \hat a^\dagger \hat a\rangle$')
        ax.legend()
        fig.savefig(FIG / 'task1_0a_photon_loss.png', dpi=150)
        print('1.0(a) max |n_sim - n_analytic| =', float(jnp.max(jnp.abs(n_sim - n_th))))

        # Wigner snapshots: the blob should spiral in and stay round
        idx = [0, 25, 50, 100]
        fig2, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4))
        for ax2, i in zip(axes, idx):
            dq.plot.wigner(result.states[i], ax=ax2, cross=True)
            ax2.set_title(r'$t = %.2f\ \mu$s' % float(self.tsave[i]))
        fig2.savefig(FIG / 'task1_0a_wigner.png', dpi=150)
        plt.show()



test = HarmonicOscillator()
test.photonLoss()