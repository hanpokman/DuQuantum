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
        self.Delta = self.two_pi * 1.0  # rad/us  (Delta/2pi = 1 MHz)
        self.kappa = 4 * (0.763 * self.two_pi) ** 2 / (2.6 * self.two_pi)  # rad/us
        self.alpha0 = 2.0
        self.T = 1.0  # us

        self.a = dq.destroy(self.N)
        self.ad = dq.dag(self.a)
        self.H = 0.0 * self.ad @ self.a
        self.jump_ops = [jnp.sqrt(self.kappa) * (self.a @ self.a)]
        self.psi0 = dq.coherent(self.N, self.alpha0)
        self.tsave = jnp.linspace(0.0, self.T, 200)

    def printDevices(self):
        print(jax.devices())

    def photonLoss(self):
        result = dq.mesolve(self.H, self.jump_ops, self.psi0, self.tsave)

        n_sim = dq.expect(self.ad @ self.a, result.states).real
        parity = dq.expect(dq.parity(self.N), result.states).real
        n_pred = (1 - jnp.exp(-2 * self.alpha0 ** 2)) / 2   # odd-photon probability of |alpha0>
        print('1.0(b) final <n> =', float(n_sim[-1]), '  prediction =', float(n_pred))
        print('1.0(b) parity drift =', float(jnp.max(jnp.abs(parity - parity[0]))))

        fig, ax = plt.subplots()
        ax.plot(self.tsave, n_sim, 'o', markersize=3, label='simulation (mesolve)')
        ax.axhline(n_pred, color='r', linestyle='--', label=r'prediction $(1-e^{-2|\alpha_0|^2})/2$')
        ax.set_xlabel(r'$t$ ($\mu$s)')
        ax.set_ylabel(r'$\langle \hat a^\dagger \hat a\rangle$')
        ax.legend()
        fig.savefig(FIG / 'task1_0b_two_photon_loss.png', dpi=150)
        plt.show()


        fig2, (ax_i, ax_f) = plt.subplots(1, 2, figsize=(10, 4))
        dq.plot.fock(result.states[0], ax=ax_i)
        ax_i.set_title(r'start: $|\alpha_0 = 2\rangle$ ($t = 0$)')
        dq.plot.fock(result.states[-1], ax=ax_f)
        ax_f.set_title(r'end ($t = %.1f\ \mu$s): only $|0\rangle$ and $|1\rangle$ left' % self.T)
        fig2.tight_layout()
        fig2.savefig(FIG / 'fock_final.png', dpi=150)

        fig3, ax3 = plt.subplots()
        dq.plot.fock_evolution(result.states, times=self.tsave, ax=ax3)
        ax3.set_xlabel(r'$t$ ($\mu$s)')
        fig3.savefig(FIG / 'fock_evolution.png', dpi=150)
        plt.show()


test = HarmonicOscillator()
test.photonLoss()