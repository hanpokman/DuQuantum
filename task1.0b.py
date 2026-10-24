import dynamiqs as dq
import jax
import math
import jax.numpy as jnp
import matplotlib.pyplot as plt

dq.set_device('cpu')


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

        fig, ax = plt.subplots()
        ax.plot(self.tsave, n_sim, 'o', markersize=3, label='simulation (mesolve)')
        ax.set_xlabel(r'$t$ ($\mu$s)')
        ax.set_ylabel(r'$\langle \hat a^\dagger \hat a\rangle$')
        ax.legend()
        fig.savefig('photon_loss.png', dpi=150)
        plt.show()

        # Fock distribution at the final time
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        dq.plot.fock(result.states[0], ax=ax2)
        dq.plot.fock_evolution(result,ax=ax3)
        ax2.set_title(r'Fock distribution at $t = %.1f\ \mu$s' % self.T)
        fig2.savefig('fock_final.png', dpi=150)
        plt.show()


test = HarmonicOscillator()
test.photonLoss()