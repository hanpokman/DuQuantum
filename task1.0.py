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
        fig.savefig('photon_loss.png', dpi=150)
        plt.show()



test = HarmonicOscillator()
test.photonLoss()