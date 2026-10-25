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
        self.kappa = 4 * (0.763 * self.two_pi) ** 2 / (2.6 * self.two_pi)  # rad/us  (part b)
        self.alpha0 = 2.0
        self.T = 1.0  # us

        # part (c) parameters
        self.kappa2 = self.kappa        # nominal two-photon loss rate
        self.alpha_c = 2.1              # drive/stabilization amplitude
        self.kappa_a = 0.0              # TODO: set to the provided kappa_a value
        self.T_c = 2.0                  # us

        self.a = dq.destroy(self.N)
        self.ad = dq.dag(self.a)
        self.I = dq.eye(self.N)
        self.H = 0.0 * self.ad @ self.a
        self.jump_ops = [jnp.sqrt(self.kappa) * (self.a @ self.a)]
        self.psi0 = dq.coherent(self.N, self.alpha0)
        self.tsave = jnp.linspace(0.0, self.T, 200)

        self.psi_vac = dq.fock(self.N, 0)
        self.tsave_c = jnp.linspace(0.0, self.T_c, 400)

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

        fig2, ax2 = plt.subplots()
        dq.plot.fock(result.states[-1], ax=ax2)
        ax2.set_title(r'Fock distribution at $t = %.1f\ \mu$s' % self.T)
        fig2.savefig('fock_final.png', dpi=150)
        plt.show()

    def stabilization(self, kappa2):
        # two-photon loss + two-photon drive: sqrt(k2)(a^2 - alpha^2) and one-photon loss sqrt(ka) a
        jump_ops = [
            jnp.sqrt(kappa2) * (self.a @ self.a - self.alpha_c ** 2 * self.I),
            jnp.sqrt(self.kappa_a) * self.a,
        ]
        result = dq.mesolve(self.H, jump_ops, self.psi_vac, self.tsave_c)
        return dq.expect(self.ad @ self.a, result.states).real

    def plotStabilization(self):
        cases = [
            (self.kappa2, r'$\kappa_2$ (nominal)'),
            (self.kappa2 / 4, r'$\kappa_2/4$'),
            (4 * self.kappa2, r'$4\kappa_2$'),
        ]

        fig, ax = plt.subplots()
        for k2, label in cases:
            n_t = self.stabilization(k2)
            ax.plot(self.tsave_c, n_t, label=label)
        ax.axhline(self.alpha_c ** 2, color='k', linestyle='--', linewidth=1,
                   label=r'$\alpha^2$')
        ax.set_xlabel(r'$t$ ($\mu$s)')
        ax.set_ylabel(r'$\langle \hat a^\dagger \hat a\rangle$')
        ax.legend()
        fig.savefig('cat_stabilization.png', dpi=150)
        plt.show()
    def plotStabilizationWigner(self, kappa2=None):
        if kappa2 is None:
            kappa2 = self.kappa2
        jump_ops = [
            jnp.sqrt(kappa2) * (self.a @ self.a - self.alpha_c ** 2 * self.I),
            jnp.sqrt(self.kappa_a) * self.a,
        ]
        result = dq.mesolve(self.H, jump_ops, self.psi_vac, self.tsave_c)

        # pick a few times: start (vacuum), early transient, and settled
        idx = [0, 40, 100, -1]
        states = [result.states[i] for i in idx]
        times = [float(self.tsave_c[i]) for i in idx]

        fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4))
        for ax, state, t in zip(axes, states, times):
            dq.plot.wigner(state, ax=ax, cross=True)
            ax.set_title(r'$t = %.2f\ \mu$s' % t)
        fig.savefig('cat_wigner.png', dpi=150)
        plt.show()

    def plotStabilizationGif(self, kappa2=None, step=4, fps=20):
        from matplotlib.animation import FuncAnimation, PillowWriter

        if kappa2 is None:
            kappa2 = self.kappa2
        jump_ops = [
            jnp.sqrt(kappa2) * (self.a @ self.a - self.alpha_c ** 2 * self.I),
            jnp.sqrt(self.kappa_a) * self.a,
        ]
        result = dq.mesolve(self.H, jump_ops, self.psi_vac, self.tsave_c)

        frames = list(range(0, len(self.tsave_c), step))
        fig, ax = plt.subplots(figsize=(5, 5))

        def update(i):
            ax.clear()
            dq.plot.wigner(result.states[i], ax=ax, cross=True)
            ax.set_title(r'$t = %.2f\ \mu$s' % float(self.tsave_c[i]))

        anim = FuncAnimation(fig, update, frames=frames)
        anim.save('cat_stabilization.gif', writer=PillowWriter(fps=fps))
        plt.close(fig)
        print('saved cat_stabilization.gif with %d frames' % len(frames))


test = HarmonicOscillator()
#test.plotStabilization()
test.plotStabilizationWigner()
test.plotStabilizationGif()