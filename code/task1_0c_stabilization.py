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
        self.kappa = 4 * (0.763 * self.two_pi) ** 2 / (2.6 * self.two_pi)  # rad/us  (part b)
        self.alpha0 = 2.0
        self.T = 1.0  # us

        # part (c) parameters
        self.kappa2 = self.kappa        # nominal two-photon loss rate
        self.alpha_c = 2.1              # drive/stabilization amplitude
        self.kappa_a = self.two_pi * 9.3e-3  # rad/us  (kappa_a/2pi = 9.3 kHz)
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
        fig.savefig(FIG / 'task1_0b_two_photon_loss.png', dpi=150)
        plt.show()

        fig2, ax2 = plt.subplots()
        dq.plot.fock(result.states[-1], ax=ax2)
        ax2.set_title(r'Fock distribution at $t = %.1f\ \mu$s' % self.T)
        fig2.savefig(FIG / 'fock_final.png', dpi=150)
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

        target = self.alpha_c ** 2
        fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(11, 4))
        for k2, label in cases:
            n_t = self.stabilization(k2)
            ax.plot(self.tsave_c, n_t, label=label)
            ax_r.plot(self.tsave_c * k2, n_t, label=label)   # same curves vs kappa2 * t
            # settling time = last time <n> is more than 1 % away from alpha^2
            outside = jnp.where(jnp.abs(n_t - target) > 0.01 * target)[0]
            t_settle = float(self.tsave_c[outside[-1] + 1])
            print('1.0(c) %-22s final <n> = %.4f  t_settle = %.3f us  kappa2*t_settle = %.2f'
                  % (label, float(n_t[-1]), t_settle, float(k2) * t_settle))
        for axis in (ax, ax_r):
            axis.axhline(target, color='k', linestyle='--', linewidth=1, label=r'$\alpha^2$')
            axis.set_ylabel(r'$\langle \hat a^\dagger \hat a\rangle$')
        ax.set_xlabel(r'$t$ ($\mu$s)')
        ax_r.set_xlabel(r'$\kappa_2 t$')
        ax_r.set_xlim(0, 3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / 'cat_stabilization.png', dpi=150)
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
        parity = dq.expect(dq.parity(self.N), result.states).real
        print('1.0(c) parity at t = %.1f us: %.3f' % (float(self.tsave_c[-1]), float(parity[-1])))
        states = [result.states[i] for i in idx]
        times = [float(self.tsave_c[i]) for i in idx]

        fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4))
        for ax, state, t in zip(axes, states, times):
            dq.plot.wigner(state, ax=ax, cross=True)
            ax.set_title(r'$t = %.2f\ \mu$s' % t)
        fig.savefig(FIG / 'cat_wigner.png', dpi=150)
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
        anim.save(FIG / 'cat_stabilization.gif', writer=PillowWriter(fps=fps))
        plt.close(fig)
        print('saved cat_stabilization.gif with %d frames' % len(frames))


test = HarmonicOscillator()
test.plotStabilization()
test.plotStabilizationWigner()
test.plotStabilizationGif()