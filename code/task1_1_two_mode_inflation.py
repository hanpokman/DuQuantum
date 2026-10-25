import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from cat_model import effective_model      # Task 1.0(c) model: sqrt(k2)(a^2 - alpha^2), sqrt(ka) a

FIG = Path(__file__).resolve().parent.parent / "figures" / "task1_1"
FIG.mkdir(parents=True, exist_ok=True)

dq.set_device('cpu')
dq.set_precision('double')


class HarmonicOscillator:
    def __init__(self):
        self.Na = 24
        self.Nb = 6
        self.g2 = 2 * jnp.pi * 0.763
        self.kappa_b = 2 * jnp.pi * 2.6
        self.kappa_a = 2 * jnp.pi * 9.3e-3
        self.alpha = 2.1
        self.eps_d = self.alpha ** 2 * self.g2
        self.kappa2 = 4 * self.g2 ** 2 / self.kappa_b    # effective two-photon rate (Task 1.0)
        self.T = 2.0

        self.a, self.b = dq.destroy(self.Na, self.Nb)
        self.ad = dq.dag(self.a)
        self.bd = dq.dag(self.b)

        self.H = (self.g2 * (self.a @ self.a @ self.bd + self.ad @ self.ad @ self.b) - self.eps_d * (self.b + self.bd))
        self.jump_ops = [jnp.sqrt(self.kappa_b) * self.b, jnp.sqrt(self.kappa_a) * self.a]

        self.psi0 = dq.tensor(dq.fock(self.Na, 0), dq.fock(self.Nb, 0))
        self.tsave = jnp.linspace(0.0, self.T, 201)   # dt = 0.01 us

    def simulate(self):
        result = dq.mesolve(self.H, self.jump_ops, self.psi0, self.tsave)
        n_memory = dq.expect(self.ad @ self.a, result.states).real
        n_buffer = dq.expect(self.bd @ self.b, result.states).real

        # ---- overlay the effective single-mode model from Task 1.0(c), nominal kappa2
        n_effective = effective_model(self.tsave, kappa2=self.kappa2, kappa_a=self.kappa_a)

        plt.figure()
        plt.plot(self.tsave, n_memory, label="Memory (two-mode model)")
        plt.plot(self.tsave, n_effective, '--', label=r"Memory (effective model, Task 1.0(c))")
        plt.plot(self.tsave, n_buffer, label="Buffer (two-mode model)")
        plt.axhline(self.alpha ** 2, color='k', linestyle=":", label=r"Target $\alpha^2$")
        plt.xlabel(r"Time ($\mu$s)")
        plt.ylabel("Average photon number")
        plt.legend()
        plt.savefig(FIG / 'task1_1_photon_numbers.png', dpi=150)
        plt.show()

        i_peak = int(jnp.argmax(n_memory))
        print('two-mode : peak <n_a> = %.3f at t = %.2f us, final <n_a> = %.4f'
              % (float(n_memory[i_peak]), float(self.tsave[i_peak]), float(n_memory[-1])))
        print('effective: peak <n_a> = %.3f, final <n_a> = %.4f'
              % (float(n_effective.max()), float(n_effective[-1])))
        print('buffer   : peak <n_b> = %.3f, final <n_b> = %.4f'
              % (float(n_buffer.max()), float(n_buffer[-1])))
        return result

    def checkEffectiveModelCondition(self):
        # The effective model needs kappa_b >> g2 and kappa_b >> eps_d (buffer much faster than everything else)
        print('g2 / kappa_b    = %.3f' % float(self.g2 / self.kappa_b))
        print('eps_d / kappa_b = %.3f' % float(self.eps_d / self.kappa_b))

    def plotMemoryWigner(self, result):
        # memory alone: partial-trace out the buffer (subsystem 0 = memory, 1 = buffer)
        rho_a = dq.ptrace(result.states, 0)

        times = [0.0, 0.1, 0.2, 0.3, 0.6, 2.0]                       # us, through the transient
        idx = [int(jnp.argmin(jnp.abs(self.tsave - t))) for t in times]

        fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4))
        for ax, i in zip(axes, idx):
            dq.plot.wigner(rho_a[i], ax=ax, cross=True)
            ax.set_title(r'$t = %.2f\ \mu$s' % float(self.tsave[i]))
        fig.savefig(FIG / 'task1_1_memory_wigner.png', dpi=150)
        plt.show()

    def plotMemoryWignerGif(self, result, step=2, fps=15):
        from matplotlib.animation import FuncAnimation, PillowWriter

        rho_a = dq.ptrace(result.states, 0)
        frames = list(range(0, len(self.tsave), step))
        fig, ax = plt.subplots(figsize=(5, 5))

        def update(i):
            ax.clear()
            dq.plot.wigner(rho_a[i], ax=ax, cross=True)
            ax.set_title(r'memory, $t = %.2f\ \mu$s' % float(self.tsave[i]))

        anim = FuncAnimation(fig, update, frames=frames)
        anim.save(FIG / 'task1_1_memory_wigner.gif', writer=PillowWriter(fps=fps))
        plt.close(fig)
        print('saved task1_1_memory_wigner.gif with %d frames' % len(frames))


test = HarmonicOscillator()
test.checkEffectiveModelCondition()
result = test.simulate()
test.plotMemoryWigner(result)
test.plotMemoryWignerGif(result)
