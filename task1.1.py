import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
        self.eps_d = self.alpha**2 * self.g2
        self.T = 2.0

        self.a, self.b = dq.destroy(self.Na, self.Nb)
        self.ad = dq.dag(self.a)
        self.bd = dq.dag(self.b)

        self.H = (self.g2 * (self.a @ self.a @ self.bd + self.ad @ self.ad @ self.b) - self.eps_d * (self.b + self.bd))
        self.jump_ops = [jnp.sqrt(self.kappa_b) * self.b, jnp.sqrt(self.kappa_a) * self.a]

        self.psi0 = dq.tensor(dq.fock(self.Na, 0), dq.fock(self.Nb, 0))
        self.tsave = jnp.linspace(0.0, self.T, 201)

    def simulate(self):
        result = dq.mesolve(self.H, self.jump_ops, self.psi0, self.tsave)
        n_memory = dq.expect(self.ad @ self.a, result.states).real
        n_buffer = dq.expect(self.bd @ self.b, result.states).real
        plt.figure()
        plt.plot(self.tsave, n_memory, label="Memory")
        plt.plot(self.tsave, n_buffer, label="Buffer")
        plt.axhline(self.alpha ** 2, linestyle="--", label="Target alpha²")
        plt.xlabel("Time (us)")
        plt.ylabel("Average photon number")
        plt.legend()
        plt.show()

test = HarmonicOscillator()
result = test.simulate()