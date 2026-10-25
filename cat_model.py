"""Shared two-mode cat-qubit model (memory a + buffer b) for Tasks 1.1-1.3.

Units: time in us, rates in rad/us (omega = 2*pi*f with f in MHz).
"""
import jax.numpy as jnp
import dynamiqs as dq

dq.set_device('cpu')
dq.set_precision('double')

TWO_PI = 2 * jnp.pi
G2 = TWO_PI * 0.763          # two-photon exchange rate
KAPPA_B = TWO_PI * 2.6       # buffer loss
KAPPA_A = TWO_PI * 9.3e-3    # memory single-photon loss
ALPHA = 2.1                  # target cat amplitude
EPS_D = ALPHA**2 * G2        # buffer drive, alpha^2 = eps_d / g2
KAPPA_2 = 4 * G2**2 / KAPPA_B  # effective two-photon rate

if hasattr(dq, 'set_progress_meter'):
    dq.set_progress_meter(False)


class TwoModeCat:
    """Full memory + buffer model: H = g2(a^2 b' + a'^2 b) - eps_d(b + b')."""

    def __init__(self, Na=24, Nb=10, alpha=ALPHA, kappa_a=KAPPA_A):
        self.Na, self.Nb, self.alpha = Na, Nb, alpha
        self.eps_d = alpha**2 * G2
        self.a, self.b = dq.destroy(Na, Nb)          # a acts on memory, b on buffer
        self.ad, self.bd = dq.dag(self.a), dq.dag(self.b)
        self.H = (G2 * (self.a @ self.a @ self.bd + self.ad @ self.ad @ self.b)
                  - self.eps_d * (self.b + self.bd))
        self.jump_ops = [jnp.sqrt(KAPPA_B) * self.b, jnp.sqrt(kappa_a) * self.a]
        self.vacuum = dq.tensor(dq.fock(Na, 0), dq.fock(Nb, 0))

    def run(self, tsave, psi0=None, H=None):
        psi0 = self.vacuum if psi0 is None else psi0
        H = self.H if H is None else H
        return dq.mesolve(H, self.jump_ops, psi0, tsave)

    def memory(self, states):
        """Reduced memory state rho_a (buffer traced out)."""
        return dq.ptrace(states, 0)

    def buffer(self, states):
        return dq.ptrace(states, 1)

    def n_memory(self, states):
        return dq.expect(self.ad @ self.a, states).real

    def n_buffer(self, states):
        return dq.expect(self.bd @ self.b, states).real


def effective_model(tsave, N=30, alpha=ALPHA, kappa2=KAPPA_2, kappa_a=KAPPA_A):
    a = dq.destroy(N)
    jump_ops = [jnp.sqrt(kappa2) * (a @ a - alpha**2 * dq.eye(N)),
                jnp.sqrt(kappa_a) * a]
    res = dq.mesolve(0.0 * dq.dag(a) @ a, jump_ops, dq.fock(N, 0), tsave)
    return dq.expect(dq.dag(a) @ a, res.states).real