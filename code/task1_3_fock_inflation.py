"""Task 1.3 extension: inflate from memory Fock |0> and |1> (same model as Task 1.1).

Parity is conserved by the stabilization, so
    |0> (even) -> |C+>,   |1> (odd) -> |C->,
two orthogonal cat states = the logical |+_L>, |-_L> of a cat qubit.
Their superpositions should then approach |+alpha> = (|C+> + |C->)/sqrt2 etc.

Runs every initial state twice: with the real kappa_a and with kappa_a = 0, so the
conservation law and the phase-flip error from kappa_a can be told apart.
Outputs task1_3_fock.png and task1_3_fock_wigner.png.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dynamiqs as dq
from cat_model import TwoModeCat, ALPHA
from pathlib import Path
FIG = Path(__file__).resolve().parent.parent / "figures" / "task1_3"
FIG.mkdir(parents=True, exist_ok=True)

Na, Nb = 24, 10
T = 2.0
tsave = jnp.linspace(0.0, T, 201)

plus, minus = dq.coherent(Na, ALPHA), dq.coherent(Na, -ALPHA)
cat_p, cat_m = dq.unit(plus + minus), dq.unit(plus - minus)
P = dq.parity(Na)
a_mem = dq.destroy(Na)
vac_b = dq.fock(Nb, 0)

initial = {
    r'$|0\rangle$': dq.fock(Na, 0),
    r'$|1\rangle$': dq.fock(Na, 1),
    r'$(|0\rangle+|1\rangle)/\sqrt{2}$': dq.unit(dq.fock(Na, 0) + dq.fock(Na, 1)),
    r'$(|0\rangle-|1\rangle)/\sqrt{2}$': dq.unit(dq.fock(Na, 0) - dq.fock(Na, 1)),
}


def pop(ket, rho):
    """<ket|rho|ket> for every saved time."""
    return dq.expect(dq.todm(ket), rho).real


def evolve(kappa_a=None):
    model = TwoModeCat(Na, Nb) if kappa_a is None else TwoModeCat(Na, Nb, kappa_a=kappa_a)
    out = {}
    for name, mem0 in initial.items():
        rho_a = model.memory(model.run(tsave, psi0=dq.tensor(mem0, vac_b)).states)
        out[name] = dict(
            rho=rho_a,
            parity=dq.expect(P, rho_a).real,
            f_cp=pop(cat_p, rho_a), f_cm=pop(cat_m, rho_a),
            f_plus=pop(plus, rho_a), f_minus=pop(minus, rho_a),
            re_a=dq.expect(a_mem, rho_a).real,
            purity=dq.purity(rho_a),
        )
    return out


runs = {'kappa_a on': evolve(), 'kappa_a = 0': evolve(kappa_a=0.0)}

names = list(initial)
n0, n1 = names[0], names[1]
for tag, res in runs.items():
    print(f"\n=== {tag} ===")
    for tt in (0.5, 2.0):
        i = int(jnp.argmin(jnp.abs(tsave - tt)))
        for name in names:
            d = res[name]
            print(f"t={tt:.1f}us {name:34s} P={float(d['parity'][i]):+.4f} "
                  f"F(C+)={float(d['f_cp'][i]):.4f} F(C-)={float(d['f_cm'][i]):.4f} "
                  f"F(+a)={float(d['f_plus'][i]):.4f} Re<a>={float(d['re_a'][i]):+.3f} "
                  f"purity={float(d['purity'][i]):.4f}")
        # orthogonality of the two inflated Fock states: Tr(rho_0 rho_1) = 0 means orthogonal
        r0 = res[n0]['rho'].to_jax()[i]
        r1 = res[n1]['rho'].to_jax()[i]
        print(f"t={tt:.1f}us overlap Tr(rho_|0> rho_|1>) = {float(jnp.trace(r0 @ r1).real):.2e}")

# ---- plots: parity and target-state fidelities -----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for col, (tag, res) in enumerate(runs.items()):
    ax = axes[0, col]
    ax.plot(tsave, res[n0]['parity'], label=r'$\langle\hat P\rangle$ from $|0\rangle$')
    ax.plot(tsave, res[n1]['parity'], label=r'$\langle\hat P\rangle$ from $|1\rangle$')
    ax.plot(tsave, res[n0]['f_cp'], '--', label=r'$F(|\mathcal{C}_+\rangle)$ from $|0\rangle$')
    ax.plot(tsave, res[n1]['f_cm'], '--', label=r'$F(|\mathcal{C}_-\rangle)$ from $|1\rangle$')
    ax.set_title(tag); ax.set_ylim(-1.05, 1.05); ax.legend(fontsize=8)
    ax = axes[1, col]
    ax.plot(tsave, res[names[2]]['f_plus'], label=r'$F(|{+\alpha}\rangle)$ from $(|0\rangle+|1\rangle)/\sqrt{2}$')
    ax.plot(tsave, res[names[3]]['f_minus'], '--', label=r'$F(|{-\alpha}\rangle)$ from $(|0\rangle-|1\rangle)/\sqrt{2}$')
    ax.plot(tsave, res[names[2]]['purity'], ':', label='purity')
    ax.set_ylim(0, 1.05); ax.set_xlabel(r'$t$ ($\mu$s)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'task1_3_fock.png', dpi=150)

# ---- Wigner of the four final states (kappa_a on) ---------------------------------
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
for ax, name in zip(axes, names):
    dq.plot.wigner(runs['kappa_a on'][name]['rho'][-1], ax=ax, xmax=4.0, colorbar=False)
    ax.set_title(f'from {name}, t = {T:.0f} μs')
fig.tight_layout(); fig.savefig(FIG / 'task1_3_fock_wigner.png', dpi=150)
plt.show()