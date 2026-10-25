"""Task 1.3: which logical state does inflation from vacuum prepare, and how to
prepare |0_L> = |+alpha> instead.

(a) Track parity <P>, fidelity with |C+>, purity Tr(rho_a^2) and the coherence
    <+a|rho_a|-a> (zero for a classical mixture) over 10 us, with a kappa_a = 0
    control run; fit the parity decay to get the superposition lifetime.
(b) Two symmetry-breaking methods, quantified by Re<a> and F(|+alpha>):
    b1: start the memory in a small coherent state |beta> instead of vacuum;
    b2: start in vacuum but apply a memory drive i*eps*(a' - a) for the first T_on.
Outputs task1_3a.png, task1_3a_wigner.png, task1_3b.png.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dynamiqs as dq
from cat_model import TwoModeCat, ALPHA, KAPPA_A, TWO_PI

Na, Nb = 24, 10          # converged for these quantities (see Task 1.2)
model = TwoModeCat(Na, Nb)

plus, minus = dq.coherent(Na, ALPHA), dq.coherent(Na, -ALPHA)
cat_p = dq.unit(plus + minus)                         # |C+>
mixture = 0.5 * (dq.todm(plus) + dq.todm(minus))      # classical mixture
P = dq.parity(Na)
a_mem = dq.destroy(Na)


def diagnostics(rho_a):
    rho = rho_a.to_jax()
    return dict(
        parity=dq.expect(P, rho_a).real,
        f_cat=dq.expect(dq.todm(cat_p), rho_a).real,        # <C+|rho|C+>
        f_plus=dq.expect(dq.todm(plus), rho_a).real,        # <+a|rho|+a>
        purity=dq.purity(rho_a),
        # off-diagonal <+a|rho|-a>: ~0.5 for a cat, ~0 for a classical mixture.
        # (Tr(rho * mixture) cannot tell them apart -- it is 0.5 for both.)
        coherence=jnp.einsum('i,...ij,j->...', plus.to_jax()[:, 0].conj(), rho,
                             minus.to_jax()[:, 0]).real,
        a=dq.expect(a_mem, rho_a),
    )

# ---------------------------------------------------------------- (a)
tsave = jnp.linspace(0.0, 10.0, 501)
rho_a = model.memory(model.run(tsave).states)
d = diagnostics(rho_a)

# control: same run with kappa_a = 0 -> parity is exactly conserved
model0 = TwoModeCat(Na, Nb, kappa_a=0.0)
d0 = diagnostics(model0.memory(model0.run(tsave).states))

mask = tsave > 1.5                                   # after inflation is done
rate = -jnp.polyfit(tsave[mask], jnp.log(d['parity'][mask]), 1)[0]
pred = 2 * KAPPA_A * ALPHA**2
for tt in (0.2, 0.5, 1.0, 2.0):
    i = int(jnp.argmin(jnp.abs(tsave - tt)))
    for tag, dd in (('ka on ', d), ('ka = 0', d0)):
        print(f"(a) {tag} t={tt:.1f}us: <P>={float(dd['parity'][i]):.4f} "
              f"F(C+)={float(dd['f_cat'][i]):.4f} purity={float(dd['purity'][i]):.4f} "
              f"<+a|rho|-a>={float(dd['coherence'][i]):.4f}")
print(f"    parity decay rate {float(rate):.4f} /us -> lifetime {1/float(rate):.2f} us "
      f"(prediction 2 ka alpha^2 = {float(pred):.4f} /us -> {1/float(pred):.2f} us)")
i1 = int(jnp.argmin(jnp.abs(tsave - 1.0)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, dd, title in ((axes[0], d, r'$\kappa_a/2\pi = 9.3$ kHz'), (axes[1], d0, r'$\kappa_a = 0$ (control)')):
    ax.plot(tsave, dd['parity'], label=r'$\langle \hat P\rangle$')
    ax.plot(tsave, dd['f_cat'], label=r'$\langle\mathcal{C}_+|\rho_a|\mathcal{C}_+\rangle$')
    ax.plot(tsave, dd['purity'], label=r'Tr$\,\rho_a^2$')
    ax.plot(tsave, dd['coherence'], label=r'$\langle{+\alpha}|\rho_a|{-\alpha}\rangle$ (0 for a mixture)')
    ax.set_xlabel(r'$t$ ($\mu$s)'); ax.set_title(title); ax.set_ylim(0, 1.05)
axes[0].plot(tsave, jnp.exp(-pred * tsave), 'k:', lw=1, label=r'$e^{-2\kappa_a\alpha^2 t}$')
axes[0].legend(loc='lower left', fontsize=8)
fig.tight_layout(); fig.savefig('task1_3a.png', dpi=150)

# Wigner: cat at 1 us vs the mixture vs late time (fringes fade as parity decays)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, (st, title) in zip(axes, [(rho_a[i1], r'prepared, $t=1\,\mu$s'),
                                  (mixture, 'mixture (no fringes)'),
                                  (rho_a[-1], r'prepared, $t=10\,\mu$s')]):
    dq.plot.wigner(st, ax=ax, xmax=4.0, colorbar=False); ax.set_title(title)
fig.tight_layout(); fig.savefig('task1_3a_wigner.png', dpi=150)

# ---------------------------------------------------------------- (b)
tb = jnp.linspace(0.0, 2.0, 201)
vac_b = dq.fock(Nb, 0)

print("\n(b1) initial memory state |beta>:")
b1 = []
for beta in [0.3, 0.6, 1.0, 1.5, 2.1]:
    psi0 = dq.tensor(dq.coherent(Na, beta), vac_b)
    r = diagnostics(model.memory(model.run(tb, psi0=psi0).states)[-1])
    b1.append((beta, float(r['a'].real), float(r['f_plus'])))
    print(f"  beta = {beta:.1f}: Re<a> = {b1[-1][1]:.3f} (target {ALPHA}), F(+alpha) = {b1[-1][2]:.4f}")

print("\n(b2) vacuum start + memory drive i*eps*(a' - a) for T_on = 0.3 us:")
T_ON = 0.3
b2 = []
for eps_MHz in [0.5, 1.0, 2.0]:
    eps = TWO_PI * eps_MHz
    H_on = model.H + 1j * eps * (model.ad - model.a)        # pushes <a> along +Re
    t1 = jnp.linspace(0.0, T_ON, 31)
    s1 = model.run(t1, H=H_on).states[-1]
    t2 = jnp.linspace(0.0, 2.0 - T_ON, 171)
    r = diagnostics(model.memory(model.run(t2, psi0=s1).states)[-1])
    b2.append((eps_MHz, float(r['a'].real), float(r['f_plus'])))
    print(f"  eps/2pi = {eps_MHz:.1f} MHz: Re<a> = {b2[-1][1]:.3f}, F(+alpha) = {b2[-1][2]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot([x[0] for x in b1], [x[2] for x in b1], 'o-')
axes[0].set_xlabel(r'initial displacement $\beta$'); axes[0].set_ylabel(r'$F(\rho_a, |{+\alpha}\rangle)$ at 2 $\mu$s')
axes[1].plot([x[0] for x in b2], [x[2] for x in b2], 's-')
axes[1].set_xlabel(r'drive $\epsilon/2\pi$ (MHz), on for 0.3 $\mu$s')
for ax in axes: ax.set_ylim(0, 1.02); ax.axhline(1, color='k', lw=0.6, ls=':')
fig.tight_layout(); fig.savefig('task1_3b.png', dpi=150)
plt.show()