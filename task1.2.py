"""Task 1.2: semiclassical check and convergence study.

Varies Na at fixed Nb, then Nb at fixed Na, and records two quantities:
  n_ss   : steady-state <a'a> (value at t = T)
  n_peak : peak of <a'a>(t) during the overshoot
A quantity is "converged" at the smallest cutoff whose value is within TOL of the
largest-cutoff value in that sweep. Outputs task1_2_convergence.png.

Runtime: the (Na=32, Nb=14) runs dominate; expect a few minutes on a laptop.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
from cat_model import TwoModeCat, ALPHA

TOL = 1e-3                     # 0.1 % relative
T = 3.0
tsave = jnp.linspace(0.0, T, 601)
NA_SWEEP, NB_FIXED = [12, 16, 20, 24, 28, 32], 12
NB_SWEEP, NA_FIXED = [4, 6, 8, 10, 12, 14], 32


def measure(Na, Nb):
    m = TwoModeCat(Na, Nb)
    n = m.n_memory(m.run(tsave).states)
    settled = abs(float(n[-1] - n[-41])) / float(n[-1])   # change over last 0.2 us
    return n, float(n[-1]), float(n.max()), settled


def sweep(pairs, label):
    rows = []
    for Na, Nb in pairs:
        n, ss, peak, settled = measure(Na, Nb)
        rows.append(dict(Na=Na, Nb=Nb, n=n, ss=ss, peak=peak))
        print(f"  Na={Na:3d} Nb={Nb:3d}  n_ss={ss:.5f}  n_peak={peak:.4f}  settled={settled:.1e}")
    ref = rows[-1]
    for q in ('ss', 'peak'):
        ok = next(r for r in rows if abs(r[q] - ref[q]) / ref[q] < TOL)
        print(f"  -> {q:4s} converged to {TOL:.0e} at {label} = {ok[label]}")
    return rows


print(f"semiclassical prediction alpha^2 = eps_d/g2 = {ALPHA**2:.4f}")
print(f"\nvary Na at Nb = {NB_FIXED}")
rows_a = sweep([(Na, NB_FIXED) for Na in NA_SWEEP], 'Na')
print(f"\nvary Nb at Na = {NA_FIXED}")
rows_b = sweep([(NA_FIXED, Nb) for Nb in NB_SWEEP], 'Nb')

best = rows_b[-1]['ss']
print(f"\nconverged n_ss = {best:.4f}; deviation from alpha^2 = "
      f"{100*(best-ALPHA**2)/ALPHA**2:+.2f} %")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, rows, key in ((axes[0], rows_a, 'Na'), (axes[1], rows_b, 'Nb')):
    for r in rows:
        ax.plot(tsave, r['n'], label=f"{key} = {r[key]}")
    ax.axhline(ALPHA**2, color='k', ls=':', lw=0.8)
    ax.set_xlim(0, 1.2); ax.set_xlabel(r'$t$ ($\mu$s)'); ax.legend()
axes[0].set_title(rf'vary $N_a$ ($N_b$ = {NB_FIXED})')
axes[1].set_title(rf'vary $N_b$ ($N_a$ = {NA_FIXED})')
axes[0].set_ylabel(r'$\langle \hat a^\dagger \hat a\rangle$')
fig.tight_layout(); fig.savefig('task1_2_convergence.png', dpi=150)
plt.show()