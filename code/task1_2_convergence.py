import math
import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import dynamiqs as dq

dq.set_device('cpu')
dq.set_precision('double')   # Section 0.5: float32 is too coarse for this challenge
dq.set_progress_meter(False)

# Tight solver tolerances: with the defaults, integrator error of ~1e-3 shows up in the
# high-Fock diagonals of the two-mode simulation and masquerades as truncation leakage.
METHOD = dq.method.Tsit5(rtol=1e-9, atol=1e-11)

two_pi = 2 * math.pi
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Provided parameters (Reglade et al., Table S2), in rad/us
g2      = two_pi * 0.763
kappa_b = two_pi * 2.6
kappa_a = two_pi * 9.3e-3
alpha   = 2.1
eps_d   = alpha**2 * g2
kappa_2 = 4 * g2**2 / kappa_b   # effective two-photon dissipation rate

print(f'kappa_2/2pi = {kappa_2 / two_pi:.4f} MHz   (kappa_2 = {kappa_2:.4f} rad/us)')
print(f'eps_d/2pi   = {eps_d / two_pi:.4f} MHz')

def two_mode_expects(Na, Nb, ka=kappa_a, T=4.0, nt=1601, method=METHOD):
    # Expectation values only (no saved states), so large truncations stay cheap in memory
    a, b = dq.destroy(Na, Nb)
    H = g2 * (a @ a @ b.dag()) + g2 * (a.dag() @ a.dag() @ b) - eps_d * (b + b.dag())
    jump_ops = [jnp.sqrt(kappa_b) * b] + ([jnp.sqrt(ka) * a] if ka else [])
    top_a = dq.tensor(dq.fock_dm(Na, Na - 1) + dq.fock_dm(Na, Na - 2), dq.eye(Nb))
    top_b = dq.tensor(dq.eye(Na), dq.fock_dm(Nb, Nb - 1))
    psi0 = dq.tensor(dq.fock(Na, 0), dq.fock(Nb, 0))
    tsave = jnp.linspace(0.0, T, nt)
    res = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=[a.dag() @ a, top_a, top_b],
                     method=method, save_states=False)
    e = res.expects.real
    return dict(t=tsave, n_a=e[0], ss=float(e[0, -1]), peak=float(e[0].max()),
                top_a=float(e[1].max()), top_b=float(e[2].max()))

Na_list = np.array([16, 20, 24, 28, 32, 40])
Nb_list = np.array([4, 6, 8, 10, 12, 14, 16])
CACHE = 'results/convergence_grid.npz'

cached = os.path.exists(CACHE) and np.load(CACHE)
if cached and np.array_equal(cached['Na_list'], Na_list) and np.array_equal(cached['Nb_list'], Nb_list):
    grid = {k: cached[k] for k in cached.files}
    print(f'loaded {CACHE}')
else:
    runs = [[two_mode_expects(int(Na), int(Nb)) for Nb in Nb_list] for Na in Na_list]
    grid = {k: np.array([[r[k] for r in row] for row in runs]) for k in ['ss', 'peak', 'top_a', 'top_b', 'n_a']}
    grid.update(Na_list=Na_list, Nb_list=Nb_list, t=np.asarray(runs[0][0]['t']))
    os.makedirs('results', exist_ok=True)
    np.savez(CACHE, **grid)
    print(f'computed and saved {CACHE}')

TOL = 1e-3

def converged_index(values, tol=TOL):
    # smallest i such that values[j] for all j >= i agree with values[-1] within tol
    ok = np.abs(values - values[-1]) / np.abs(values[-1]) < tol
    return next(i for i in range(len(values)) if ok[i:].all())

def global_corner(Q, tol=TOL):
    # cheapest (Na, Nb) whose whole upper-right rectangle is within tol of the reference Q[-1, -1]
    ok = np.abs(Q - Q[-1, -1]) / np.abs(Q[-1, -1]) < tol
    corners = [(Na_list[i] * Nb_list[j], Na_list[i], Nb_list[j])
               for i in range(len(Na_list)) for j in range(len(Nb_list)) if ok[i:, j:].all()]
    return tuple(int(x) for x in min(corners)[1:])

for key, name in [('ss', 'steady-state <n_a>(4 us)'), ('peak', 'peak <n_a>')]:
    Q = grid[key]
    print(f'=== {name}: reference (40, 16) = {Q[-1, -1]:.6f} ===')
    print('relative deviation from reference (rows N_a, columns N_b):')
    print('  Na\\Nb ' + ''.join(f'{Nb:>9}' for Nb in Nb_list))
    for i, Na in enumerate(Na_list):
        print(f'  {Na:>5} ' + ''.join(f'{abs(Q[i, j] / Q[-1, -1] - 1):9.1e}' for j in range(len(Nb_list))))
    print('  N_a* at each fixed N_b: ' + ', '.join(f'{Nb}->{Na_list[converged_index(Q[:, j])]}'
                                                  for j, Nb in enumerate(Nb_list)))
    print('  N_b* at each fixed N_a: ' + ', '.join(f'{Na}->{Nb_list[converged_index(Q[i, :])]}'
                                                  for i, Na in enumerate(Na_list)))
    print(f'  globally converged corner (tol {TOL:.0e}): (N_a, N_b) = {global_corner(Q)}\n')


fig, axs = plt.subplots(1, 4, figsize=(19, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.1, 1.1]))
for ax, key, title in [(axs[0], 'ss', r'steady state $\langle\hat n_a\rangle$'), (axs[1], 'peak', r'peak $\langle\hat n_a\rangle$')]:
    Q = grid[key]
    err = np.log10(np.maximum(np.abs(Q / Q[-1, -1] - 1), 1e-12))
    im = ax.imshow(err, origin='lower', cmap='viridis_r', vmin=-10, vmax=0, aspect='auto')
    for i in range(len(Na_list)):
        for j in range(len(Nb_list)):
            ax.text(j, i, f'{err[i, j]:.0f}', ha='center', va='center', fontsize=8,
                    color='w' if err[i, j] > -5 else 'k',
                    fontweight='bold' if err[i, j] < np.log10(TOL) else 'normal')
    ax.set_xticks(range(len(Nb_list)), Nb_list)
    ax.set_yticks(range(len(Na_list)), Na_list)
    ax.set_xlabel(r'$N_b$')
    ax.set_ylabel(r'$N_a$')
    ax.set_title(title + '\n' + r'$\log_{10}$ rel. deviation from (40, 16)')
fig.colorbar(im, ax=axs[:2], shrink=0.8)
t_grid = grid['t']
for j, Nb in enumerate(Nb_list):
    axs[2].plot(t_grid, grid['n_a'][-1, j], label=rf'$N_b = {Nb}$')
axs[2].set_title(r'transient vs $N_b$ at $N_a = 40$')
for i, Na in enumerate(Na_list):
    axs[3].plot(t_grid, grid['n_a'][i, -1], label=rf'$N_a = {Na}$')
axs[3].set_title(r'transient vs $N_a$ at $N_b = 16$')
for ax in axs[2:]:
    ax.axhline(alpha**2, ls=':', c='k')
    ax.set_xlim(0, 1.2)
    ax.set_xlabel(r'$t$ ($\mu$s)')
    ax.set_ylabel(r'$\langle \hat a^\dagger \hat a\rangle$')
    ax.legend(fontsize=7)
fig.savefig(f'{FIG_DIR}/1_2_convergence_grid.png', dpi=150, bbox_inches='tight')
plt.show()