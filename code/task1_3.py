# ===== notebook cell 1 =====
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


# ===== notebook cell 26 =====
Na3, Nb3 = 28, 12
a3, b3 = dq.destroy(Na3, Nb3)
H3 = g2 * (a3 @ a3 @ b3.dag()) + g2 * (a3.dag() @ a3.dag() @ b3) - eps_d * (b3 + b3.dag())
ka_batch = jnp.sqrt(jnp.array([kappa_a, 0.0]))[:, None, None] * a3   # nominal kappa_a, and kappa_a = 0
tsave_3 = jnp.linspace(0.0, 6.0, 121)

res_3 = dq.mesolve(H3, [jnp.sqrt(kappa_b) * b3, ka_batch], dq.tensor(dq.fock(Na3, 0), dq.fock(Nb3, 0)),
                   tsave_3, method=METHOD, save_states=False,
                   save_extra=lambda t, rho: dq.ptrace(rho, 0, (Na3, Nb3)).to_jax())
rho_a3 = dq.asqarray(res_3.extra, dims=(Na3,))      # shape (2, nt, Na3, Na3)

cat_plus = (dq.coherent(Na3, alpha) + dq.coherent(Na3, -alpha)).unit()
P3   = dq.expect(dq.parity(Na3), rho_a3).real
F3   = dq.fidelity(rho_a3, cat_plus)
pur3 = dq.purity(rho_a3).real

late = tsave_3 >= 1.5                               # after the inflation transient
Gamma_fit = -np.polyfit(np.asarray(tsave_3[late]), np.log(np.asarray(P3[0, late])), 1)[0]
t_half = float(tsave_3[jnp.argmax(P3[0] < 0.5)])
print(f'kappa_a = 0      : max |<P_a> - 1| = {jnp.max(jnp.abs(P3[1] - 1)):.1e},  '
      f'F(C+) at 2 us = {F3[1, 40]:.4f},  purity at 2 us = {pur3[1, 40]:.4f}')
print(f'nominal kappa_a  : fitted parity decay Gamma = {Gamma_fit:.4f} /us   vs   2 kappa_a alpha^2 = {2 * kappa_a * alpha**2:.4f} /us')
print(f'                   1/Gamma = {1 / Gamma_fit:.2f} us;  <P_a> drops below 0.5 at t = {t_half:.2f} us')
for t_i in [0.5, 1.0, 2.0, 4.0, 6.0]:
    i = int(jnp.searchsorted(tsave_3, t_i))
    print(f'  t = {t_i:3.1f} us:  <P_a> = {P3[0, i]:.3f}   F(C+) = {F3[0, i]:.3f}   purity = {pur3[0, i]:.3f}')

fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
for k, (lab, ls) in enumerate([(r'nominal $\kappa_a$', '-'), (r'$\kappa_a = 0$', '--')]):
    axs[0].plot(tsave_3, P3[k], ls, label=lab)
    axs[1].plot(tsave_3, F3[k], ls, label=lab)
    axs[2].plot(tsave_3, pur3[k], ls, label=lab)
axs[0].plot(tsave_3, np.exp(-2 * kappa_a * alpha**2 * np.asarray(tsave_3)), ':', c='k', label=r'$e^{-2\kappa_a\alpha^2 t}$')
axs[0].set_ylabel(r'$\langle\hat P_a\rangle$')
axs[1].set_ylabel(r'fidelity with $|\mathcal{C}_+\rangle$')
axs[2].axhline(0.5, ls=':', c='k', label='mixture of $|\\pm\\alpha\\rangle$')
axs[2].set_ylabel(r'purity Tr$\,\hat\rho_a^2$')
for ax in axs:
    ax.set_xlabel(r'$t$ ($\mu$s)')
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/1_3a_parity_fidelity_purity.png', dpi=150)
plt.show()

idx3 = jnp.searchsorted(tsave_3, jnp.array([0.5, 1.0, 2.0, 4.0, 6.0]))
dq.plot.wigner_mosaic(rho_a3[0][idx3], n=5, nrows=1, xmax=4)
plt.suptitle(r'Memory Wigner function, nominal $\kappa_a$: $t$ = 0.5, 1, 2, 4, 6 $\mu$s', y=1.05)
plt.savefig(f'{FIG_DIR}/1_3a_wigner.png', dpi=150, bbox_inches='tight')
plt.show()


# ===== notebook cell 29 =====
betas = jnp.array([0.0, 0.5, 1.0, 1.5, 2.1, 3, 4])
psi0_b = dq.tensor(dq.coherent(Na3, betas), dq.fock(Nb3, 0))           # batched initial states
proj_p = dq.tensor(dq.coherent(Na3, alpha).todm(), dq.eye(Nb3))       # |+alpha><+alpha| on memory
proj_m = dq.tensor(dq.coherent(Na3, -alpha).todm(), dq.eye(Nb3))
tsave_b3 = jnp.linspace(0.0, 3.0, 301)

res_b3 = dq.mesolve(H3, [jnp.sqrt(kappa_b) * b3, jnp.sqrt(kappa_a) * a3], psi0_b, tsave_b3,
                    exp_ops=[a3, proj_p, proj_m], method=METHOD, save_states=False)
re_a, F_p, F_m = res_b3.expects[:, 0].real, res_b3.expects[:, 1].real, res_b3.expects[:, 2].real

print(f"{'beta':>5} | {'F(+alpha), 1 us':>15} {'F(+alpha), 3 us':>15} {'F(-alpha), 3 us':>15} | {'Re<a>, 3 us':>11}")
for k, beta in enumerate(betas):
    print(f'{float(beta):5.1f} | {F_p[k, 100]:15.5f} {F_p[k, -1]:15.5f} {F_m[k, -1]:15.1e} | {re_a[k, -1]:11.4f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
for k, beta in enumerate(betas):
    ax1.plot(tsave_b3, F_p[k], label=rf'$\beta = {float(beta):g}$')
    ax2.plot(tsave_b3, re_a[k], label=rf'$\beta = {float(beta):g}$')
ax1.set_ylabel(r'fidelity with $|+\alpha\rangle$')
ax2.axhline(alpha, ls=':', c='k', label=r'$\alpha$')
ax2.set_ylabel(r'Re$\langle\hat a\rangle$')
for ax in (ax1, ax2):
    ax.set_xlabel(r'$t$ ($\mu$s)')
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/1_3b_displace_then_stabilize.png', dpi=150)
plt.show()


# ===== notebook cell 32 =====
# Symmetry-breaking drive on the memory: favour +alpha during inflation, starting from vacuum
H_bias = 1j * (a3.dag() - a3)                                   # Hermitian; pushes <a> along +Re
pulses = [(0.10, 2.0), (0.15, 2.0), (0.10, 3.0), (0.10, 4.0)]   # (duration in us, eps_z/2pi in MHz)
tsave_z = jnp.linspace(0.0, 3.0, 301)
psi0_z = dq.tensor(dq.fock(Na3, 0), dq.fock(Nb3, 0))

print(f"{'pulse':>8} {'eps_z/2pi':>10} | {'F(+alpha), 3 us':>15} {'F(-alpha), 3 us':>15} {'Re<a>, 3 us':>11}")
for tau, f_z in pulses:
    # drive is on for 0 <= t < tau, then off; the stabilizer runs throughout
    H3_bias = H3 + dq.pwc(jnp.array([0.0, tau]), jnp.array([two_pi * f_z]), H_bias)
    res_z = dq.mesolve(H3_bias, [jnp.sqrt(kappa_b) * b3, jnp.sqrt(kappa_a) * a3], psi0_z, tsave_z,
                       exp_ops=[a3, proj_p, proj_m], method=METHOD, save_states=False)
    e = res_z.expects.real
    print(f'{tau * 1e3:5.0f} ns {f_z:6.1f} MHz | {e[1, -1]:15.5f} {e[2, -1]:15.1e} {e[0, -1]:11.4f}')
