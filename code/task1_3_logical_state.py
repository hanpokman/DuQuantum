import numpy as np
import jax
import jax.numpy as jnp
import dynamiqs as dq
import matplotlib.pyplot as plt

dq.set_device('cpu')
dq.set_precision('double')

Na, Nb = 24, 6
alpha = 2.1
T = 4.0
tsave = jnp.linspace(0, T, 401)
t = np.asarray(tsave)
g2 = 2 * jnp.pi * 0.763
kappa_b = 2 * jnp.pi * 2.6
kappa_a = 2 * jnp.pi * 0.0093
eps_d = alpha**2 * g2
a, b = dq.destroy(Na, Nb)
ad, bd = dq.dag(a), dq.dag(b)
H = g2 * (a @ a @ bd + ad @ ad @ b) - eps_d * (b + bd)
jump_ops = [jnp.sqrt(kappa_b) * b, jnp.sqrt(kappa_a) * a]
psi_vac = dq.tensor(dq.fock(Na, 0), dq.fock(Nb, 0))
result_vac = dq.mesolve(H, jump_ops, psi_vac, tsave, save_states = True) # Lindblad evolve
rho_a = dq.ptrace(result_vac.states, keep = 0, dims = (Na, Nb))

ket_plus = dq.coherent(Na, +alpha)
ket_minus = dq.coherent(Na, -alpha)
even_cat = dq.unit(ket_plus + ket_minus)
cat_projector = even_cat @ dq.dag(even_cat)

mixture = 0.5 * (ket_plus @ dq.dag(ket_plus) + ket_minus @ dq.dag(ket_minus))

P = dq.parity(Na)
am = dq.destroy(Na)
nm = dq.dag(am) @ am

parity = np.asarray(dq.expect(P, rho_a).real)

cat_overlap = np.asarray(dq.expect(cat_projector, rho_a).real)
n_photons = np.asarray(dq.expect(nm, rho_a).real)

cat_parity_reference = float(dq.expect(P, even_cat).real)
mix_parity_reference = float(dq.expect(P, mixture).real)
cat_overlap_reference = float(dq.expect(cat_projector, even_cat).real)
mix_overlap_reference = float(dq.expect(cat_projector, mixture).real)

print(f'Even-cat parity = {cat_parity_reference:.6f}')
print(f'Mixture parity = {mix_parity_reference:.6f}')
print(f'Even-cat cat-overlap = {cat_overlap_reference:.6f}')
print(f'Mixture cat-overlap = {mix_overlap_reference:.6f}')

i_best = int(np.argmax(cat_overlap)) # index of best-prepared cut
t_best = t[i_best]
print(f'Best cat time = {t_best:.3f} us')
print(f'Cat overlap then = {cat_overlap[i_best]:.6f}')
print(f'Parity then = {parity[i_best]:.6f}')
print(f'Photon number then = {n_photons[i_best]:.6f}')
print(f'Cat overlap at T = {cat_overlap[-1]:.6f}')
print(f'Parity at T = {parity[-1]:.6f}')