"""Independent NumPy/SciPy re-solve of Task 1.0 (a), (b), (c).

Same physics as the repo's dynamiqs scripts (task1.0.py, task1.0b.py, task1.0c.py):
single mode, N = 30, hbar = 1, time in us, rates in rad/us.
Lindblad equation is vectorised (column stacking) and propagated with expm(L*dt).
"""
import json
from pathlib import Path
FIG = Path(__file__).resolve().parent.parent / "figures" / "task1_0" / "numpy_check"
FIG.mkdir(parents=True, exist_ok=True)
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 30
TWO_PI = 2 * np.pi
G2, KB, KA = TWO_PI * 0.763, TWO_PI * 2.6, TWO_PI * 9.3e-3
K2 = 4 * G2**2 / KB                        # effective two-photon rate

a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
ad = a.conj().T
I = np.eye(N)
num = ad @ a


def coherent(alpha):
    # build in a big space, then truncate + renormalise (same as a displaced vacuum)
    M = 120
    n = np.arange(M)
    from scipy.special import gammaln
    c = np.exp(-abs(alpha)**2 / 2 + n * np.log(abs(alpha) + 1e-300) - 0.5 * gammaln(n + 1))
    c = c.astype(complex) * np.exp(1j * np.angle(alpha) * n)
    c = c[:N]
    return c / np.linalg.norm(c)


def liouvillian(H, Ls):
    """Column-stacking vec: vec(A X B) = (B^T kron A) vec(X)."""
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for J in Ls:
        JdJ = J.conj().T @ J
        L += np.kron(J.conj(), J) - 0.5 * np.kron(I, JdJ) - 0.5 * np.kron(JdJ.T, I)
    return L


def mesolve(H, Ls, psi0, tsave):
    rho = np.outer(psi0, psi0.conj())
    v = rho.reshape(-1, order="F")
    dt = tsave[1] - tsave[0]
    U = expm(liouvillian(H, Ls) * dt)
    out = [rho]
    for _ in tsave[1:]:
        v = U @ v
        out.append(v.reshape(N, N, order="F"))
    return np.array(out)


def expect(op, rhos):
    return np.real(np.einsum("ij,tji->t", op, rhos))


def wigner(rho, xv, yv):
    """Iterative Laguerre-free Wigner (Clenshaw-style), coordinates = alpha."""
    X, Y = np.meshgrid(xv, yv)
    A = X + 1j * Y                     # g = 2 convention -> axes are Re/Im alpha
    M = rho.shape[0]
    Wl = [np.exp(-2 * abs(A)**2) / np.pi]
    W = np.real(rho[0, 0]) * np.real(Wl[0])
    for n in range(1, M):
        Wl.append(2 * A * Wl[n - 1] / np.sqrt(n))
        W = W + 2 * np.real(rho[0, n] * Wl[n])
    for m in range(1, M):
        temp = Wl[m].copy()
        Wl[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wl[m - 1]) / np.sqrt(m)
        W = W + np.real(rho[m, m] * Wl[m])
        for n in range(m + 1, M):
            temp2 = (2 * A * Wl[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = Wl[n].copy()
            Wl[n] = temp2
            W = W + 2 * np.real(rho[m, n] * Wl[n])
    return 2 * W   # 0.5*g^2 with g = 2


# -------- style ----------
INK, MUTED, GRID = "#1F2430", "#6B7280", "#E5E7EB"
C1, C2, C3, ACC = "#2563EB", "#D97706", "#059669", "#DC2626"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
WMAX = 2 / np.pi
xv = np.linspace(-4, 4, 161)


def wigner_row(rhos_times, fname, figsize):
    fig, axes = plt.subplots(1, len(rhos_times), figsize=figsize)
    for ax, (rho, t) in zip(axes, rhos_times):
        W = wigner(rho, xv, xv)
        ax.imshow(W, extent=[-4, 4, -4, 4], origin="lower", cmap="RdBu_r",
                  vmin=-WMAX, vmax=WMAX)
        ax.set_title(f"t = {t:.2f} µs", fontsize=12, color=INK)
        ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
        ax.grid(False)
        ax.axhline(0, color=MUTED, lw=0.4); ax.axvline(0, color=MUTED, lw=0.4)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Im α", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Re α", fontsize=11)
    fig.savefig(fname)
    plt.close(fig)


results = {}

# ======================= (a) single-photon loss ============================
Delta, kap, a0 = TWO_PI * 1.0, TWO_PI * 1.0, 2.0
ta = np.linspace(0, 1, 401)
ra = mesolve(Delta * num, [np.sqrt(kap) * a], coherent(a0), ta)
n_a = expect(num, ra)
n_a_th = a0**2 * np.exp(-kap * ta)
amean = np.einsum("ij,tji->t", a, ra)
amean_th = a0 * np.exp(-1j * Delta * ta - kap * ta / 2)
# "stays coherent": fidelity with |alpha(t)> analytic
fid_coh = [np.real(coherent(z).conj() @ r @ coherent(z)) for z, r in zip(amean_th, ra)]
results["a"] = dict(max_abs_err_n=float(np.max(abs(n_a - n_a_th))),
                    max_rel_err_n=float(np.max(abs(n_a - n_a_th) / n_a_th)),
                    min_fid_coherent=float(np.min(fid_coh)),
                    max_err_amean=float(np.max(abs(amean - amean_th))))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(ta, n_a_th, color=INK, lw=2, label=r"analytic $|\alpha_0|^2 e^{-\kappa t}$")
ax.plot(ta[::10], n_a[::10], "o", color=C1, ms=6, label="simulation")
ax.plot(ta, 4 + 0 * ta, "--", color=MUTED, lw=1.2, label=r"$\kappa = 0$ (no loss)")
ax.set_xlabel("t (µs)"); ax.set_ylabel(r"$\langle a^\dagger a\rangle$")
ax.set_ylim(-0.1, 4.4); ax.legend(loc="center right")
fig.savefig(FIG / "fig_a_n.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot(amean_th.real, amean_th.imag, color=INK, lw=1.5, label="analytic")
ax.plot(amean.real[::8], amean.imag[::8], "o", color=C1, ms=4, label="simulation")
ax.plot([a0], [0], "o", color=ACC, ms=8, label="t = 0")
ax.set_aspect("equal"); ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3)
ax.set_xlabel(r"Re$\langle a\rangle$"); ax.set_ylabel(r"Im$\langle a\rangle$")
ax.legend(loc="lower left", fontsize=10)
fig.savefig(FIG / "fig_a_spiral.png"); plt.close(fig)

idx = [0, 50, 100, 200]
wigner_row([(ra[i], ta[i]) for i in idx], FIG / "fig_a_wigner.png", (10, 2.8))

# ======================= (b) pure two-photon loss ==========================
tb = np.linspace(0, 3, 601)
rb = mesolve(0 * num, [np.sqrt(K2) * a @ a], coherent(2.0), tb)
n_b = expect(num, rb)
psi = coherent(2.0)
p_odd = float(np.sum(abs(psi[1::2])**2))
p_odd_exact = (1 - np.exp(-8)) / 2
par = expect(np.diag((-1.0) ** np.arange(N)), rb)
pops_end = np.real(np.diag(rb[-1]))
i1 = np.searchsorted(tb, 1.0)
results["b"] = dict(kappa2=float(K2), kappa2_over_2pi_MHz=float(K2 / TWO_PI),
                    n_final=float(n_b[-1]), n_at_1us=float(n_b[i1]),
                    p_odd_initial=p_odd, p_odd_exact=float(p_odd_exact),
                    parity_drift=float(np.max(abs(par - par[0]))),
                    P0_final=float(pops_end[0]), P1_final=float(pops_end[1]))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(tb, n_b, color=C1, lw=2.5, label="simulation")
ax.axhline(p_odd_exact, color=ACC, ls="--", lw=1.5,
           label=r"prediction $P_{\rm odd}=(1-e^{-8})/2$")
ax.set_xlabel("t (µs)"); ax.set_ylabel(r"$\langle a^\dagger a\rangle$")
ax.set_ylim(0, 4.2); ax.legend(loc="upper right")
fig.savefig(FIG / "fig_b_n.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 4.2))
k = np.arange(10)
ax.bar(k - 0.2, abs(psi[:10])**2, width=0.4, color=GRID, edgecolor=MUTED,
       label="t = 0  (|α=2⟩)")
ax.bar(k + 0.2, pops_end[:10], width=0.4, color=C1, label="t = 3 µs")
ax.set_xticks(k); ax.set_xlabel("Fock level n"); ax.set_ylabel("population")
ax.legend(fontsize=10)
fig.savefig(FIG / "fig_b_fock.png"); plt.close(fig)

# ======================= (c) stabilisation =================================
alpha = 2.1
tc = np.linspace(0, 2, 801)
vac = np.zeros(N, complex); vac[0] = 1
curves = {}
for lab, k2 in [("nominal", K2), ("quarter", K2 / 4), ("x4", 4 * K2)]:
    rc = mesolve(0 * num, [np.sqrt(k2) * (a @ a - alpha**2 * I), np.sqrt(KA) * a], vac, tc)
    curves[lab] = (k2, expect(num, rc), rc)
# kappa_a = 0 comparison (what the repo currently runs)
rc0 = mesolve(0 * num, [np.sqrt(K2) * (a @ a - alpha**2 * I)], vac, tc)
n_c0 = expect(num, rc0)


def t_settle(t, n, target, tol=0.01):
    bad = np.where(abs(n - target) > tol * target)[0]
    return float(t[bad[-1] + 1]) if len(bad) else 0.0


results["c"] = {}
for lab, (k2, n, rc) in curves.items():
    results["c"][lab] = dict(k2=float(k2), n_final=float(n[-1]),
                             t_settle_1pct_us=t_settle(tc, n, alpha**2),
                             t_settle_x_k2=t_settle(tc, n, alpha**2) * float(k2),
                             max_n=float(n.max()))
results["c"]["kappa_a0_n_final"] = float(n_c0[-1])
results["c"]["max_diff_ka_vs_0"] = float(np.max(abs(curves["nominal"][1] - n_c0)))
rnom = curves["nominal"][2]
results["c"]["top2_pop_max"] = float(np.max(np.real(rnom[:, N - 2, N - 2] + rnom[:, N - 1, N - 1])))
cplus = coherent(alpha) + coherent(-alpha); cplus /= np.linalg.norm(cplus)
results["c"]["fid_Cplus_final"] = float(np.real(cplus.conj() @ rnom[-1] @ cplus))
results["c"]["parity_final"] = float(expect(np.diag((-1.0) ** np.arange(N)), rnom[-1:])[0])

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
for (lab, (k2, n, _)), col, name in zip(curves.items(), [C1, C2, C3],
                                         [r"$\kappa_2$ (nominal)", r"$\kappa_2/4$", r"$4\kappa_2$"]):
    ax.plot(tc, n, color=col, lw=2.5, label=name)
    ax2.plot(tc * k2, n, color=col, lw=2.5)
for axx in (ax, ax2):
    axx.axhline(alpha**2, color=INK, ls="--", lw=1)
ax.text(1.98, alpha**2 + 0.12, r"$\alpha^2 = 4.41$", ha="right", color=INK, fontsize=11)
ax.set_xlabel("t (µs)"); ax.set_ylabel(r"$\langle a^\dagger a\rangle$")
ax.legend(loc="lower right"); ax.set_ylim(0, 4.9)
ax2.set_xlim(0, 3); ax2.set_ylim(0, 4.9)
ax2.set_xlabel(r"rescaled time  $\kappa_2 t$")
ax2.set_title("curves collapse → timescale ∝ 1/κ₂", fontsize=11, color=MUTED)
fig.tight_layout()
fig.savefig(FIG / "fig_c_n.png"); plt.close(fig)

idx = [0, np.searchsorted(tc, 0.05), np.searchsorted(tc, 0.1), len(tc) - 1]
wigner_row([(rnom[i], tc[i]) for i in idx], FIG / "fig_c_wigner.png", (10, 2.8))

json.dump(results, open(FIG / "results_1_0.json", "w"), indent=2)
print(json.dumps(results, indent=2))
