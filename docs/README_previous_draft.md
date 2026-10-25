# Dissipation One Ingredient at a Time: Building a Stabilized Cat Qubit

**DuQuantum 2026 · Alice & Bob Challenge · Introductory Track — Task 1.0 (a), (b), (c)**

---

### Abstract

We simulate a single bosonic mode (the cat-qubit *memory*) under three increasingly rich dissipative dynamics, adding one physical ingredient at a time. **(a)** Ordinary single-photon loss shrinks a coherent state along a spiral in phase space while preserving its shape; the simulated photon number matches the analytic law $|\alpha_0|^2 e^{-\kappa t}$ to $5\times10^{-15}$. **(b)** Pure two-photon loss does not empty the mode: the photon number stalls at $\langle \hat n\rangle_\infty = 0.49983$, which we derive exactly from parity conservation as the odd-photon probability $(1-e^{-8})/2$ of the initial state. **(c)** Two-photon loss combined with a two-photon drive inflates the vacuum into an even cat state with $\langle\hat n\rangle \to \alpha^2$; sweeping the two-photon rate $\kappa_2$ by a factor of 16 changes only the speed of the inflation, and plotting against $\kappa_2 t$ collapses all curves onto one. Every result was obtained twice, with `dynamiqs` and with an independent NumPy/SciPy solver, and is checked against a closed-form prediction.

---

## 1. Conventions and parameters

We set $\hbar = 1$. **Time is in µs and all rates are angular frequencies in rad/µs**, converted from the quoted ordinary frequencies by $\omega = 2\pi f$ (so $f = 1$ MHz becomes $2\pi$ rad/µs). All simulations use a single mode truncated to $N = 30$ Fock states and double precision.

**Table 1.** Parameters (provided values from Réglade *et al.*, arXiv:2307.06617, Table S2).

| Symbol | Meaning | Value | In code (rad/µs) |
|---|---|---|---|
| $g_2/2\pi$ | two-photon exchange rate | 0.763 MHz | `2*pi*0.763` = 4.794 |
| $\kappa_b/2\pi$ | buffer loss rate | 2.6 MHz | `2*pi*2.6` = 16.34 |
| $\kappa_a/2\pi$ | memory single-photon loss | 9.3 kHz | `2*pi*9.3e-3` = 0.0584 |
| $\kappa_2 = 4g_2^2/\kappa_b$ | effective two-photon rate | 0.896 MHz | 5.628 |
| $\alpha$ | target cat amplitude | 2.1 | $\alpha^2 = 4.41$ |

## 2. Method

### 2.1 The Lindblad master equation

Each task evolves the density matrix $\hat\rho$ of the memory under

$$
\frac{d\hat\rho}{dt} = -i\big[\hat H,\hat\rho\big] + \sum_k \Big( \hat L_k\hat\rho\hat L_k^\dagger - \tfrac12 \big\lbrace \hat L_k^\dagger\hat L_k,\ \hat\rho \big\rbrace \Big),
\tag{1}
$$

where the rates are absorbed into the jump operators $\hat L_k$. The three tasks differ only in $\hat H$ and $\lbrace\hat L_k\rbrace$:

**Table 2.** The three models.

| Task | Hamiltonian $\hat H$ | Jump operators $\hat L_k$ | Initial state | Window |
|---|---|---|---|---|
| 1.0(a) | $\Delta\,\hat a^\dagger\hat a$, $\ \Delta/2\pi = 1$ MHz | $\sqrt{\kappa}\,\hat a$, $\ \kappa/2\pi = 1$ MHz | $\lvert\alpha_0 = 2\rangle$ | 0–1 µs |
| 1.0(b) | $0$ | $\sqrt{\kappa_2}\,\hat a^2$ | $\lvert\alpha_0 = 2\rangle$ | 0–3 µs |
| 1.0(c) | $0$ | $\sqrt{\kappa_2}\,(\hat a^2 - \alpha^2\hat{\mathbb 1})$, $\ \sqrt{\kappa_a}\,\hat a$ | vacuum $\lvert 0\rangle$ | 0–2 µs |

### 2.2 Two independent solvers

1. **`dynamiqs`** (`task1.0.py`, `task1.0b.py`, `task1.0c.py`): `dq.mesolve` with the operators of Table 2.
2. **NumPy/SciPy cross-check** (`task10_numpy_crosscheck.py`): we write Eq. (1) as a linear equation $\frac{d}{dt}\mathrm{vec}(\hat\rho) = \mathcal L\,\mathrm{vec}(\hat\rho)$ using the column-stacking identity $\mathrm{vec}(AXB) = (B^{T}\otimes A)\,\mathrm{vec}(X)$:

$$
\mathcal L = -i\big(\mathbb 1\otimes \hat H - \hat H^{T}\otimes\mathbb 1\big) + \sum_k \Big( \hat L_k^{*}\otimes\hat L_k - \tfrac12\,\mathbb 1\otimes\hat L_k^\dagger\hat L_k - \tfrac12\,(\hat L_k^\dagger\hat L_k)^{T}\otimes\mathbb 1 \Big).
\tag{2}
$$

For $N = 30$, $\mathcal L$ is only $900\times900$, so we propagate exactly with $\mathrm{vec}(\hat\rho)(t+\delta t) = e^{\mathcal L\,\delta t}\,\mathrm{vec}(\hat\rho)(t)$ (no ODE-solver tolerances involved). Wigner functions are computed with the standard iterative Laguerre recursion, in coordinates where a coherent state $\lvert\beta\rangle$ is centred at $(\mathrm{Re}\,\beta,\ \mathrm{Im}\,\beta)$.

The two solvers share no code, so agreement between them rules out operator-construction and API mistakes. All figures in this README come from the cross-check script; they agree with the `dynamiqs` output.

### 2.3 Truncation check

A truncated simulation is trustworthy only if the highest Fock levels stay empty at all times. For the most demanding run, Task 1.0(c), the population of the top two levels ($n = 28, 29$) never exceeds $8\times10^{-14}$, so $N = 30$ is more than sufficient for all three tasks.

---

## 3. Task 1.0(a) — Single-photon loss

### 3.1 Prediction

With $\hat H = \Delta\,\hat a^\dagger\hat a$ and $\hat L = \sqrt\kappa\,\hat a$, the coherent state remains coherent at all times:

$$
\lvert\alpha_0\rangle \ \longrightarrow\ \lvert\alpha(t)\rangle, \qquad \alpha(t) = \alpha_0\, e^{-i\Delta t}\, e^{-\kappa t/2},
\qquad \langle\hat a^\dagger\hat a\rangle(t) = |\alpha_0|^2 e^{-\kappa t}.
\tag{3}
$$

The reason is that $\hat a\lvert\alpha\rangle = \alpha\lvert\alpha\rangle$: a quantum jump (one photon leaking out) leaves a coherent state unchanged, so the only effect of loss is the deterministic shrinking from the no-jump part of Eq. (1).

### 3.2 Result

![Task 1.0(a) photon number](figures/fig_a_n.png)

**Figure 1.** Mean photon number versus time. Points: simulation. Solid line: Eq. (3). Dashed line: the $\kappa = 0$ case, where the state only rotates and $\langle\hat n\rangle = 4$ stays constant.

![Task 1.0(a) Wigner snapshots](figures/fig_a_wigner.png)

**Figure 2.** Wigner function at $t = 0,\ 0.125,\ 0.25,\ 0.5$ µs. The blob rotates clockwise once per µs and spirals toward the origin, but stays round throughout.

![Task 1.0(a) phase-space trajectory](figures/fig_a_spiral.png)

**Figure 3.** Trajectory of $\langle\hat a\rangle$ in the complex plane: simulation (points) versus $\alpha(t)$ from Eq. (3).

### 3.3 Checks

| Quantity | Value |
|---|---|
| $\max_t \lvert \langle\hat n\rangle_{\rm sim} - \lvert\alpha_0\rvert^2 e^{-\kappa t}\rvert$ | $5\times10^{-15}$ |
| $\max_t \lvert \langle\hat a\rangle_{\rm sim} - \alpha(t)\rvert$ | $3\times10^{-15}$ |
| $\min_t \langle\alpha(t)\rvert\hat\rho(t)\lvert\alpha(t)\rangle$ (fidelity with Eq. 3) | $> 0.99999$ |

The agreement is at machine precision. It also confirms the unit conversion: a missing factor of $2\pi$ would make the decay about 6 times too slow.

---

## 4. Task 1.0(b) — Pure two-photon loss

### 4.1 Prediction

With $\hat H = 0$ and $\hat L = \sqrt{\kappa_2}\,\hat a^2$, two facts determine the final state.

1. **Parity is conserved.** $\hat a^2$ removes photons only in pairs, so the photon-number parity $\hat P = e^{i\pi\hat a^\dagger\hat a}$ never changes: $\frac{d}{dt}\langle\hat P\rangle = 0$.
2. **Two states are dark.** $\hat a^2\lvert 0\rangle = \hat a^2\lvert 1\rangle = 0$, so the vacuum and the one-photon state are untouched by the dissipation.

Every even Fock state therefore decays down to $\lvert 0\rangle$ and every odd one down to $\lvert 1\rangle$. The final photon number equals the initial odd-photon probability. For the Poisson distribution of $\lvert\alpha_0\rangle$:

$$
\langle\hat a^\dagger\hat a\rangle_\infty = P_{\rm odd} = e^{-|\alpha_0|^2}\sum_{n\ \rm odd}\frac{|\alpha_0|^{2n}}{n!} = e^{-|\alpha_0|^2}\sinh|\alpha_0|^2 = \frac{1 - e^{-2|\alpha_0|^2}}{2}.
\tag{4}
$$

For $\alpha_0 = 2$ this gives $(1 - e^{-8})/2 = 0.499832$.

### 4.2 Result

![Task 1.0(b) photon number](figures/fig_b_n.png)

**Figure 4.** Mean photon number under pure two-photon loss. It falls quickly at first, because the loss rate grows like $\kappa_2\langle\hat n\rangle^2$ for large $\langle \hat n\rangle$, and then stalls on the prediction of Eq. (4) (dashed line) within about 1 µs.

![Task 1.0(b) Fock populations](figures/fig_b_fock.png)

**Figure 5.** Fock populations of the initial coherent state (grey) and of the final state at $t = 3$ µs (blue). All population ends in $\lvert0\rangle$ and $\lvert1\rangle$, split 50.017 % / 49.983 %.

### 4.3 Checks

| Quantity | Simulation | Prediction |
|---|---|---|
| $\langle\hat n\rangle$ at $t = 3$ µs | 0.499832 | 0.499832 (Eq. 4) |
| $P(\lvert1\rangle)$ at $t = 3$ µs | 0.499832 | $P_{\rm odd}$ of the initial state |
| $\max_t \lvert\langle\hat P\rangle(t) - \langle\hat P\rangle(0)\rvert$ | $1\times10^{-15}$ | 0 (parity conserved) |
| change in $\langle\hat n\rangle$ between 1 µs and 3 µs | $1.5\times10^{-5}$ | — (confirms the window reaches steady state) |

---

## 5. Task 1.0(c) — Stabilization: two-photon loss plus two-photon drive

### 5.1 Prediction

The jump operator $\hat L_2 = \sqrt{\kappa_2}\,(\hat a^2 - \alpha^2)$ combines two-photon loss with a two-photon drive. Its dark states are the two coherent states $\lvert\pm\alpha\rangle$, because $\hat a^2\lvert\pm\alpha\rangle = \alpha^2\lvert\pm\alpha\rangle$. We therefore expect:

* **Where it settles:** $\langle\hat a^\dagger\hat a\rangle \to \alpha^2 = 4.41$, independent of $\kappa_2$.
* **How fast:** with $\kappa_a = 0$, Eq. (1) reads $\dot{\hat\rho} = \kappa_2\,\mathcal D[\hat a^2-\alpha^2]\hat\rho$. The generator is $\kappa_2$ times a fixed, $\kappa_2$-independent superoperator, so $\hat\rho(t)$ depends on time **only through $\kappa_2 t$**. The settling time is then exactly proportional to $1/\kappa_2$. A nonzero $\kappa_a$ adds a second timescale and breaks this scaling slightly.
* **Which state:** the vacuum has even parity, and the pair processes in $\hat L_2$ preserve parity, so the state inflates into the even cat $\lvert\mathcal C_+\rangle \propto \lvert\alpha\rangle + \lvert-\alpha\rangle$, not into a single codeword. Single-photon loss $\kappa_a$ then flips the parity, so we expect the cat's parity to decay roughly as $e^{-2\kappa_a\alpha^2 t}$ (the phase-flip rate studied in Task 2.2).

### 5.2 Result

![Task 1.0(c) kappa2 sweep](figures/fig_c_n.png)

**Figure 6.** Left: $\langle\hat a^\dagger\hat a\rangle(t)$ from vacuum for $\kappa_2/4$, $\kappa_2$ and $4\kappa_2$ (with $\kappa_a$ included). All three rise monotonically, without overshoot, to $\alpha^2$ (dashed). Right: the same curves against rescaled time $\kappa_2 t$, where they collapse onto one.

![Task 1.0(c) Wigner snapshots](figures/fig_c_wigner.png)

**Figure 7.** Wigner function at the nominal $\kappa_2$ for $t = 0,\ 0.05,\ 0.1,\ 2$ µs. The vacuum blob stretches along the real axis, splits into two blobs at $\pm 2.1$, and develops interference fringes (negative regions) between them. These fringes are the signature of a superposition rather than a mixture. By 2 µs the fringes are visibly fainter because of $\kappa_a$.

**Table 3.** Settling time to within 1 % of $\alpha^2$.

| Rate | $t_{\rm settle}$ | $\kappa_2\, t_{\rm settle}$ | $\langle\hat n\rangle$ at 2 µs |
|---|---|---|---|
| $\kappa_2/4$ | 1.105 µs | 1.55 | 4.387 |
| $\kappa_2$ | 0.253 µs | 1.42 | 4.404 |
| $4\kappa_2$ | 0.063 µs | 1.41 | 4.408 |

The product $\kappa_2 t_{\rm settle}$ is nearly constant, as predicted. The small spread at $\kappa_2/4$ is due to $\kappa_a$: the slower the stabilization, the more single-photon loss competes with it. That competition also lowers the steady state slightly below 4.41. Rerunning with $\kappa_a = 0$ gives $\langle\hat n\rangle = 4.409$ at 2 µs, and the two runs differ by at most 0.006 photons over the window.

### 5.3 Checks

| Quantity | Value | Expectation |
|---|---|---|
| steady-state $\langle\hat n\rangle$ (nominal $\kappa_2$) | 4.404 | $\alpha^2 = 4.41$, slightly lowered by $\kappa_a$ |
| $\kappa_2 t_{\rm settle}$ across a 16× change in $\kappa_2$ | 1.41–1.55 | constant (exactly, if $\kappa_a = 0$) |
| $\langle\hat P\rangle$ at 2 µs | 0.373 | $\approx e^{-2\kappa_a\alpha^2\cdot 2\,\mu\text{s}} = 0.357$ |
| fidelity with $\lvert\mathcal C_+\rangle$ at 2 µs | 0.687 | below 1, since parity has decayed |
| max population of the top two Fock levels | $8\times10^{-14}$ | negligible, so $N = 30$ is converged |

The parity at 2 µs is close to the simple estimate $e^{-2\kappa_a\alpha^2 t}$. It lies slightly above it because the cat only reaches full size after about 0.25 µs, and the phase-flip rate grows with the photon number. This is the "ideal" reference behaviour that Task 1.1 compares against the full memory–buffer model.

---

## 6. Summary of results

| Task | Main finding | Verified against | Agreement |
|---|---|---|---|
| 1.0(a) | Coherent state spirals in but stays coherent | Eq. (3), closed form | $5\times10^{-15}$ |
| 1.0(b) | $\langle\hat n\rangle$ stalls at $P_{\rm odd}$ because parity is conserved | Eq. (4), closed form | 6 digits |
| 1.0(c) | Inflation to $\alpha^2$; speed $\propto 1/\kappa_2$; even cat forms | scaling argument, $\alpha^2$, $e^{-2\kappa_a\alpha^2 t}$ | within 1 % (α², κ₂ scaling) / 5 % (parity) |

All three tasks were also cross-checked between `dynamiqs` and the independent NumPy/SciPy solver.

## 7. Reproducing the results

```bash
pip install dynamiqs numpy scipy matplotlib
python task1.0.py                 # 1.0(a) with dynamiqs  -> fig_1_0a_photon_loss.png
python task1.0b.py                # 1.0(b) with dynamiqs  -> fig_1_0b_two_photon_loss.png, fock_final.png, fock_evolution.png
python task1.0c.py                # 1.0(c) with dynamiqs  -> cat_wigner.png, cat_stabilization.gif
python task10_numpy_crosscheck.py # all three, independent solver -> figures/*.png, figures/results_1_0.json (~10 s)
```

Every number quoted in this README is written to `figures/results_1_0.json` by the last command.

## 8. What went wrong, and what we learned

* **Units.** Every rate is quoted as $f$ but must enter the code as $2\pi f$. The machine-precision match in Task 1.0(a) is our check that the conversion is right everywhere.
* **A forgotten parameter.** Our first version of 1.0(c) ran with $\kappa_a = 0$ (left as a TODO). The photon-number curves barely changed, but the fringes stayed at full contrast. Including $\kappa_a$ revealed the slow parity decay, which is the physics of phase flips.
* **Overwritten figures.** Our 1.0(a) and 1.0(b) scripts originally saved to the same file name, so one figure silently replaced the other. We also plotted the initial instead of the final Fock distribution once (`states[0]` instead of `states[-1]`). The second, independent solver is what caught these problems.
* **Physics.** Task 1.0(b) was the key insight for us: two-photon processes conserve parity. This is why the stabilized state starting from vacuum is a cat superposition rather than a single logical state, which sets up Task 1.3.

## 9. Use of AI tools

As permitted by the challenge rules, we used Claude (Anthropic) to help write the NumPy/SciPy cross-check script, to make the presentation figures, and to draft this README. The `dynamiqs` simulations, the parameter choices and the physical interpretation are our own. We checked every number above against the closed-form predictions in Eqs. (3)–(4) and against the `dynamiqs` output.

## References

1. U. Réglade, A. Bocquet *et al.*, "Quantum control of a cat qubit with bit-flip times exceeding ten seconds," arXiv:2307.06617 (2023).
2. D. Manzano, "A short introduction to the Lindblad master equation," arXiv:1906.04478 (2019).
3. A. Joshi, K. Noh, Y. Y. Gao, "Quantum information processing with bosonic qubits in circuit QED," arXiv:2008.13471 (2020).
4. `dynamiqs` documentation, https://www.dynamiqs.org