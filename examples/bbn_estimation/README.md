# BBN Parameter Estimation Example

Bayesian parameter estimation of cosmological parameters from primordial light-element
abundances, using a 4-species Big Bang Nucleosynthesis (BBN) ODE integrated with the
Rodas5 implicit solver and fitted via nested sampling.

Run it with:

```bash
uv run python examples/bbn_estimation/main.py
```

The menodax solve runs as a Numba-CUDA kernel, so this needs a CUDA GPU.
The `scipy` backend below is the CPU-only path.

## Solver-backend benchmark

The batched forward solve (the chi-squared grid / sampler-population inner loop)
can be timed across solver backends on identical right-hand-side code with the
`--benchmark` flag: menodax Rodas5P (GPU), Diffrax Kvaerno5 (GPU), and serial
`scipy.solve_ivp` LSODA (the no-GPU baseline).

```bash
# head-to-head over a 2000-point (eta, N_eff) grid (scipy is the slow one)
uv run python examples/bbn_estimation/main.py --benchmark --n 2000
# GPU backends at a larger grid
uv run python examples/bbn_estimation/main.py --benchmark --backends menodax diffrax --n 50000
```

## Expansion rate

In radiation domination the Hubble rate is given by the Friedmann equation:

$$H(T) = \sqrt{\frac{4\pi^3 g_*(T)}{45}} \frac{T^2}{M_\mathrm{Pl}}$$

where $g_*(T)$ counts the effective relativistic degrees of freedom (10.75 above the
$e^+e^-$ annihilation threshold at $T \approx 0.511\,\mathrm{MeV}$, falling to 3.91
below it), and $M_\mathrm{Pl} = 1.22 \times 10^{22}\,\mathrm{MeV}$ is the unreduced
Planck mass.

## Weak interaction rates

The neutron-to-proton rate follows the Bernstein, Brown & Feinberg (1989) polynomial:

$$\Gamma_{n\to p}(x) = \frac{255}{\tau_n} \frac{12 + 6x + x^2}{x^5} + \frac{1}{\tau_n}$$

where $x = Q/T$ ($Q = 1.293\,\mathrm{MeV}$ is the neutron–proton mass difference) and
$\tau_n = 879.4\,\mathrm{s}$ is the neutron lifetime. Detailed balance gives
$\Gamma_{p\to n} = \Gamma_{n\to p} e^{-x}$.

## Deuterium Saha equilibrium

The equilibrium ratio $K_D = Y_d^\mathrm{eq}/(Y_n Y_p)$ follows from the Saha
equation for the two-body binding reaction $n + p \leftrightarrow D + \gamma$:

$$K_D(T, \eta) = n_b \,\frac{3}{4} \left(\frac{4\pi}{M_N T}\right)^{3/2} e^{B_D/T}$$

with $B_D = 2.225\,\mathrm{MeV}$, $M_N = 938.3\,\mathrm{MeV}$, and baryon number
density $n_b = \eta \, n_\gamma$.

## ODE system

Using $x = Q/T$ as the independent variable (so $dx/dt = H x$), the nucleon
number-fraction state $Y = [Y_n, Y_p, Y_d, Y_{^4\mathrm{He}}]$ evolves as:

$$\frac{dY_n}{dx} = \frac{\Gamma_{pn} Y_p - \Gamma_{np} Y_n - R_{np}}{H x}$$

$$\frac{dY_p}{dx} = \frac{\Gamma_{np} Y_n - \Gamma_{pn} Y_p - R_{np}}{H x}$$

$$\frac{dY_d}{dx} = \frac{R_{np} - 2 R_{dd}}{H x}$$

$$\frac{dY_{^4\mathrm{He}}}{dx} = \frac{R_{dd}}{H x}$$

where the net nuclear rates are

$$R_{np} = n_b \langle\sigma v\rangle_{np} \left(Y_n Y_p - \frac{Y_d}{K_D}\right), \qquad R_{dd} = n_b \langle\sigma v\rangle_{dd} Y_d^2.$$

**Conservation law:** $Y_n + Y_p + 2 Y_d + 4 Y_{^4\mathrm{He}} = 1$ at all times.

## Parameters and observables

| Parameter | Symbol | Prior range |
|-----------|--------|-------------|
| Baryon-to-photon ratio (log) | $\log_{10}(\eta_{10})$ | $[0.5,\, 1.0]$ |
| Effective neutrino species | $N_\mathrm{eff}$ | $[2.0,\, 4.0]$ |

| Observable | Model value* | 1-sigma |
| --- | --- | --- |
| Helium mass fraction Y_P | 0.106 | 0.004 |
| Deuterium ratio D/H | 4.2e-6 | 0.3e-6 |

*Calibrated to CMB Planck 2018 parameters ($\eta_{10}\approx6.1$, $N_\mathrm{eff}=3.044$) using this
simplified 4-species network. The full BBN network gives $Y_P\approx0.245$ (Aver et al. 2015)
and $D/H\approx2.5\times10^{-5}$ (Cooke et al. 2018); the simplified network gives lower values
because the $D\to{}^4\mathrm{He}$ path through ${}^3\mathrm{He}$ and $T$ is omitted.

## References

1. J. Bernstein, L. S. Brown, G. Feinberg (1989). *Cosmological helium production simplified.* Rev. Mod. Phys. **61**, 25.
2. C. Angulo et al. / NACRE Collaboration (1999). *A compilation of charged-particle induced thermonuclear reaction rates.* Nucl. Phys. A **656**, 3.
3. R. J. Cooke, M. Pettini, C. C. Steidel (2018). *One Percent Determination of the Primordial Deuterium Abundance.* ApJ **855**, 102.
4. E. Aver, K. A. Olive, E. D. Skillman (2015). *The effects of He I λ10830 on helium abundance determinations.* JCAP **07**, 011.
5. J. Skilling (2006). *Nested sampling for general Bayesian computation.* Bayesian Analysis **1**, 833.
6. D. Yallup, M. Kroupa, W. Handley (2026). *Nested slice sampling.* arXiv:2601.23252.
