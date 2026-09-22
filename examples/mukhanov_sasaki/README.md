# Mukhanov-Sasaki Example

This example computes the primordial scalar power spectrum from single-field
quadratic inflation by doing two numerical solves:

1. the homogeneous inflationary background,
2. the Mukhanov-Sasaki perturbation equation for many independent Fourier modes.

It then extracts the scalar amplitude $A_s$ and scalar spectral index $n_s$,
and compares them to approximate slow-roll estimates.

Run it with:

```bash
uv run python examples/mukhanov_sasaki/main.py
```

The menodax solve runs as a Numba-CUDA kernel, so this needs a CUDA GPU.
The `scipy` backend below is the CPU-only path.

The perturbation equations and slow-roll approximation follow the conventions in Daniel Baumann's TASI lectures on inflation.

## Solver-backend benchmark

The mode solve can be timed across solver backends on identical right-hand-side
code with the `--benchmark` flag. Because the mode equation is non-stiff and
oscillatory, the comparison uses explicit solvers: menodax Tsit5 (GPU), Diffrax
Tsit5 (GPU), and `scipy.solve_ivp` RK45 (serial CPU baseline).

```bash
# all three backends at 64 modes (scipy is the slow one)
uv run python examples/mukhanov_sasaki/main.py --benchmark --n 64
# GPU backends at a saturating ensemble size
uv run python examples/mukhanov_sasaki/main.py --benchmark --backends menodax diffrax --n 4096
```

## Background Inflation

The model is quadratic inflation in reduced Planck units:

$$
V(\phi) = \frac{1}{2}m^2\phi^2.
$$

The mass $m$ is chosen from a slow-roll estimate so that the scalar amplitude
is close to the observed value

$$
A_s \simeq 2.1 \times 10^{-9}.
$$

The background is evolved in e-fold time

$$
N = \log a
$$

with state

$$
y = \left(\phi, \frac{d\phi}{dN}\right).
$$

The script solves

$$
\epsilon = \frac{1}{2}\left(\frac{d\phi}{dN}\right)^2,
$$

$$
H^2 = \frac{V(\phi)}{3 - \epsilon},
$$

The scalar field equation in an expanding FLRW universe is the homogeneous
Klein-Gordon equation with Hubble friction:

$$
\ddot{\phi} + 3H\dot{\phi} + \frac{dV}{d\phi} = 0.
$$

Switching variables from cosmic time $t$ to e-fold time $N$ using

$$
\frac{dN}{dt} = H,
$$

and

$$
\epsilon = -\frac{\dot{H}}{H^2},
$$

yields the e-fold-time background equation used in the script:

$$
\frac{d^2\phi}{dN^2}
=
-(3-\epsilon)\frac{d\phi}{dN}
-
\frac{1}{H^2}\frac{dV}{d\phi}.
$$

Inflation ends when

$$
\epsilon = 1.
$$

The pivot mode is defined to leave the horizon $55$ e-folds before the end of
inflation:

$$
N_* = N_{\rm end} - N_{\rm pivot} = 55.
$$

## Background Tables

After solving the background, the script builds interpolation tables for the
quantities needed by the perturbation equation:

$$
\epsilon(N),
$$

$$
\log(aH)(N),
$$

$$
z(N) = a\frac{d\phi}{dN},
$$

and

$$
q(N) = \frac{z''/z}{(aH)^2}.
$$

Here primes mean derivatives with respect to conformal time $\tau$, but the
script computes the needed combination in e-fold time as

$$
q(N)
=
\frac{1}{z}\frac{d^2z}{dN^2}
+
(1-\epsilon)\frac{1}{z}\frac{dz}{dN}.
$$

That identity follows from

$$
\frac{d}{d\tau} = aH\frac{d}{dN}.
$$

## Mode Setup

The script chooses $N_{\rm MODES}$ comoving wavenumbers between

$$
k_{\min} = 10^{-4}\ {\rm Mpc}^{-1}
$$

and

$$
k_{\max} = 1\ {\rm Mpc}^{-1}.
$$

The pivot scale is

$$
k_{\rm pivot} = 0.05\ {\rm Mpc}^{-1}.
$$

Internally, the physical $k$ values are mapped into code units by enforcing
horizon crossing at the pivot:

$$
k_{\rm pivot} = aH(N_{\rm pivot}).
$$

For each mode, the script starts the Mukhanov-Sasaki solve when the mode is
deep inside the horizon:

$$
\frac{k}{aH} = X_{\rm initial} = 80,
$$

and stops when it is far outside the horizon:

$$
\frac{k}{aH} = X_{\rm final} = 10^{-3}.
$$

## Mukhanov-Sasaki Equation

The conformal-time Mukhanov-Sasaki equation is

$$
v_k'' + \left(k^2 - \frac{z''}{z}\right)v_k = 0.
$$

The script solves the equivalent e-fold-time form:

$$
\frac{d^2v_k}{dN^2}
+
(1-\epsilon)\frac{dv_k}{dN}
+
\left[
\left(\frac{k}{aH}\right)^2
-
\frac{z''/z}{(aH)^2}
\right]v_k
= 0.
$$

Using

$$
q(N) = \frac{z''/z}{(aH)^2},
$$

this becomes

$$
\frac{d^2v_k}{dN^2}
=
-(1-\epsilon)\frac{dv_k}{dN}
-
\left[
\left(\frac{k}{aH}\right)^2 - q
\right]v_k.
$$

Because $v_k$ is complex and the solver uses real-valued states, the script
stores

$$
y =
\left[
\operatorname{Re}(v_k),
\operatorname{Im}(v_k),
\operatorname{Re}\left(\frac{dv_k}{dN}\right),
\operatorname{Im}\left(\frac{dv_k}{dN}\right)
\right].
$$

The real and imaginary parts obey the same real equation.

## Initial Conditions

Each mode starts in the Bunch-Davies vacuum while deep inside the horizon:

$$
v_k \simeq \frac{e^{-ik\tau}}{\sqrt{2k}}.
$$

The script uses

$$
x = \frac{k}{aH}
$$

and initializes at

$$
x = X_{\rm initial}.
$$

With the chosen phase convention, it sets

$$
v_k = \frac{e^{iX_{\rm initial}}}{\sqrt{2k}},
$$

and

$$
\frac{dv_k}{dN}
=
-iX_{\rm initial}v_k.
$$

## Ensemble Solve

Each $k$-mode is an independent ODE system. The script batches all modes into
one call to `tsit5_solve(...)` from `menodax.tsit5`.

Each mode has its own start and stop e-folds, so the script maps each trajectory
to a normalized solver time

$$
s \in [0,1],
$$

with

$$
N(s) = N_{\rm start} + s(N_{\rm stop} - N_{\rm start}).
$$

Therefore the right-hand side is multiplied by

$$
\frac{dN}{ds} = N_{\rm stop} - N_{\rm start}.
$$

## Power Spectrum

At the end of each mode solve, the curvature perturbation is

$$
\mathcal{R}_k = \frac{v_k}{z}.
$$

The dimensionless scalar power spectrum is

$$
\mathcal{P}_{\mathcal{R}}(k)
=
\frac{k^3}{2\pi^2}
\left|\frac{v_k}{z}\right|^2.
$$

The script computes this for all sampled $k$-modes.

## Extracting $A_s$ and $n_s$

The scalar amplitude is the power spectrum at the pivot:

$$
A_s = \mathcal{P}_{\mathcal{R}}(k_{\rm pivot}).
$$

The scalar spectral index is

$$
n_s - 1
=
\frac{d\log \mathcal{P}_{\mathcal{R}}}{d\log k}.
$$

The script estimates $n_s$ by fitting a local straight line to

$$
\log \mathcal{P}_{\mathcal{R}}
\quad \text{vs.} \quad
\log k
$$

near the pivot.

## Slow-Roll Comparison

Slow roll is the approximation that the inflaton moves slowly enough for the
potential energy to dominate the kinetic energy, and for the field acceleration
to be small compared with Hubble friction. In cosmic time this means

$$
\frac{1}{2}\dot{\phi}^2 \ll V(\phi),
$$

and

$$
|\ddot{\phi}| \ll 3H|\dot{\phi}|.
$$

Under these conditions the background equations reduce approximately to

$$
H^2 \simeq \frac{V}{3},
$$

and

$$
3H\dot{\phi} \simeq -\frac{dV}{d\phi}.
$$

The smallness of slow roll is usually measured by potential slow-roll
parameters:

$$
\epsilon_V =
\frac{1}{2}
\left(\frac{1}{V}\frac{dV}{d\phi}\right)^2,
$$

and

$$
\eta_V =
\frac{1}{V}\frac{d^2V}{d\phi^2}.
$$

Slow roll requires

$$
\epsilon_V \ll 1,
$$

and

$$
|\eta_V| \ll 1.
$$

For the quadratic potential, these become

$$
\epsilon_V = \eta_V = \frac{2}{\phi^2}.
$$

Large field values therefore correspond to slow roll. Inflation ends when the
slow-roll approximation predicts

$$
\epsilon_V \simeq 1,
$$

which gives

$$
\phi_{\rm end} \simeq \sqrt{2}.
$$

The number of e-folds remaining before the end of inflation is approximately

$$
N_* \simeq \frac{\phi_*^2 - \phi_{\rm end}^2}{4}.
$$

Equivalently, for quadratic inflation,

$$
\phi_*^2 \simeq 4N_* + 2,
$$

so at horizon exit

$$
\epsilon_V = \frac{1}{2N_* + 1},
$$

$$
\eta_V = \epsilon_V,
$$

The script solves the full background ODE rather than imposing these
approximations. The slow-roll formulas are used only to choose a sensible mass
scale and to provide an analytical comparison for the numerical
Mukhanov-Sasaki result. At the pivot, the approximate scalar amplitude is

$$
A_s^{\rm SR}
=
\frac{V}{24\pi^2\epsilon_V},
$$

and

$$
n_s^{\rm SR}
=
1 - 6\epsilon_V + 2\eta_V.
$$

It also prints the common compact approximation

$$
n_s \simeq 1 - \frac{2}{N_*}.
$$

Finally, it prints a table comparing the full numerical Mukhanov-Sasaki results
with the slow-roll estimates.

## References

- Daniel Baumann, "TASI Lectures on Inflation", arXiv:0907.5424,
  <https://arxiv.org/abs/0907.5424>.
- Andrew R. Liddle, "An Introduction to Cosmological Inflation", section 5.5,
  "A worked example: polynomial chaotic inflation",
  <https://ned.ipac.caltech.edu/level5/Liddle/Liddle5_5.html>.
