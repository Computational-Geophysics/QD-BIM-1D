# 1D fault evolution simulator

## Overview

This repository contains a numerical implementation of a one-dimensional quasi-dynamic fault evolution model using the Boundary Integral Method (BIM) with Dieterich-Ruina rate-and-state friction. The code simulates the spatiotemporal evolution of slip on a vertical strike-slip fault embedded in an elastic half-space, capturing the full earthquake cycle from interseismic loading through dynamic rupture and postseismic relaxation.

## Physical problem

### Geometry

The model considers a vertical strike-slip fault in a homogeneous elastic half-space with the following coordinate system:

- **$x_1$**: Along-strike direction (parallel to fault, out-of-plane)
- **$x_2$**: Across-fault horizontal direction (perpendicular to fault)
- **$x_3$**: Down-dip direction (depth, positive downward)

The fault extends infinitely in the strike direction ($x_1$) and from the free surface ($x_3 = 0$) to a maximum depth. The problem is formulated as a 2D antiplane shear problem where:

- Only $u_1$ (strike-slip) displacements are non-zero
- $u_2 = u_3 = 0$
- All quantities vary only in the $x_2$-$x_3$ plane
- Slip varies only along the down-dip direction ($x_3$)

### Boundary conditions

- **Free surface**: Zero shear stress at $x_3 = 0$
- **Far-field loading**: Imposed through back-slip at tectonic plate rate $V_{\text{pl}}$
- **Fault boundary**: Governed by rate-and-state friction law

## Governing equations

### 1. Elastodynamic stress transfer

The shear stress $\tau$ on the fault is related to slip $\delta$ through the elastostatic Green's functions:

$$\tau(x_3, t) = \tau_0 + \int K(x_3, y_3) [\delta(y_3, t) - V_{\text{pl}} \cdot t] \, dy_3 - \eta \cdot V(x_3, t)$$

where:
- $\tau_0$ is the initial stress
- $K(x_3, y_3)$ is the stress interaction kernel (computed using Okada's solutions)
- $V_{\text{pl}}$ is the plate loading rate
- $\eta = \mu/(2c_s)$ is the radiation damping coefficient
- $V = d\delta/dt$ is the slip rate
- $\mu$ is the shear modulus
- $c_s$ is the shear wave speed

The radiation damping term approximates inertial effects in the quasi-dynamic formulation, providing a first-order correction to the quasi-static approximation.

### 2. Rate-and-state friction law

The fault resistance follows the regularized Dieterich-Ruina rate-and-state friction formulation:

$$\tau = a \sigma \, \text{asinh}\left[\frac{V}{2V_0} \exp\left(\frac{f_0 + b\psi}{a}\right)\right]$$

where:
- $a$, $b$ are empirical friction parameters
- $\sigma$ is the effective normal stress
- $V_0$ is a reference slip rate
- $f_0$ is the reference friction coefficient
- $\psi = \ln(\theta V_0/D_c)$ is the normalized state variable
- $\theta$ is the state variable with dimensions of time
- $D_c$ is the critical slip distance

The regularized form (using asinh) avoids singularities at $V = 0$ and is numerically more stable than the standard logarithmic form.

#### Friction regimes

- **Velocity-weakening (VW)**: $b > a$, unstable sliding, generates earthquakes
- **Velocity-strengthening (VS)**: $a > b$, stable sliding, creeps steadily

### 3. State evolution law (aging law)

The evolution of the state variable follows the aging law:

$$\frac{d\theta}{dt} = 1 - \frac{V \theta}{D_c}$$

This is transformed into logarithmic form for numerical stability:

$$\frac{d\psi}{dt} = \frac{V_0 \exp(-\psi) - V}{D_c}$$

where $\psi = \ln(\theta V_0/D_c)$.

### 4. Slip Rate evolution

Combining the friction law and stress balance, and taking time derivatives, yields the evolution equation for slip rate:

$$\frac{d(\ln V)}{dt} = \frac{K(V - V_{\text{pl}}) - b\sigma \frac{d\psi}{dt} Q}{a\sigma Q + \eta V}$$

where:

$$Q = \frac{1}{\sqrt{1 + \left(\frac{2V_0}{V}\right)^2 \exp\left(-\frac{2(f_0 + b\psi)}{a}\right)}}$$

This regularization factor $Q$ ensures smooth behavior across all slip rates.

### 5. Complete ODE system

The full system consists of four coupled ordinary differential equations per fault cell:

$$\begin{align}
\frac{d\delta}{dt} &= V && \text{(slip evolution)} \\
\frac{d\tau}{dt} &= K(V - V_{\text{pl}}) - \eta V \frac{d(\ln V)}{dt} && \text{(stress evolution)} \\
\frac{d\psi}{dt} &= \frac{V_0 \exp(-\psi) - V}{D_c} && \text{(state evolution)} \\
\frac{d(\ln V)}{dt} &= \frac{K(V - V_{\text{pl}}) - b\sigma \frac{d\psi}{dt} Q}{a\sigma Q + \eta V} && \text{(velocity evolution)}
\end{align}$$

## Numerical method

### Spatial discretization

The fault is discretized into $M$ uniform rectangular cells of width $\Delta z$:

- Cell positions: $z_i = (i - 1)\Delta z$, $i = 1, \ldots, M$
- Cell centers: $z_i^c = z_i + \Delta z/2$
- Uniform slip assumed within each cell

### Stress interaction kernels

The stress kernel $K_{ij}$ represents the shear stress at cell $i$ due to unit slip on cell $j$. These kernels are computed using Okada's (1985, 1992) analytical solutions for rectangular dislocations in an elastic half-space. The free surface effect is incorporated through the method of images, requiring four terms per kernel evaluation:

1. Direct contribution from the source patch
2. Image contribution from the free surface
3. Contribution from the bottom of the source patch
4. Image contribution from the bottom of the source patch

### Temporal integration

The ODE system is solved using the `scipy.integrate.solve_ivp` function with the RK45 (Runge-Kutta 4-5) adaptive time-stepping method:

- **Relative tolerance**: $10^{-8}$
- **Absolute tolerance**: $10^{-6}$
- **Maximum time step**: Limited to prevent missing dynamic events
- **Adaptive stepping**: Automatically refines during rapid slip (earthquakes)

The adaptive time-stepping is crucial as slip rates vary over 10+ orders of magnitude during the earthquake cycle.

## Code Structure

### Section 1: Utility functions

Provides mathematical helper functions including boxcar, Heaviside, and ramp functions used for defining spatially varying friction parameters.

### Section 2: Faultproblem class

Container class that organizes all model parameters including:
- Bulk material properties (density, wave speeds, elastic moduli)
- Frictional parameters ($a$, $b$, $f_0$, $D_c$, $V_0$)
- Loading conditions ($\sigma$, $V_{\text{pl}}$)
- Numerical parameters (kernels, damping coefficients)

### Section 3: Green's functions

Implementation of elastostatic Green's functions:

- `stress_kernel_sigma12()`: Computes shear stress $\sigma_{12}$ at receiver due to unit slip on source patch
- `displacement_kernel_u1()`: Computes displacement $u_1$ at receiver due to unit slip on source

Both functions incorporate the free surface through the method of images.

### Section 4: Spatial discretization

Sets up the computational domain:
- Defines fault geometry and grid spacing
- Generates cell positions and boundaries
- Creates virtual GPS receiver network for surface observations

### Section 5: Compute interaction kernels

Assembles the full stress interaction kernel matrix $K$ and surface displacement kernel:
- Evaluates Green's functions at all source-receiver pairs
- Stores results in matrices for efficient computation

### Section 6: Fault frictional properties

Defines spatially varying friction parameters:
- Reference friction coefficient $f_0$
- Direct effect parameter $a$ (may vary with depth)
- Evolution effect parameter $b$
- Normal stress $\sigma$
- Critical slip distance $D_c$
- Plate rate $V_{\text{pl}}$
- Radiation damping coefficient $\eta$

### Section 7: Characteristic scales and validation

Computes diagnostic quantities to validate the numerical setup:

- **Critical nucleation size**: $h^* = \frac{\pi}{2} \frac{\mu b D_c}{(b-a)^2 \sigma}$
- **Cohesive zone size**: $\Lambda = \frac{9\pi}{32} \frac{\mu D_c}{b\sigma}$
- **Recurrence interval**: $T \approx \frac{5(b-a)\sigma L}{2\mu V_{\text{pl}}}$

Checks that grid resolution satisfies $\Delta z < \Lambda/3$ for numerical accuracy.

### Section 8: Governing equations

Implements the `rate_state_ode_system()` function that computes time derivatives for the ODE solver. This function:

1. Extracts state variables from the state vector
2. Computes slip rate from logarithmic velocity
3. Evaluates state variable evolution (aging law)
4. Computes stress perturbations from slip rate variations
5. Applies regularization factors
6. Computes velocity evolution from friction-stress balance
7. Updates stress evolution

### Section 9: Initial conditions

Establishes steady-state initial conditions:
- Zero initial slip
- Stress equilibrium for steady sliding at plate rate
- State variable consistent with steady-state friction
- Uniform velocity equal to plate rate

### Section 10: Time integration

Solves the ODE system using adaptive Runge-Kutta integration:
- Configures solver parameters and tolerances
- Monitors and reports computation progress
- Handles stiff dynamics during earthquake rupture

### Section 11: Post-processing and analysis

Extracts and processes results:
- Computes slip rates and maximum slip rates over time
- Evaluates synthetic surface displacements at GPS stations
- Identifies earthquake events and characteristic behavior

### Section 12: visualization

Generates three figure sets:

1. **Space-Time Diagrams**: Color maps showing $\log_{10}(V)$ evolution in depth-time space and maximum slip rate time series
2. **Time-Step Domain**: Same plots in time-step coordinates to reveal adaptive stepping behavior
3. **Synthetic GPS**: Surface displacement profiles and time series at near-field and far-field stations

## Key Physical parameters

### Material Properties
- **Density**: $\rho = 2670$ kg/m³
- **Shear wave speed**: $c_s = 3464$ m/s
- **Shear modulus**: $\mu = \rho c_s^2 \approx 32$ GPa

### Friction parameters
- **Reference friction**: $f_0 = 0.6$
- **Direct effect**: $a = 0.01 - 0.025$ (depth-dependent)
- **Evolution effect**: $b = 0.015$
- **Critical slip distance**: $D_c = 8$ mm
- **Reference velocity**: $V_0 = 10^{-6}$ m/s

### Loading conditions
- **Normal stress**: $\sigma = 50$ MPa
- **Plate rate**: $V_{\text{pl}} = 10^{-9}$ m/s (≈ 31.5 mm/yr)

### Domain configuration
- **Fault depth**: 40 km
- **Number of cells**: 400
- **Cell size**: 100 m
- **Velocity-weakening zone**: ~15-16 km depth range

## Model capabilities

This implementation can simulate:

1. **Full earthquake cycles**: Interseismic loading, nucleation, dynamic rupture, and postseismic processes
2. **Multiple earthquake events**: Long-term behavior over hundreds of years
3. **Realistic slip rate variations**: From plate rate ($\sim 10^{-9}$ m/s) to seismic slip ($\sim 1$ m/s)
4. **Surface deformation**: Synthetic GPS time series for comparison with observations
5. **Parametric studies**: Effects of friction properties, loading rates, and fault geometry

## Theoretical Background

### Quasi-dynamic approximation

The quasi-dynamic approximation includes radiation damping ($\eta V$ term) to account for inertial effects without solving the full wave equation. This is valid when:

- Rupture velocities remain sub-Rayleigh ($V < c_s$)
- Interest is in long-term behavior rather than detailed dynamic rupture
- Computational efficiency is important for multi-cycle simulations

### Regularized Rate-and-state friction

The regularized form using asinh instead of ln provides:

- Numerical stability at very low velocities
- Smooth behavior during the transition from quasi-static to dynamic slip
- Elimination of singularities in the original formulation

### State evolution laws

The aging law (used here) assumes:

- State evolution depends on contact time and slip
- Appropriate for modeling earthquake nucleation
- Produces velocity-weakening behavior when $b > a$

Alternative formulations (slip law, composite laws) exist but are not implemented in this version.

## Validation and benchmarking

The code has been validated against:

- Analytical solutions for steady-state sliding
- Scaling relationships for nucleation size and recurrence interval
- Previous implementations of rate-and-state earthquake cycle models

Grid resolution requirements are automatically checked against the cohesive zone size to ensure numerical accuracy.

## Output files

The code generates:

1. **Space-time plots**: Visualizing slip rate evolution across the fault
2. **Maximum slip rate time series**: Identifying earthquake events
3. **GPS time series**: Surface displacement patterns and evolution

All plots include proper units, labels, and physical interpretation aids.

## Dependencies

- Python 3.7+
- NumPy: Array operations and linear algebra
- SciPy: ODE integration (solve_ivp)
- Matplotlib: Visualization and plotting

## Running on HPC Systems

### WildFly HPC System (NTU)

The code can be executed on the WildFly High-Performance Computing cluster at Nanyang Technological University using the PBS (Portable Batch System) job scheduler.

#### Prerequisites

- Access to WildFly HPC system
- Python 3.11.5 environment with required dependencies (NumPy, SciPy, Matplotlib)
- PBS job script: `run_code.pbs`

#### Setup

Before running the simulation, edit `run_code.pbs` and replace `your.username` with your NTU account name in the virtual environment path:

```bash
source /home/your.username/venv_lstm_py311/bin/activate
```

#### Submitting a Job

To submit the simulation to the job queue:

```bash
qsub run_code.pbs
```

#### Monitoring Job Status

To check the status of your submitted job:

```bash
qstat -u your.username
```

Replace `your.username` with your NTU account name.

#### Job Output

By default, the PBS system creates two files in your working directory:
- `fault_evolution_1d.o<JOB_ID>`: Standard output
- `fault_evolution_1d.e<JOB_ID>`: Standard error/log messages

where `<JOB_ID>` is the unique job identifier assigned by the scheduler.

#### Job Configuration

The PBS script is configured with the following default settings:
- **Queue**: `qamd_wfly` (WildFly AMD queue)
- **Wall time**: 120 hours (5 days)
- **Resources**: 1 node, 1 CPU core
- **Python version**: 3.11.5

To modify these settings, edit the corresponding `#PBS` directives in `run_code.pbs`.

#### Additional PBS Commands

Useful commands for managing jobs:

```bash
# Delete a running or queued job
qdel <JOB_ID>

# View detailed job information
qstat -f <JOB_ID>

# View all jobs in the queue
qstat -a

# View queue status
qstat -Q
```


## References

### Fundamental theory

- Dieterich, J. H. (1979). Modeling of rock friction: 1. Experimental results and constitutive equations. *Journal of Geophysical Research*, 84(B5), 2161-2168.
- Ruina, A. (1983). Slip instability and state variable friction laws. *Journal of Geophysical Research*, 88(B12), 10359-10370.
- Rice, J. R., Lapusta, N., & Ranjith, K. (2001). Rate and state dependent friction and the stability of sliding between elastically deformable solids. *Journal of the Mechanics and Physics of Solids*, 49(9), 1865-1898.

### Numerical methods

- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. *Bulletin of the Seismological Society of America*, 75(4), 1135-1154.
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. *Bulletin of the Seismological Society of America*, 82(2), 1018-1040.
- Lapusta, N., Rice, J. R., Ben-Zion, Y., & Zheng, G. (2000). Elastodynamic analysis for slow tectonic loading with spontaneous rupture episodes on faults with rate- and state-dependent friction. *Journal of Geophysical Research*, 105(B10), 23765-23789.

### Rate-and-state friction applications

- Marone, C. (1998). Laboratory-derived friction laws and their application to seismic faulting. *Annual Review of Earth and Planetary Sciences*, 26, 643-696.
- Scholz, C. H. (1998). Earthquakes and friction laws. *Nature*, 391(6662), 37-42.
- Barbot, S. (2019). Slow-slip, slow earthquakes, period-two cycles, full and partial ruptures, and deterministic chaos in a single asperity fault. *Tectonophysics*, 768, 228171.

## Author

Luca Dal Zilio (luca.dalzilio@ntu.edu.sg)

## Contact

For questions, suggestions, or bug reports, please open an issue in the repository.
