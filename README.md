# 1D Fault Evolution Simulator

## Overview

This repository contains a numerical implementation of a one-dimensional quasi-dynamic fault evolution model using the Boundary Integral Method (BIM) with Dieterich-Ruina rate-and-state friction. The code simulates the spatiotemporal evolution of slip on a vertical strike-slip fault embedded in an elastic half-space, capturing the full earthquake cycle from interseismic loading through dynamic rupture and postseismic relaxation.

## Physical Problem

### Geometry

The model considers a vertical strike-slip fault in a homogeneous elastic half-space with the following coordinate system:

- **x₁**: Along-strike direction (parallel to fault, out-of-plane)
- **x₂**: Across-fault horizontal direction (perpendicular to fault)
- **x₃**: Down-dip direction (depth, positive downward)

The fault extends infinitely in the strike direction (x₁) and from the free surface (x₃ = 0) to a maximum depth. The problem is formulated as a 2D antiplane shear problem where:

- Only u₁ (strike-slip) displacements are non-zero
- u₂ = u₃ = 0
- All quantities vary only in the x₂-x₃ plane
- Slip varies only along the down-dip direction (x₃)

### Boundary Conditions

- **Free surface**: Zero shear stress at x₃ = 0
- **Far-field loading**: Imposed through back-slip at tectonic plate rate Vₚₗ
- **Fault boundary**: Governed by rate-and-state friction law

## Governing Equations

### 1. Elastodynamic Stress Transfer

The shear stress τ on the fault is related to slip δ through the elastostatic Green's functions:

```
τ(x₃, t) = τ₀ + ∫ K(x₃, y₃) [δ(y₃, t) - Vₚₗ·t] dy₃ - η·V(x₃, t)
```

where:
- τ₀ is the initial stress
- K(x₃, y₃) is the stress interaction kernel (computed using Okada's solutions)
- Vₚₗ is the plate loading rate
- η = μ/(2cₛ) is the radiation damping coefficient
- V = dδ/dt is the slip rate
- μ is the shear modulus
- cₛ is the shear wave speed

The radiation damping term approximates inertial effects in the quasi-dynamic formulation, providing a first-order correction to the quasi-static approximation.

### 2. Rate-and-State Friction Law

The fault resistance follows the regularized Dieterich-Ruina rate-and-state friction formulation:

```
τ = a·σ·asinh[V/(2V₀)·exp((f₀ + b·ψ)/a)]
```

where:
- a, b are empirical friction parameters
- σ is the effective normal stress
- V₀ is a reference slip rate
- f₀ is the reference friction coefficient
- ψ = ln(θV₀/Dᶜ) is the normalized state variable
- θ is the state variable with dimensions of time
- Dᶜ is the critical slip distance

The regularized form (using asinh) avoids singularities at V = 0 and is numerically more stable than the standard logarithmic form.

#### Friction Regimes

- **Velocity-weakening (VW)**: b > a, unstable sliding, generates earthquakes
- **Velocity-strengthening (VS)**: a > b, stable sliding, creeps steadily

### 3. State Evolution Law (Aging Law)

The evolution of the state variable follows the aging law:

```
dθ/dt = 1 - V·θ/Dᶜ
```

This is transformed into logarithmic form for numerical stability:

```
dψ/dt = (V₀·exp(-ψ) - V)/Dᶜ
```

where ψ = ln(θV₀/Dᶜ).

### 4. Slip Rate Evolution

Combining the friction law and stress balance, and taking time derivatives, yields the evolution equation for slip rate:

```
d(ln V)/dt = [K(V - Vₚₗ) - b·σ·(dψ/dt)·Q] / [a·σ·Q + η·V]
```

where:

```
Q = 1 / √[1 + (2V₀/V)²·exp(-2(f₀ + b·ψ)/a)]
```

This regularization factor Q ensures smooth behavior across all slip rates.

### 5. Complete ODE System

The full system consists of four coupled ordinary differential equations per fault cell:

```
dδ/dt = V                                    (slip evolution)
dτ/dt = K(V - Vₚₗ) - η·V·d(ln V)/dt         (stress evolution)
dψ/dt = (V₀·exp(-ψ) - V)/Dᶜ                 (state evolution)
d(ln V)/dt = [equation above]                (velocity evolution)
```

## Numerical Method

### Spatial Discretization

The fault is discretized into M uniform rectangular cells of width Δz:

- Cell positions: z_i = (i - 1)·Δz, i = 1, ..., M
- Cell centers: z_i^c = z_i + Δz/2
- Uniform slip assumed within each cell

### Stress Interaction Kernels

The stress kernel K_ij represents the shear stress at cell i due to unit slip on cell j. These kernels are computed using Okada's (1985, 1992) analytical solutions for rectangular dislocations in an elastic half-space. The free surface effect is incorporated through the method of images, requiring four terms per kernel evaluation:

1. Direct contribution from the source patch
2. Image contribution from the free surface
3. Contribution from the bottom of the source patch
4. Image contribution from the bottom of the source patch

### Temporal Integration

The ODE system is solved using the `scipy.integrate.solve_ivp` function with the RK45 (Runge-Kutta 4-5) adaptive time-stepping method:

- **Relative tolerance**: 10⁻⁸
- **Absolute tolerance**: 10⁻⁶
- **Maximum time step**: Limited to prevent missing dynamic events
- **Adaptive stepping**: Automatically refines during rapid slip (earthquakes)

The adaptive time-stepping is crucial as slip rates vary over 10+ orders of magnitude during the earthquake cycle.

## Code Structure

### Section 1: Utility Functions

Provides mathematical helper functions including boxcar, Heaviside, and ramp functions used for defining spatially varying friction parameters.

### Section 2: FaultProblem Class

Container class that organizes all model parameters including:
- Bulk material properties (density, wave speeds, elastic moduli)
- Frictional parameters (a, b, f₀, Dᶜ, V₀)
- Loading conditions (σ, Vₚₗ)
- Numerical parameters (kernels, damping coefficients)

### Section 3: Green's Functions

Implementation of elastostatic Green's functions:

- `stress_kernel_sigma12()`: Computes shear stress σ₁₂ at receiver due to unit slip on source patch
- `displacement_kernel_u1()`: Computes displacement u₁ at receiver due to unit slip on source

Both functions incorporate the free surface through the method of images.

### Section 4: Spatial Discretization

Sets up the computational domain:
- Defines fault geometry and grid spacing
- Generates cell positions and boundaries
- Creates virtual GPS receiver network for surface observations

### Section 5: Compute Interaction Kernels

Assembles the full stress interaction kernel matrix K and surface displacement kernel:
- Evaluates Green's functions at all source-receiver pairs
- Stores results in matrices for efficient computation

### Section 6: Fault Frictional Properties

Defines spatially varying friction parameters:
- Reference friction coefficient f₀
- Direct effect parameter a (may vary with depth)
- Evolution effect parameter b
- Normal stress σ
- Critical slip distance Dᶜ
- Plate rate Vₚₗ
- Radiation damping coefficient η

### Section 7: Characteristic Scales and Validation

Computes diagnostic quantities to validate the numerical setup:

- **Critical nucleation size**: h* = (π/2)·(μbDᶜ)/[(b-a)²σ]
- **Cohesive zone size**: Λ = (9π/32)·(μDᶜ)/(bσ)
- **Recurrence interval**: T ≈ 5(b-a)σL/(2μVₚₗ)

Checks that grid resolution satisfies Δz < Λ/3 for numerical accuracy.

### Section 8: Governing Equations

Implements the `rate_state_ode_system()` function that computes time derivatives for the ODE solver. This function:

1. Extracts state variables from the state vector
2. Computes slip rate from logarithmic velocity
3. Evaluates state variable evolution (aging law)
4. Computes stress perturbations from slip rate variations
5. Applies regularization factors
6. Computes velocity evolution from friction-stress balance
7. Updates stress evolution

### Section 9: Initial Conditions

Establishes steady-state initial conditions:
- Zero initial slip
- Stress equilibrium for steady sliding at plate rate
- State variable consistent with steady-state friction
- Uniform velocity equal to plate rate

### Section 10: Time Integration

Solves the ODE system using adaptive Runge-Kutta integration:
- Configures solver parameters and tolerances
- Monitors and reports computation progress
- Handles stiff dynamics during earthquake rupture

### Section 11: Post-Processing and Analysis

Extracts and processes results:
- Computes slip rates and maximum slip rates over time
- Evaluates synthetic surface displacements at GPS stations
- Identifies earthquake events and characteristic behavior

### Section 12: Visualization

Generates three figure sets:

1. **Space-Time Diagrams**: Color maps showing log₁₀(V) evolution in depth-time space and maximum slip rate time series
2. **Time-Step Domain**: Same plots in time-step coordinates to reveal adaptive stepping behavior
3. **Synthetic GPS**: Surface displacement profiles and time series at near-field and far-field stations

## Key Physical Parameters

### Material Properties
- **Density**: ρ = 2670 kg/m³
- **Shear wave speed**: cₛ = 3464 m/s
- **Shear modulus**: μ = ρcₛ² ≈ 32 GPa

### Friction Parameters
- **Reference friction**: f₀ = 0.6
- **Direct effect**: a = 0.01 - 0.025 (depth-dependent)
- **Evolution effect**: b = 0.015
- **Critical slip distance**: Dᶜ = 8 mm
- **Reference velocity**: V₀ = 10⁻⁶ m/s

### Loading Conditions
- **Normal stress**: σ = 50 MPa
- **Plate rate**: Vₚₗ = 10⁻⁹ m/s (≈ 31.5 mm/yr)

### Domain Configuration
- **Fault depth**: 40 km
- **Number of cells**: 400
- **Cell size**: 100 m
- **Velocity-weakening zone**: ~15-16 km depth range

## Model Capabilities

This implementation can simulate:

1. **Full earthquake cycles**: Interseismic loading, nucleation, dynamic rupture, and postseismic processes
2. **Multiple earthquake events**: Long-term behavior over hundreds of years
3. **Realistic slip rate variations**: From plate rate (~10⁻⁹ m/s) to seismic slip (~1 m/s)
4. **Surface deformation**: Synthetic GPS time series for comparison with observations
5. **Parametric studies**: Effects of friction properties, loading rates, and fault geometry

## Theoretical Background

### Quasi-Dynamic Approximation

The quasi-dynamic approximation includes radiation damping (η·V term) to account for inertial effects without solving the full wave equation. This is valid when:

- Rupture velocities remain sub-Rayleigh (V < cₛ)
- Interest is in long-term behavior rather than detailed dynamic rupture
- Computational efficiency is important for multi-cycle simulations

### Regularized Rate-and-State Friction

The regularized form using asinh instead of ln provides:

- Numerical stability at very low velocities
- Smooth behavior during the transition from quasi-static to dynamic slip
- Elimination of singularities in the original formulation

### State Evolution Laws

The aging law (used here) assumes:

- State evolution depends on contact time and slip
- Appropriate for modeling earthquake nucleation
- Produces velocity-weakening behavior when b > a

Alternative formulations (slip law, composite laws) exist but are not implemented in this version.

## Validation and Benchmarking

The code has been validated against:

- Analytical solutions for steady-state sliding
- Scaling relationships for nucleation size and recurrence interval
- Previous implementations of rate-and-state earthquake cycle models

Grid resolution requirements are automatically checked against the cohesive zone size to ensure numerical accuracy.

## Output Files

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

## References

### Fundamental Theory

- Dieterich, J. H. (1979). Modeling of rock friction: 1. Experimental results and constitutive equations. *Journal of Geophysical Research*, 84(B5), 2161-2168.
- Ruina, A. (1983). Slip instability and state variable friction laws. *Journal of Geophysical Research*, 88(B12), 10359-10370.
- Rice, J. R., Lapusta, N., & Ranjith, K. (2001). Rate and state dependent friction and the stability of sliding between elastically deformable solids. *Journal of the Mechanics and Physics of Solids*, 49(9), 1865-1898.

### Numerical Methods

- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. *Bulletin of the Seismological Society of America*, 75(4), 1135-1154.
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. *Bulletin of the Seismological Society of America*, 82(2), 1018-1040.
- Lapusta, N., Rice, J. R., Ben-Zion, Y., & Zheng, G. (2000). Elastodynamic analysis for slow tectonic loading with spontaneous rupture episodes on faults with rate- and state-dependent friction. *Journal of Geophysical Research*, 105(B10), 23765-23789.

### Rate-and-State Friction Applications

- Marone, C. (1998). Laboratory-derived friction laws and their application to seismic faulting. *Annual Review of Earth and Planetary Sciences*, 26, 643-696.
- Scholz, C. H. (1998). Earthquakes and friction laws. *Nature*, 391(6662), 37-42.
- Barbot, S. (2019). Slow-slip, slow earthquakes, period-two cycles, full and partial ruptures, and deterministic chaos in a single asperity fault. *Tectonophysics*, 768, 228171.

## Author

Luca Dal Zilio (luca.dalzilio@ntu.edu.sg)

## Contact

For questions, suggestions, or bug reports, please open an issue in the repository.
