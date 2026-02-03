"""
================================================================================
1D FAULT EVOLUTION SIMULATOR
Boundary Integral Method with Rate-and-State Friction
================================================================================

This script simulates the evolution of a vertical strike-slip fault in a
half-space using the quasi-dynamic boundary integral method with Dieterich-Ruina
rate-and-state friction.

Physical Setup:
- Vertical fault in a half-space (2D antiplane shear problem)
- Out-of-plane displacements only (strike-slip motion)
- Free surface at depth = 0
- Slip varies only in the down-dip (depth) direction
- Uses method of images to account for free surface

Author: Luca Dal Zilio
================================================================================
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import time

# ============================================================================
# SECTION 1: UTILITY FUNCTIONS
# ============================================================================

def boxcar_function(x):
    """
    Boxcar (rectangular) function centered at x=0 with width 1.
    
    Parameters:
        x : float or array
            Input value(s)
    
    Returns:
        Box function: 1 if -0.5 <= x < 0.5, 0 otherwise
    """
    return 1 * (x + 0.5 >= 0) - 1 * (x - 0.5 >= 0)


def heaviside_function(x):
    """
    Heaviside step function.
    
    Parameters:
        x : float or array
            Input value(s)
    
    Returns:
        0 if x < 0, 1 if x >= 0
    """
    return 0 + x >= 0


def ramp_function(x):
    """
    Smooth ramp function that transitions from 0 to 1.
    
    Parameters:
        x : float or array
            Input value(s)
    
    Returns:
        Ramp function value
    """
    return x * boxcar_function(x - 1/2) + heaviside_function(x - 1)


# ============================================================================
# SECTION 2: FAULT PROBLEM CLASS
# ============================================================================

class FaultProblem:
    """
    Container class for all fault and material properties.
    
    This class organizes all the parameters needed for the fault evolution
    simulation, including bulk material properties, frictional parameters,
    and stress kernels.
    """
    
    def __init__(self, density, shear_wave_speed, shear_modulus, 
                 stress_kernel, radiation_damping,
                 friction_ref, reference_slip_rate, 
                 friction_param_a, friction_param_b,
                 critical_slip_distance, normal_stress, 
                 plate_velocity, degrees_of_freedom):
        """
        Initialize fault problem with all necessary parameters.
        
        Bulk Properties:
            density : float
                Rock density (kg/m³)
            shear_wave_speed : float
                Shear wave speed (m/s)
            shear_modulus : float
                Elastic shear modulus (MPa)
            stress_kernel : array
                Stress interaction kernel matrix (M x M)
            radiation_damping : float
                Radiation damping coefficient (MPa·s/m)
        
        Friction Properties:
            friction_ref : array
                Reference friction coefficient
            reference_slip_rate : array
                Reference slip rate V₀ (m/s)
            friction_param_a : array
                Direct effect parameter 'a' in rate-state friction
            friction_param_b : array
                Evolution effect parameter 'b' in rate-state friction
            critical_slip_distance : array
                Critical slip distance Dc (m)
            normal_stress : array
                Effective normal stress σ (MPa)
            plate_velocity : array
                Tectonic plate velocity (m/s)
        
        Numerical Parameters:
            degrees_of_freedom : int
                Number of state variables per grid cell
        """
        # Bulk material properties
        self.rho = density
        self.cs = shear_wave_speed
        self.mu = shear_modulus
        self.Kk = stress_kernel
        self.eta = radiation_damping
        
        # Frictional parameters (rate-and-state friction)
        self.fo = friction_ref
        self.Vo = reference_slip_rate
        self.cca = friction_param_a
        self.ccb = friction_param_b
        self.Drs = critical_slip_distance
        self.sigma = normal_stress
        self.Vpl = plate_velocity
        
        # Numerical setup
        self.dgf = degrees_of_freedom


# ============================================================================
# SECTION 3: GREEN'S FUNCTIONS (STRESS AND DISPLACEMENT KERNELS)
# ============================================================================

"""
Elastostatic Green's functions for a vertical fault in a half-space.

These functions compute stress and displacement at receiver locations due to
uniform slip on rectangular patches. They account for the free surface using
the method of images (Okada 1985, 1992).

Coordinate System:
    x1 : along-strike direction (out of plane)
    x2 : across-fault horizontal direction
    x3 : down-dip direction (depth, positive downward)

For 2D antiplane shear:
    - Only u1 (strike-slip) displacement is non-zero
    - u2 = u3 = 0
    - We compute σ₁₂ (shear stress on fault)
"""

pi = np.pi


def stress_kernel_sigma12(receiver_x2, receiver_x3, source_x2, source_x3, 
                          patch_width, shear_modulus):
    """
    Compute shear stress σ₁₂ at receiver due to unit slip on source patch.
    
    This kernel includes the effect of the free surface through the method
    of images (four terms: direct, surface image, bottom image, and 
    bottom-surface image).
    
    Parameters:
        receiver_x2 : float or array
            Horizontal position of receiver (m)
        receiver_x3 : float or array
            Depth of receiver (m)
        source_x2 : float
            Horizontal position of source patch (m)
        source_x3 : float
            Top depth of source patch (m)
        patch_width : float
            Down-dip width of source patch (m)
        shear_modulus : float
            Shear modulus (MPa)
    
    Returns:
        Shear stress contribution (MPa per meter of slip)
    """
    dx2 = receiver_x2 - source_x2
    dx3_top = receiver_x3 - source_x3
    dx3_bot = receiver_x3 - source_x3 - patch_width
    dx3_img_top = receiver_x3 + source_x3
    dx3_img_bot = receiver_x3 + source_x3 + patch_width
    
    # Four terms from method of images
    term1 = -dx3_top / (dx2**2 + dx3_top**2)
    term2 = +dx3_img_top / (dx2**2 + dx3_img_top**2)
    term3 = +dx3_bot / (dx2**2 + dx3_bot**2)
    term4 = -dx3_img_bot / (dx2**2 + dx3_img_bot**2)
    
    sigma12 = shear_modulus * (term1 + term2 + term3 + term4) / (2 * pi)
    
    return sigma12


def displacement_kernel_u1(receiver_x2, receiver_x3, source_x2, source_x3, 
                           patch_width):
    """
    Compute strike-slip displacement u₁ at receiver due to unit slip on source.
    
    Uses arctangent formulation with method of images for free surface.
    
    Parameters:
        receiver_x2 : float or array
            Horizontal position of receiver (m)
        receiver_x3 : float or array
            Depth of receiver (m)
        source_x2 : float
            Horizontal position of source patch (m)
        source_x3 : float
            Top depth of source patch (m)
        patch_width : float
            Down-dip width of source patch (m)
    
    Returns:
        Displacement (meters per meter of slip)
    """
    dx2 = receiver_x2 - source_x2
    dx3_top = receiver_x3 - source_x3
    dx3_bot = receiver_x3 - source_x3 - patch_width
    dx3_img_top = receiver_x3 + source_x3
    dx3_img_bot = receiver_x3 + source_x3 + patch_width
    
    # Four arctangent terms from method of images
    term1 = np.arctan(dx3_top / dx2)
    term2 = -np.arctan(dx3_img_top / dx2)
    term3 = -np.arctan(dx3_bot / dx2)
    term4 = np.arctan(dx3_img_bot / dx2)
    
    u1 = (term1 + term2 + term3 + term4) / (2 * pi)
    
    return u1


# ============================================================================
# SECTION 4: SPATIAL DISCRETIZATION
# ============================================================================

print("=" * 80)
print("SETTING UP SPATIAL DISCRETIZATION")
print("=" * 80)

# Domain geometry
fault_depth = 40e3  # Total depth extent of frictional domain (m)
num_cells = 400     # Number of computational cells along fault

# Grid spacing
cell_size = fault_depth / num_cells

# Cell positions
cell_top_depths = np.linspace(0, num_cells - 1, num_cells) * cell_size
cell_center_depths = cell_top_depths + 0.5 * cell_size
cell_boundary_depths = np.linspace(0, num_cells, num_cells + 1) * cell_size

# Patch widths (uniform grid)
patch_widths = np.ones((num_cells, 1)) * cell_size

print(f"Fault depth: {fault_depth/1e3:.1f} km")
print(f"Number of cells: {num_cells}")
print(f"Cell size: {cell_size:.1f} m")

# Virtual GPS receivers on the surface
num_gps_stations = 100
epsilon = 1e-6  # Small offset to avoid singularity at x2 = 0

# GPS stations on right side of fault (positive x2)
gps_x2_right = 200e3 * np.tan(epsilon + np.linspace(0, num_gps_stations/2, 
                                                     num_gps_stations) * pi / 
                              (2 * num_gps_stations))

# GPS stations on left side of fault (negative x2)
gps_x2_left = -200e3 * np.tan(epsilon + np.linspace(0, num_gps_stations/2, 
                                                      num_gps_stations) * pi / 
                               (2 * num_gps_stations))

# Combine into single array (left to right across fault)
gps_positions_x2 = np.concatenate((np.flipud(gps_x2_left), gps_x2_right), 
                                   axis=0)

print(f"GPS stations: {len(gps_positions_x2)}")
print()


# ============================================================================
# SECTION 5: COMPUTE INTERACTION KERNELS
# ============================================================================

print("=" * 80)
print("COMPUTING STRESS AND DISPLACEMENT KERNELS")
print("=" * 80)

# Material properties
density = 2670              # Rock density (kg/m³)
shear_wave_speed = 3464     # Shear wave speed (m/s)
shear_modulus = density * shear_wave_speed**2 / 1e6  # Shear modulus (MPa)

print(f"Density: {density} kg/m³")
print(f"Shear wave speed: {shear_wave_speed} m/s")
print(f"Shear modulus: {shear_modulus:.1f} MPa")

# Initialize kernel matrices
stress_interaction_kernel = np.zeros((num_cells, num_cells))
surface_displacement_kernel = np.zeros((len(gps_positions_x2), num_cells))

# Compute kernels
print("\nComputing interaction kernels...")
for source_idx in range(num_cells):
    # Stress kernel: stress at cell centers due to slip on each patch
    # Receiver coordinates: on-fault at cell centers (x2=0, x3=cell_center_depths)
    # Source coordinates: top of slip patches at cell_top_depths
    stress_interaction_kernel[:, source_idx] = stress_kernel_sigma12(
        receiver_x2=0,
        receiver_x3=cell_center_depths,
        source_x2=0,
        source_x3=cell_top_depths[source_idx],
        patch_width=patch_widths[source_idx],
        shear_modulus=shear_modulus
    )
    
    # Displacement kernel: surface displacement at GPS stations
    # Receiver coordinates: surface (x3=0) at GPS positions
    # Source coordinates: fault patches
    surface_displacement_kernel[:, source_idx] = displacement_kernel_u1(
        receiver_x2=gps_positions_x2,
        receiver_x3=0,
        source_x2=0,
        source_x3=cell_top_depths[source_idx],
        patch_width=patch_widths[source_idx]
    )

print("Kernels computed successfully!")
print()


# ============================================================================
# SECTION 6: FAULT FRICTIONAL PROPERTIES
# ============================================================================

print("=" * 80)
print("DEFINING FAULT FRICTIONAL PROPERTIES")
print("=" * 80)

# Reference friction coefficient
friction_coefficient_ref = 0.6 * np.ones(cell_center_depths.size)

# Rate-and-state friction parameters
# a: direct effect (velocity strengthening contribution)
# b: evolution effect (state variable contribution)
# Velocity-weakening (VW) where b > a, velocity-strengthening (VS) where a > b

friction_a = (1e-2 + ramp_function((cell_center_depths - 15e3) / 3e3) * 
              (0.025 - 0.01))
friction_b = 0.015 * np.ones(cell_center_depths.size)

# Effective normal stress (MPa)
effective_normal_stress = 50.0 * np.ones(cell_center_depths.size)

# Critical slip distance (characteristic weakening distance)
critical_slip_distance = 8e-3 * np.ones(cell_center_depths.size)

# Plate loading rate (tectonic velocity)
plate_rate = 1e-9 * np.ones(cell_center_depths.size)  # m/s

# Reference slip rate
reference_velocity = 1e-6 * np.ones(cell_center_depths.size)  # m/s

# Radiation damping coefficient (quasi-dynamic approximation)
radiation_damping = shear_modulus / (2 * shear_wave_speed)

print(f"Reference friction: {friction_coefficient_ref[0]:.3f}")
print(f"Friction parameter 'a': {friction_a[0]:.4f} - {friction_a[-1]:.4f}")
print(f"Friction parameter 'b': {friction_b[0]:.4f}")
print(f"Normal stress: {effective_normal_stress[0]:.1f} MPa")
print(f"Critical slip distance: {critical_slip_distance[0]*1e3:.1f} mm")
print(f"Plate rate: {plate_rate[0]*1e9:.2f} mm/yr")
print(f"Radiation damping: {radiation_damping:.1f} MPa·s/m")
print()


# ============================================================================
# SECTION 7: CHARACTERISTIC SCALES AND VALIDATION
# ============================================================================

print("=" * 80)
print("CHARACTERISTIC SCALES AND VALIDATION")
print("=" * 80)

# Identify velocity-weakening (VW) region (where b > a)
vw_indices = np.argwhere(friction_b > friction_a)[:, 0]

# Critical nucleation size (Rice et al., 2001)
# h* = (π/2) * (G * b * Dc) / [(b-a)² * σ]
critical_nucleation_size = np.min(
    pi / 2 * shear_modulus * critical_slip_distance[vw_indices] * 
    friction_b[vw_indices] / 
    (friction_b[vw_indices] - friction_a[vw_indices])**2 / 
    effective_normal_stress[vw_indices]
)

# Size of velocity-weakening region
vw_region_size = (cell_center_depths[vw_indices[-1]] - 
                  cell_center_depths[vw_indices[0]])

# Quasi-static cohesive zone size
# Λ = (9π/32) * (G * Dc) / (b * σ)
cohesive_zone_size = np.min(
    9 / 32 * pi * shear_modulus * critical_slip_distance[vw_indices] / 
    friction_b[vw_indices] / effective_normal_stress[vw_indices]
)

# Estimate of earthquake recurrence interval
# T ≈ 5 * (b-a) * σ * L / (2 * μ * Vpl)
recurrence_time = (
    5 * np.mean((friction_b[vw_indices] - friction_a[vw_indices]) * 
                effective_normal_stress[vw_indices]) * 
    0.5 * vw_region_size / 
    (shear_modulus * np.mean(plate_rate[vw_indices]))
)

# Validation: check grid resolution
print(f"Grid size: {cell_size:.2f} m")
print(f"Velocity-weakening zone: {vw_region_size/1e3:.2f} km")
print(f"Critical nucleation size h*: {critical_nucleation_size:.2f} m")
print(f"Quasi-static cohesive zone: {cohesive_zone_size:.2f} m")
print(f"Estimated recurrence time: {recurrence_time/3.15e7:.2f} years")
print()

# Check that grid is sufficiently refined
if cell_size < cohesive_zone_size / 3:
    print("Grid resolution is adequate (cell_size < cohesive_zone/3)")
else:
    print("WARNING: Grid may be too coarse for accurate results!")
print()


# ============================================================================
# SECTION 8: GOVERNING EQUATIONS
# ============================================================================

def rate_state_ode_system(time, state_vector, fault_params):
    """
    Compute time derivatives for the fault evolution ODE system.
    
    This function implements the regularized Dieterich-Ruina rate-and-state
    friction law with the aging state evolution law, using the quasi-dynamic
    approximation with radiation damping.
    
    State Vector Layout:
        state_vector = [slip₁, stress₁, log(θ₁V₀/Dc), log(V₁/V₀),
                       slip₂, stress₂, log(θ₂V₀/Dc), log(V₂/V₀),
                       ..., slip_M, stress_M, log(θ_MV₀/Dc), log(V_M/V₀)]
    
    where for each cell i:
        slip_i : accumulated slip (m)
        stress_i : shear stress (MPa)
        log(θ_iV₀/Dc) : logarithm of normalized state variable
        log(V_i/V₀) : logarithm of normalized slip rate
    
    Friction Law (Regularized Form):
        τ = a σ asinh[V/(2V₀) exp((f₀ + b ψ)/a)]
    
    Quasi-Dynamic Stress Balance:
        τ = τ_load + K(δ - Vpl·t) - η·V
    
    where:
        τ : shear stress
        a, b : rate-state friction parameters
        σ : effective normal stress
        V : slip rate
        V₀ : reference slip rate
        f₀ : reference friction coefficient
        ψ = ln(θV₀/Dc) : normalized state variable
        θ : state variable
        Dc : critical slip distance
        K : stress interaction kernel
        δ : slip
        Vpl : plate loading rate
        η : radiation damping coefficient
    
    State Evolution Law (Aging):
        dθ/dt = 1 - V·θ/Dc
    
    which in logarithmic form becomes:
        dψ/dt = (V₀ exp(-ψ) - V) / Dc
    
    Parameters:
        time : float
            Current time (not used in autonomous system)
        state_vector : array
            Current state [slip, stress, log(θV₀/Dc), log(V/V₀)] for all cells
        fault_params : FaultProblem
            Object containing all fault and material parameters
    
    Returns:
        state_derivatives : array
            Time derivatives [dslip/dt, dstress/dt, dψ/dt, d(log V)/dt]
    """
    # Extract state variables from state vector
    # State variable: ψ = ln(θV₀/Dc)
    log_state_normalized = state_vector[2::fault_params.dgf]
    
    # Slip rate: V = V₀ exp(log(V/V₀))
    slip_rate = fault_params.Vo * np.exp(state_vector[3::fault_params.dgf])
    
    # Initialize derivatives array
    state_derivatives = np.zeros(state_vector.shape)
    
    # -----------------------------------------------------------------------
    # Slip evolution: dδ/dt = V
    # -----------------------------------------------------------------------
    state_derivatives[0::fault_params.dgf] = slip_rate
    
    # -----------------------------------------------------------------------
    # State variable evolution (aging law): dψ/dt = (V₀ exp(-ψ) - V) / Dc
    # -----------------------------------------------------------------------
    state_rate = ((fault_params.Vo * np.exp(-log_state_normalized) - slip_rate) / 
                  fault_params.Drs)
    state_derivatives[2::fault_params.dgf] = state_rate
    
    # -----------------------------------------------------------------------
    # Slip velocity evolution: d(log V)/dt
    # -----------------------------------------------------------------------
    # Stress change from slip rate perturbations
    stress_perturbation = np.matmul(fault_params.Kk, 
                                    (slip_rate - fault_params.Vpl))
    
    # Regularization factor for friction law
    # α = V / (2V₀) exp[-(f₀ + b ψ) / a]
    alpha = (2 * fault_params.Vo / slip_rate * 
             np.exp(-(fault_params.fo + fault_params.ccb * 
                     log_state_normalized) / fault_params.cca))
    
    # Q factor: Q = 1 / sqrt(1 + α²)
    # This appears in the derivative of the regularized friction law
    regularization_factor = 1 / np.sqrt(1 + alpha**2)
    
    # Rate of change of log(V/V₀)
    # Derived from equating d(friction)/dt = d(stress)/dt
    numerator = (stress_perturbation - 
                 fault_params.ccb * fault_params.sigma * state_rate * 
                 regularization_factor)
    denominator = (fault_params.cca * fault_params.sigma * 
                   regularization_factor + 
                   fault_params.eta * slip_rate)
    
    state_derivatives[3::fault_params.dgf] = numerator / denominator
    
    # -----------------------------------------------------------------------
    # Stress evolution: dτ/dt
    # -----------------------------------------------------------------------
    state_derivatives[1::fault_params.dgf] = (stress_perturbation - 
                                              fault_params.eta * slip_rate * 
                                              state_derivatives[3::fault_params.dgf])
    
    return state_derivatives


# ============================================================================
# SECTION 9: INITIAL CONDITIONS
# ============================================================================

print("=" * 80)
print("SETTING UP INITIAL CONDITIONS")
print("=" * 80)

# Number of state variables per grid cell
degrees_of_freedom = 4  # [slip, stress, log(θV₀/Dc), log(V/V₀)]

# Create fault problem object
fault_system = FaultProblem(
    density=density,
    shear_wave_speed=shear_wave_speed,
    shear_modulus=shear_modulus,
    stress_kernel=stress_interaction_kernel,
    radiation_damping=radiation_damping,
    friction_ref=friction_coefficient_ref,
    reference_slip_rate=reference_velocity,
    friction_param_a=friction_a,
    friction_param_b=friction_b,
    critical_slip_distance=critical_slip_distance,
    normal_stress=effective_normal_stress,
    plate_velocity=plate_rate,
    degrees_of_freedom=degrees_of_freedom
)

# Initialize state vector (steady-state sliding at plate rate with zero slip)
initial_state = np.zeros(num_cells * degrees_of_freedom)

# Initial slip: zero everywhere
initial_state[0::degrees_of_freedom] = np.zeros(num_cells)

# Initial stress: steady-state stress for sliding at plate rate
# τ_ss = a σ asinh[Vpl/(2V₀) exp((f₀ + b ln(V₀/Vpl))/a)] + η·Vpl
initial_state[1::degrees_of_freedom] = (
    np.max(friction_a) * effective_normal_stress * 
    np.arcsinh(plate_rate / reference_velocity / 2 * 
               np.exp((friction_coefficient_ref + friction_b * 
                      np.log(reference_velocity / plate_rate)) / 
                     np.max(friction_a))) + 
    radiation_damping * plate_rate
)

# Initial state variable: ψ_ss such that friction balances stress
# Derived from inverting the friction law at steady state
initial_state[2::degrees_of_freedom] = (
    friction_a / friction_b * 
    np.log(2 * reference_velocity / plate_rate * 
           np.sinh((initial_state[1::degrees_of_freedom] - 
                   radiation_damping * plate_rate) / 
                  friction_a / effective_normal_stress)) - 
    friction_coefficient_ref / friction_b
)

# Initial velocity: plate rate everywhere
initial_state[3::degrees_of_freedom] = np.log(plate_rate / reference_velocity)

print(f"State vector size: {len(initial_state)}")
print(f"Degrees of freedom per cell: {degrees_of_freedom}")
print(f"Initial slip: {initial_state[0]:.2e} m")
print(f"Initial stress range: {np.min(initial_state[1::degrees_of_freedom]):.2f} - "
      f"{np.max(initial_state[1::degrees_of_freedom]):.2f} MPa")
print(f"Initial velocity: {plate_rate[0]:.2e} m/s")
print()


# ============================================================================
# SECTION 10: TIME INTEGRATION
# ============================================================================

print("=" * 80)
print("SOLVING ODE SYSTEM")
print("=" * 80)

# Time integration parameters
time_initial = 0                   # Initial time (s)
time_final = 500 * 3.15e7          # Final time (xxx years in seconds)
max_time_step = 3.15e7             # Maximum time step (1 year)

# Tolerances for adaptive time stepping
relative_tolerance = 1e-8
absolute_tolerance = 1e-6

print(f"Simulation duration: {time_final/3.15e7:.1f} years")
print(f"Maximum time step: {max_time_step/3.15e7:.2f} years")
print(f"Relative tolerance: {relative_tolerance}")
print(f"Absolute tolerance: {absolute_tolerance}")
print()

# Define ODE function wrapper
def ode_function(t, y):
    """Wrapper function for ODE solver."""
    return rate_state_ode_system(t, y, fault_system)

# Solve the ODE system using adaptive Runge-Kutta method
print("Starting time integration...")
start_time = time.time()

solution = sp.integrate.solve_ivp(
    fun=ode_function,
    t_span=[time_initial, time_final],
    y0=initial_state,
    method='RK45',
    max_step=max_time_step,
    rtol=relative_tolerance,
    atol=absolute_tolerance,
    vectorized=False,
    first_step=1e-5
)

end_time = time.time()
computation_time = end_time - start_time

print(f"Integration complete!")
print(f"Computation time: {computation_time:.3f} seconds")
print(f"Number of time steps: {len(solution.t)}")
print(f"Average time step: {np.mean(np.diff(solution.t))/3.15e7:.4f} years")
print()


# ============================================================================
# SECTION 11: POST-PROCESSING AND ANALYSIS
# ============================================================================

print("=" * 80)
print("POST-PROCESSING RESULTS")
print("=" * 80)

# Extract solution arrays
time_array = np.asarray(solution.t)
state_history = np.transpose(np.asarray(solution.y))

# Extract physical quantities from state vector
slip_rate_history = (reference_velocity * 
                     np.exp(state_history[:, 3::degrees_of_freedom]))  # m/s
stress_history = state_history[:, 1::degrees_of_freedom]  # MPa

# Compute maximum slip rate at each time step
max_slip_rate = np.zeros(time_array.size)
for time_idx in range(time_array.size):
    max_slip_rate[time_idx] = np.max(slip_rate_history[time_idx, :])

# Extract slip rate at center of velocity-weakening zone
center_vw_index = int(np.floor(num_cells / 2))
center_slip_rate = slip_rate_history[:, center_vw_index]

# Compute surface displacements at GPS stations
slip_history = state_history[:, 0::degrees_of_freedom]
gps_displacement = np.matmul(surface_displacement_kernel, 
                             np.transpose(slip_history))

print(f"Maximum slip rate: {np.max(max_slip_rate):.2e} m/s")
print(f"Minimum slip rate: {np.min(max_slip_rate):.2e} m/s")
print(f"Maximum surface displacement: {np.max(gps_displacement):.3f} m")
print()


# ============================================================================
# SECTION 12: VISUALIZATION
# ============================================================================

print("=" * 80)
print("GENERATING FIGURES")
print("=" * 80)

# ---------------------------------------------------------------------------
# Figure 1: Fault Slip Rate Evolution (Space-Time Diagrams)
# ---------------------------------------------------------------------------

fig1, axes1 = plt.subplots(2, 1, figsize=[9, 9], constrained_layout=True)

# Panel A: Space-time color map of slip rate
ax = axes1[0]
colormap = ax.pcolormesh(
    time_array / 3.15e7,
    cell_center_depths / 1e3,
    np.transpose(np.log10(slip_rate_history)),
    cmap='viridis',
    shading='nearest'
)
ax.set_title('Log₁₀ Slip Velocity (m/s)', fontsize=12, fontweight='bold')
ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Depth (km)', fontsize=11)
ax.invert_yaxis()
colorbar1 = fig1.colorbar(colormap, ax=ax)
colorbar1.set_label('Log₁₀ Slip Rate (m/s)', fontsize=10)

# Panel B: Maximum slip rate time series
ax = axes1[1]
ax.plot(time_array / 3.15e7, np.log10(max_slip_rate), 'b-', linewidth=1.5)
ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Log₁₀ Max Slip Rate (m/s)', fontsize=11)
ax.set_title('Maximum Slip Rate Evolution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.show()

# ---------------------------------------------------------------------------
# Figure 2: Fault Slip Rate Evolution (Time-Step Domain)
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(2, 1, figsize=[9, 9], constrained_layout=True)

time_step_array = np.linspace(0, time_array.size - 1, time_array.size)

# Panel A: Space-time color map (time-step domain)
ax = axes2[0]
colormap = ax.pcolormesh(
    time_step_array,
    cell_center_depths / 1e3,
    np.transpose(np.log10(slip_rate_history)),
    cmap='viridis',
    shading='nearest'
)
ax.set_title('Log₁₀ Slip Velocity (Time-Step Domain)', fontsize=12, 
             fontweight='bold')
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Depth (km)', fontsize=11)
ax.invert_yaxis()
colorbar2 = fig2.colorbar(colormap, ax=ax)
colorbar2.set_label('Log₁₀ Slip Rate (m/s)', fontsize=10)

# Panel B: Maximum slip rate vs time step
ax = axes2[1]
ax.plot(time_step_array, np.log10(max_slip_rate), 'r-', linewidth=1.5)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Log₁₀ Max Slip Rate (m/s)', fontsize=11)
ax.set_title('Maximum Slip Rate (Time-Step Domain)', fontsize=12, 
             fontweight='bold')
ax.grid(True, alpha=0.3)

plt.show()

# ---------------------------------------------------------------------------
# Figure 3: Synthetic GPS Time Series
# ---------------------------------------------------------------------------

# Setup GPS sampling
gps_sampling_interval = 2.628e6          # 1 month in seconds
gps_record_start = 150 * 3.15e7          # Start at 150 years
gps_record_duration = 400 * 3.15e7       # 400 year record
gps_time_array = gps_record_start + np.arange(0, gps_record_duration, 
                                                gps_sampling_interval)

# Find indices in solution corresponding to GPS sampling times
gps_time_indices = np.zeros(gps_time_array.size, dtype=int)
for sample_idx in range(gps_time_array.size):
    gps_time_indices[sample_idx] = (
        np.searchsorted(time_array, gps_time_array[sample_idx], side="left") - 1
    )

# Create figure
fig3, axes3 = plt.subplots(3, 1, figsize=[9, 10], constrained_layout=True)

# Setup colormap for time-coded GPS profiles
num_profiles = len(gps_time_array)
time_colormap = plt.get_cmap("jet", num_profiles)
time_norm = mpl.colors.Normalize(
    vmin=min(gps_time_array / 3.15e7),
    vmax=max(gps_time_array / 3.15e7)
)
scalar_mappable = plt.cm.ScalarMappable(norm=time_norm, cmap=time_colormap)

# Panel A: Surface displacement profiles across fault
ax = axes3[0]
for sample_idx in range(num_profiles):
    time_idx = int(gps_time_indices[sample_idx])
    displacement_profile = (gps_displacement[:, time_idx] - 
                           gps_displacement[:, int(gps_time_indices[0])])
    ax.plot(gps_positions_x2 / 1e3, displacement_profile, 
            color=time_colormap(sample_idx), linewidth=1)

colorbar3 = fig3.colorbar(scalar_mappable, ax=ax)
colorbar3.set_label('Time (years)', fontsize=10)
ax.set_xlim(-100, 100)
ax.set_xlabel('Distance Across Fault (km)', fontsize=11)
ax.set_ylabel('Cumulative Displacement (m)', fontsize=11)
ax.set_title('Surface Displacement Profiles (100 Year Time Series)', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel B: Time series at near-field GPS station (~5 km from fault)
gps_near_index = np.argmin(np.absolute(gps_positions_x2 - 5e3))
ax = axes3[1]
ax.plot(time_array / 3.15e7, gps_displacement[gps_near_index, :], 
        'b-', linewidth=1.5)
ax.set_xlim(min(time_array / 3.15e7), max(time_array / 3.15e7))
ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Displacement (m)', fontsize=11)
ax.set_title(f'GPS Station at {gps_positions_x2[gps_near_index]/1e3:.1f} km '
             f'from Fault', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel C: Time series at far-field GPS station (~100 km from fault)
gps_far_index = np.argmin(np.absolute(gps_positions_x2 - 100e3))
ax = axes3[2]
ax.plot(time_array / 3.15e7, gps_displacement[gps_far_index, :], 
        'r-', linewidth=1.5)
ax.set_xlim(min(time_array / 3.15e7), max(time_array / 3.15e7))
ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Displacement (m)', fontsize=11)
ax.set_title(f'GPS Station at {gps_positions_x2[gps_far_index]/1e3:.1f} km '
             f'from Fault', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

fig3.get_layout_engine().set(hspace=0.1)

plt.show()

print("All figures generated successfully!")
print()
print("=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)