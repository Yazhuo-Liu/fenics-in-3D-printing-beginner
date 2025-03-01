from fenics import *
import numpy as np
import os
from mpi4py import MPI
import glob

# Enable optimization for compilation
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 2
parameters["form_compiler"]["representation"] = "uflacs"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank > 0: # Suppress output from non-master processes
    set_log_level(LogLevel.WARNING)

if rank == 0: # Only the master process creates the directory
    if not os.path.exists("results"):
        os.makedirs("results")
    else:
        # Clear existing files
        files = glob.glob('results/*')
        for f in files:
            os.remove(f)
comm.barrier() # Ensure all processes wait until the directory is created
    
# Material Properties
rho = Constant(2700.0)     # Density [kg/m^3]
cp = Constant(900.0)       # Specific heat capacity [J/(kg·K)]
k_bulk = Constant(237.0)   # Base thermal conductivity [W/(m·K)]
alpha = Constant(1e-3)     # Thermal conductivity temperature coefficient [1/K]

# Boundary condition parameters
h = 10.0                   # Convection coefficient [W/(m^2·K)]
T_inf = 300.0              # Ambient temperature [K]
epsilon = 0.5              # Emissivity
sigma_SB = 5.67e-8         # Stefan-Boltzmann constant

# Double ellipsoid heat source parameters
Q0 = 100.0                 # Laser power [W]
v = 1.0                    # Scanning speed [m/s]
a1, a2, b, c = 50e-6, 50e-6, 50e-6, 50e-6  # Ellipsoid parameters
f1, f2 = 0.6, 1.4          # Power distribution coefficients

# Define computational domain (unit: meters)
Lx, Ly, Lz = 1000e-6, 600e-6, 300e-6
# mesh = BoxMesh(comm, Point(0, 0, 0), Point(Lx, Ly, Lz), 100, 60, 30)
mesh = BoxMesh(Point(0, 0, 0), Point(Lx, Ly, Lz), 100, 60, 30)

# Time parameters
t_total = Lx/v             # Total time [s]
dt = t_total/100           # Time step [s]
num_steps = int(t_total/dt)

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define test functions and unknown functions
w = TestFunction(V)
T = Function(V)     # Temperature field at current time step
T_n = Function(V)    # Temperature field at previous time step

# Initial condition
T_n = interpolate(Constant(T_inf), V)  # Initial temperature field

# Boundary condition definitions
# Define boundary locations
def top(x, on_boundary):
    return near(x[2], Lz) and on_boundary

def bottom(x, on_boundary):
    return near(x[2], 0.) and on_boundary

def walls(x, on_boundary):
    left = near(x[0], 0.) and on_boundary
    right = near(x[0], Lx) and on_boundary
    front = near(x[1], 0.) and on_boundary
    back = near(x[1], Ly) and on_boundary
    return left | right | front | back

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
Top = AutoSubDomain(top)
Top.mark(boundaries, 1)
Bottom = AutoSubDomain(bottom)
Bottom.mark(boundaries, 2)
Walls = AutoSubDomain(walls)
Walls.mark(boundaries, 3)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)  # Redefine the measure 'ds' with subdomains

bc = DirichletBC(V, Constant(300.0), bottom)
bcs = [bc]

# Temperature-dependent thermal conductivity
def thermal_conductivity(T):
    return k_bulk * (1 + alpha*(T - 300))

# Define volumetric heat source for laser scanning
class HeatSource(UserExpression):
    def __init__(self, position, velocity, Q0, f1, f2, a1, a2, b, c, t, **kwargs):
        super().__init__(**kwargs)
        
        # Validate that position is a vector of length 3
        if len(position) != 3:
            raise ValueError("Position must be a vector with exactly 3 elements (x, y, z).")
        
        # Validate that velocity is a vector of length 3
        if len(velocity) != 3:
            raise ValueError("Velocity must be a vector with exactly 3 elements (vx, vy, vz).")
        
        self.position = np.array(position, dtype=float)  # Initial position [x0, y0, z0]
        self.velocity = np.array(velocity, dtype=float)  # Velocity [vx, vy, vz]
        self.Q0 = Q0
        self.f1 = f1
        self.f2 = f2
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.c = c
        self.t = t
        
    def eval(self, value, x):
        
        laser_center = self.position + self.velocity * self.t  # Laser center position
        
        # Relative position
        x_prime = x - laser_center
        
        # Calculate heat source intensity
        if x_prime[0] >= 0:
            coeff = 6*sqrt(3) * self.f1 * self.Q0/(self.a1*self.b*self.c*np.pi*sqrt(np.pi))
            exponent = -3*((x_prime[0])**2/self.a1**2 + x_prime[1]**2/self.b**2 + x_prime[2]**2/self.c**2)
        else:
            coeff = 6*sqrt(3) * self.f2 * self.Q0/(self.a2*self.b*self.c*np.pi*sqrt(np.pi))
            exponent = -3*((x_prime[0])**2/self.a2**2 + x_prime[1]**2/self.b**2 + x_prime[2]**2/self.c**2)
            
        value[0] = coeff * exp(exponent)
    
    def value_shape(self):
        return ()

# Create heat source object
position = [-a1, Ly/2, Lz]
velocity = [v, 0.0, 0.0]
q_dot = HeatSource(position, velocity, Q0, f1, f2, a1, a2, b, c, t=0, degree=1)

# Define radiative heat flux
q_bar_conv = h * (T_inf - T)  # Convective heat flux
q_bar_rad = epsilon * sigma_SB * (T_inf**4 - T**4)  # Radiative heat flux

# Define variational form
F = w * rho * cp * (T - T_n) / dt * dx \
    + inner(grad(w), thermal_conductivity(T)*grad(T))*dx \
    - w * q_dot * dx \
    + w * (q_bar_rad + q_bar_conv) * ds(3)

# Create nonlinear problem and solver
problem = NonlinearVariationalProblem(F, T, bc, J=derivative(F, T))
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relaxation_parameter'] = 1.0
solver.parameters["newton_solver"]["linear_solver"] = "cg"
solver.parameters["newton_solver"]["maximum_iterations"] = 50


# Time-stepping loop ===========================================================
t = 0.0  # Initialize time

# Create file to save results
file = XDMFFile("results/temperature.xdmf")
file.parameters["flush_output"] = True
file.write(mesh)
file.write(T_n, t)


# Initialize heat source object (Note: time parameter must be initialized first)
q_dot.t = t

# Progress counter
progress_counter = 0

for step in range(num_steps):
    # Update time
    t += dt
    
    # Print progress information
    if rank == 0:
        print(f"\n======= Computing time step {step + 1}/{num_steps} [t = {t:.6f} s] =======")
    
    # Update heat source position (Critical step!)
    # --------------------------------------------------
    q_dot.t = t  # Update heat source time parameter
    
    # Solve nonlinear problem
    # --------------------------------------------------
    try:
        # Call solver
        solver.solve()
        
        # Check solution validity
        max_T = T.vector().max()
        min_T = T.vector().min()
        if max_T > 5000 or min_T < 0:
            print(f"Warning: Temperature solution out of physical range! Max temperature: {max_T} K, Min temperature: {min_T} K")
            break
        
    except Exception as e:
        print(f"Solving failed at time step {step}, error message:")
        print(str(e))
        break
    
    # Update solution from the previous time step
    # --------------------------------------------------
    T_n.assign(T)
    
    # Save results (every 10 steps)
    # --------------------------------------------------
    if step + 1 % 5 == 0:
        T.rename("Temperature", "Temperature")
        file.write(T, t)
        progress_counter += 1

# Save final result
file.close()
if rank == 0:
    print("\nComputation completed! Results saved to results/temperature.pvd")
