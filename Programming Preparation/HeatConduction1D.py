from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Suppress FEniCS log output by setting the log level to 'ERROR'
set_log_level(40)  # 40=ERROR

# Alternatively, to completely deactivate logging, uncomment the following line:
# set_log_active(False)

# Define physical parameters
alpha = 1.0                # Thermal diffusivity (W/m·K)
rho = 1.0                  # Density (kg/m³)
cp = 1.0                   # Specific heat capacity (J/kg·K)
Q = Constant(0.0)          # Internal heat generation (W/m³)

# Define domain parameters
L = 1.0                    # Length of the rod (m)
nx = 50                    # Number of finite elements

# Define boundary conditions
q0 = Constant(100.0)       # Prescribed heat flux at x=0 (W/m²)
T0 = Constant(300.0)       # Prescribed temperature at x=L (K)

# Time-stepping parameters
T_final = 1.0              # Final time (s)
num_steps = 50             # Number of time steps
dt = T_final / num_steps   # Time step size (s)

# Create mesh and define function space
mesh = IntervalMesh(nx, 0, L)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary identification functions
def boundary_neumann(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

def boundary_dirichlet(x, on_boundary):
    return on_boundary and near(x[0], L)

# Define boundary condition for Dirichlet at x=L
bc_dirichlet = DirichletBC(V, T0, boundary_dirichlet)

# Define initial condition
T_n = Function(V)
T_n.assign(Constant(300.0))  # Initial temperature distribution

# Define trial and test functions
T = TrialFunction(V)
v = TestFunction(V)

# Define measures for boundary integration
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
neumann_marker = 1
dirichlet_marker = 2

class NeumannBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

# Mark boundaries
NeumannBoundary().mark(boundary_markers, neumann_marker)
DirichletBoundary().mark(boundary_markers, dirichlet_marker)

# Define measures with boundary markers
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Define the variational problem
a = (rho * cp / dt) * T * v * dx + alpha * dot(grad(T), grad(v)) * dx
L_form = (rho * cp / dt) * T_n * v * dx + Q * v * dx + q0 * v * ds(neumann_marker)

# Create function to hold the solution
T_sol = Function(V)

# Time-stepping loop using high-level solve
time = 0.0
for n_step in range(num_steps):
    # Update current time
    time += dt
    
    # Solve the weak form directly
    solve(a == L_form, T_sol, bc_dirichlet)
    
    # Update previous solution
    T_n.assign(T_sol)
    
    # Plot solution at certain intervals
    if n_step % 10 == 0 or n_step == num_steps - 1:
        plt.plot(mesh.coordinates(), T_sol.compute_vertex_values(mesh), label=f'Time = {time:.2f}s')

# Finalize and display the plot
plt.xlabel('Position x (m)')
plt.ylabel('Temperature T (K)')
plt.title('Time-Dependent 1D Heat Conduction')
plt.legend()
plt.grid(True)
plt.show()
