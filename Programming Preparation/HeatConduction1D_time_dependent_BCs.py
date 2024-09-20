from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import tempfile
import shutil

# Suppress FEniCS log output by setting the log level to 'ERROR'
set_log_level(40)  # 40=ERROR

# Define a custom UserExpression for time-dependent Neumann boundary condition
class Q0(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0.0  # Initialize time parameter

    def eval(self, value, x):
        # Define the time-dependent heat flux
        value[0] = 100 + 50 * np.sin(2 * np.pi * self.t)

    def value_shape(self):
        return ()

# Instantiate the UserExpression without passing 't'
q0 = Q0(degree=1)

# Define physical parameters
alpha = 1.0                # Thermal diffusivity (m^2/s)
rho = 1.0                  # Density (kg/m^3)
cp = 1.0                   # Specific heat capacity (J/kg*K)
k = 1.0                    # Thermal conductivity (W/m*K)
Q = Constant(0.0)          # Internal heat generation (W/m^3)

# Define domain parameters
L = 1.0                    # Length of the rod (m)
nx = 50                    # Number of finite elements

# Define boundary conditions
# Dirichlet boundary condition at x=L (constant temperature)
T0 = Constant(300.0)       # Prescribed temperature at x=L (K)

# Time-stepping parameters
T_final = 2.0              # Final time (s)
num_steps = 100            # Number of time steps
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

# Set up temporary directory to store frames
temp_dir = tempfile.mkdtemp()
frames = []

# Time-stepping loop using high-level solve
time = 0.0
for n_step in range(num_steps):
    # Update current time
    time += dt
    q0.t = time  # Update the time variable in the UserExpression
    
    # Solve the weak form directly
    solve(a == L_form, T_sol, bc_dirichlet)
    
    # Update previous solution
    T_n.assign(T_sol)
    
    # Capture the plot at desired intervals
    if n_step % 5 == 0 or n_step == num_steps - 1:
        plt.figure()
        plt.plot(mesh.coordinates(), T_sol.compute_vertex_values(mesh), 'r-', linewidth=2)
        plt.xlabel('Position x (m)')
        plt.ylabel('Temperature T (K)')
        plt.title(f'Time = {time:.2f} s')
        plt.grid(True)
        
        # Save the current plot to the temporary directory
        filename = f'frame_{n_step:03d}.png'
        filepath = os.path.join(temp_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        # Append the file path to the frames list
        frames.append(filepath)

# Generate GIF from the saved frames
gif_path = 'temperature_evolution.gif'
with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
    for frame in frames:
        image = imageio.v3.imread(frame)
        writer.append_data(image)

# Remove the temporary directory and its contents
shutil.rmtree(temp_dir)

print(f'GIF saved as {gif_path}')
