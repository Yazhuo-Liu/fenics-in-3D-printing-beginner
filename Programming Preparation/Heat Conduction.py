from dolfin import *
import numpy as np

# Material properties
rho = 7800.0       # Density (kg/m^3)
c = 500.0          # Specific heat capacity (J/(kg*K))
k = 45.0           # Thermal conductivity (W/(m*K))
alpha = k / (rho * c)  # Thermal diffusivity

# Create a mesh of the domain (a cube from (0,0,0) to (10,10,10))
mesh = BoxMesh(Point(0, 0, 0), Point(10, 10, 10), 20, 20, 20)

# Define the function space (continuous Lagrange elements of degree 1)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition for the bottom face (Dirichlet condition)
def bottom(x, on_boundary):
    return near(x[2], 0.0) and on_boundary

T_bottom = Constant(300.0)             # Fixed temperature at the bottom
bc_bottom = DirichletBC(V, T_bottom, bottom)  # Apply the boundary condition

# Parameters for the moving Gaussian heat source
q0 = 1e5           # Peak heat flux (W/m^2)
v_x = 0.5          # Velocity in x-direction (m/s)
v_y = 0.0          # Velocity in y-direction (m/s)
sigma = 1.0        # Standard deviation (controls the spread)

# Initial position of the heat source
x0_0 = 0.0         # Initial x-position
y0_0 = 5.0         # Initial y-position

# Expressions for the moving source position
x0 = Expression('x0_0 + v_x*t', degree=1, x0_0=x0_0, v_x=v_x, t=0.0)
y0 = Expression('y0_0 + v_y*t', degree=1, y0_0=y0_0, v_y=v_y, t=0.0)

# Heat flux expression as a user-defined function
class HeatFluxExpression(UserExpression):
    def __init__(self, x0, y0, sigma, t, **kwargs):
        super().__init__(**kwargs)
        self.x0 = x0        # x-position expression of the heat source
        self.y0 = y0        # y-position expression of the heat source
        self.sigma = sigma  # Standard deviation of the Gaussian
        self.t = t          # Current time

    def eval(self, values, x):
        x0_val = self.x0(self.t)  # Evaluate x-position at current time
        y0_val = self.y0(self.t)  # Evaluate y-position at current time
        # Compute the exponent of the Gaussian function
        exponent = -((x[0] - x0_val)**2 + (x[1] - y0_val)**2) / (2 * self.sigma**2)
        values[0] = q0 * np.exp(exponent)  # Compute the heat flux value

    def value_shape(self):
        return ()

q_flux = HeatFluxExpression(x0, y0, sigma, t=0.0, degree=2)  # Instantiate the heat flux expression

# Initial temperature distribution
T_initial = Constant(300.0)     # Initial temperature of the domain
T_n = interpolate(T_initial, V) # Project initial temperature onto the function space

# Define the unknown and test functions
T = Function(V)          # Temperature at the current time step (unknown)
v = TestFunction(V)      # Test function used in variational formulation
dt = 0.1                 # Time step size

# Define boundary measure for the top surface where the heat flux is applied
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
top = AutoSubDomain(lambda x, on_boundary: near(x[2], 10.0) and on_boundary)
top.mark(boundary_markers, 1)  # Mark the top boundary with marker '1'
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)  # Redefine the measure 'ds' with subdomains

# Weak form of the heat conduction equation
F = (T - T_n) / dt * v * dx + k * dot(grad(T), grad(v)) * dx - q_flux * v * ds(1)
J = derivative(F, T)

# Time-stepping parameters
t = 0.0                           # Initial time
T_end = 10.0                      # Final time
num_steps = int(T_end / dt)       # Number of time steps

# Create VTK file for saving the solution (for visualization)
vtkfile = File('heat_conduction/temperature.pvd')

# Time-stepping loop
for n in range(num_steps):
    # Update current time
    t += dt
    print(f'Time step {n+1}/{num_steps}: t = {t:.2f}')

    # Update moving heat source position and heat flux expression
    x0.t = t           # Update time in x-position expression
    y0.t = t           # Update time in y-position expression
    q_flux.t = t       # Update time in heat flux expression

    # Solve the variational problem
    solve(F == 0, T, bc_bottom, J=J)

    # Save the solution to file
    vtkfile << (T, t)

    # Update previous solution for next time step
    T_n.assign(T)