from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Material properties
E = 210e9          # Young's Modulus (Pa)
nu = 0.3           # Poisson's Ratio
alpha = 1.2e-5     # Thermal expansion coefficient (/°C)

# Lamé parameters
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu)*(1 - 2*nu))

# Geometry parameters
length = 1.0       # Length of the block (m)
height = 0.1       # Height of the block (m)
depth = 0.1        # Depth of the block (m)

# Mesh
mesh = BoxMesh(Point(0, 0, 0), Point(length, height, depth), 20, 5, 5)

# Function space
V = VectorFunctionSpace(mesh, 'P', 1, dim=3)

# Boundary conditions
def left_boundary(x, on_boundary):
    return near(x[0], 0.0) and on_boundary

def bottom_boundary(x, on_boundary):
    return near(x[1], 0.0) and on_boundary

# Apply zero displacement on the left boundary and fix z-displacement on the bottom
bc_left = DirichletBC(V, Constant((0.0, 0.0, 0.0)), left_boundary)
bc_bottom = DirichletBC(V.sub(2), Constant(0.0), bottom_boundary)  # Fix z-displacement on bottom

bcs = [bc_left, bc_bottom]

# Temperature distribution
T0 = 20.0          # Reference temperature (°C)
T1 = 100.0         # Maximum temperature (°C)

temperature = Expression('T0 + (T1 - T0) * x[0] / L',
                         T0=T0, T1=T1, L=length, degree=1)

# Define the strain tensor (symmetric gradient of displacement)
def epsilon(u):
    return sym(grad(u))

# Define the thermal strain tensor using as_tensor for correct shape
def epsilon_t():
    return alpha * (temperature - T0) * as_tensor(((1, 0, 0),
                                                   (0, 1, 0),
                                                   (0, 0, 1)))

# Define the stress tensor using Hooke's Law with thermal strain
def sigma(u):
    return lmbda * tr(epsilon(u) - epsilon_t()) * Identity(3) + 2 * mu * (epsilon(u) - epsilon_t())

# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

# weak form
F = inner(sigma(u), epsilon(v)) * dx

# Assemble and solve the system
u_sol = Function(V)

problem = LinearVariationalProblem(lhs(F), rhs(F), u_sol, bcs)
solver = LinearVariationalSolver(problem)
solver.solve()


# Compute stress tensor at each point
S = sigma(u_sol)

# Compute von Mises stress for visualization
V_von = FunctionSpace(mesh, 'P', 1)  # Scalar function space
von_Mises = Function(V_von)          # Function to store von Mises stress

# Expression for von Mises stress in 3D
von_Mises_expr = sqrt(0.5 * ((S[0,0] - S[1,1])**2 +
                             (S[1,1] - S[2,2])**2 +
                             (S[2,2] - S[0,0])**2 +
                             6*(S[0,1]**2 + S[1,2]**2 + S[0,2]**2)))

# Project von Mises stress to function space for plotting
von_Mises = project(von_Mises_expr, V_von)

# Plot results
stress_array = von_Mises.compute_vertex_values(mesh)
coordinates = mesh.coordinates()
x = coordinates[:,0]
y = coordinates[:,1]
z = coordinates[:,2]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(x, y, z, c=stress_array, cmap='viridis', marker='o')
fig.colorbar(p, ax=ax, label='Von Mises Stress (Pa)')
ax.set_title('Von Mises Stress Distribution in 3D Block')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.show()
