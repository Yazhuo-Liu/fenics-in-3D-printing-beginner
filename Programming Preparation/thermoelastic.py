from dolfin import *
import numpy as np
import os
import glob

# Define useful directories
crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, "ThermalMechanicalData")
vtk_dir = os.path.join(data_dir, "VTK")

# Do some cleaning before running the simulation
files = glob.glob(os.path.join(vtk_dir, f"*"))
for f in files:
    os.remove(f)
vtkfile_temp = File(os.path.join(vtk_dir, "Temperature.pvd"))
vtkfile_disp = File(os.path.join(vtk_dir, "Displacement.pvd"))
vtkfile_stress = File(os.path.join(vtk_dir, "Stress.pvd"))
vtkfile_vonmises = File(os.path.join(vtk_dir, "VonMises.pvd"))

# Define material parameters
E = 210e9          # Young's Modulus (Pa)
nu = 0.3           # Poisson's Ratio
alpha = 1.2e-5     # Thermal expansion coefficient (/°C)
Cp = 588. # Heat capacity, J/(kg K)
rho = 8440. # Density, kg/m^3
k = 15. # Thermal conductivity, W/(m K)
Tl = 1623. # Liquidus temperature, K
h = 100. # Heat convection coefficient, W/(m^2 K)
eta = 0.25 # absorption rate
SB_constant = 5.67e-8 # Stefan-Boltzmann constant, W/(m^2 K^4)
emissivity = 0.5 # Emissivity of the surface
T0 = 300. # Ambient temperature, K

# Define laser properties
vel = 0.5 # Laser velocity, m/s
rb = 0.05e-3 # Laser beam radius, m
P = 20. # Laser power, W

# Define mesh
Lx, Ly, Lz = 0.5e-3, 0.2e-3, 0.05e-3
Nx, Ny, Nz = 50, 20, 5
mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), Nx, Ny, Nz)
# ResultFile.write(mesh)

# Define time parameters
laser_on_t = 0.5*Lx / vel
simulation_time = 2*laser_on_t
dt = 10e-5
num_steps = int(simulation_time / dt)
ts = np.linspace(0, simulation_time, num_steps+1)

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

# Define function space
R = FunctionSpace(mesh, "P", 1)
T = Function(R)          # Temperature at the current time step (unknownn and solution)
Q = TestFunction(R)      # Test function of T used in variational formulation
T_old = Function(R)      # Temperature at the previous time step
V = VectorFunctionSpace(mesh, 'P', 1, dim=3)
u = TrialFunction(V)     # Displacement at the current time step (unknown)
v = TestFunction(V)      # Test function of u used in variational formulation
u_sol = Function(V)      # Displacement at the current time step (solution)
S = TensorFunctionSpace(mesh, 'P', 1)
stress = Function(S)     # Stress tensor at the current time step
Von = FunctionSpace(mesh, 'P', 1)
von_mises = Function(Von) # Von Mises stress at the current time step


######################################################################
# Thermal problem
######################################################################

# Define initial conditions
T_init = Constant(T0)
T_old.assign(T_init)

# Define the laser flux expression
class LaserFlux(UserExpression):
    def __init__(self, switch, x0, y0, vx, vy, rb, P, eta, t, **kwargs):
        super().__init__(**kwargs)
        self.switch = switch
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.rb = rb
        self.P = P
        self.eta = eta
        self.t = t
    
    def eval(self, values, x):
        laser_center_x = self.x0 + self.vx * self.t
        laser_center_y = self.y0 + self.vy * self.t
        distance = np.sqrt((x[0] - laser_center_x)**2 + (x[1] - laser_center_y)**2)
        values[0] = self.switch * 2 * self.eta * self.P / (np.pi * self.rb**2) * np.exp(-2 * distance**2 / self.rb**2)
        
    def value_shape(self):
        return ()    

# Define laser flux
q_laser = LaserFlux(switch=1, x0=Lx*0.25, y0=Ly/2, vx=vel, vy=0., rb=rb, P=P, eta=eta, t=0., degree=2)
# q_conv = h * (T0 - T_old)
# q_rad = emissivity * SB_constant * (T0**4 - T_old**4)

q_conv = h * (T0 - T)
q_rad = emissivity * SB_constant * (T0**4 - T**4)


# Define boundary condition for the bottom face (Dirichlet condition)
bc_bottom = DirichletBC(R, T0, bottom)  # Apply the boundary condition
bcsT = [bc_bottom]

# Weak form for the thermal problem
FT_1 = rho * Cp * (T - T_old) / dt * Q * dx + k * inner(grad(T), grad(Q)) * dx
FT_2 = - (q_conv + q_rad + q_laser) * Q * ds(1) - (q_conv + q_rad) * Q * ds(3)
FT = FT_1 + FT_2

# Define the variational problem
problemT = NonlinearVariationalProblem(FT, T, bcs=bcsT, J=derivative(FT, T))
solverT = NonlinearVariationalSolver(problemT)
solverT.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
solverT.parameters["newton_solver"]["relative_tolerance"] = 1e-7
solverT.parameters["newton_solver"]["maximum_iterations"] = 100
solverT.parameters["newton_solver"]["relaxation_parameter"] = 1.0


######################################################################
# Mechanical problem
######################################################################

# Lame parameters
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu)*(1 - 2*nu))

# Define the boundary conditions
bc_bottom = DirichletBC(V, Constant((0.0, 0.0, 0.0)), bottom)
bcsu = [bc_bottom]

# Define the strain tensor (symmetric gradient of displacement)
def epsilon(u):
    return sym(grad(u))

# Define the thermal strain tensor using as_tensor for correct shape
def epsilon_t(T):
    alpha_T = conditional(T < Tl, alpha, 0)
    return alpha_T * (T - T0) * as_tensor(np.eye(3))

# Define the degenration function for thermal softening and melting.
def degeneration(T):
    return conditional(T < T0, 1, conditional(T < Tl, 1 - (T - T0) / (Tl - T0), 1e-3))

# Define the stress tensor using Hooke's Law with thermal strain
def sigma(u, T):
    sig = lmbda * tr(epsilon(u) - epsilon_t(T)) * Identity(3) + 2 * mu * (epsilon(u) - epsilon_t(T))
    xi = conditional(T < Tl, 1 - 0.5 * (T - T0) / (Tl - T0), 1e-6)
    return sig * xi

# Weak form of the mechanical problem
Fu = inner(sigma(u, T), epsilon(v)) * dx
problemu = LinearVariationalProblem(lhs(Fu), rhs(Fu), u_sol, bcsu)
solveru = LinearVariationalSolver(problemu)

for t in ts:
    
    print(f"Time: {t:.5f} s")
    
    q_laser.t = t
    if t > laser_on_t:
        q_laser.switch = 0
    
    solverT.solve()
    T_old.assign(T)
    T.rename("Temperature", "Temperature")
    
    solveru.solve()
    u_sol.rename("Displacement", "Displacement")
    
    stress.assign(project(sigma(u_sol, T), S))
    stress.rename("Stress", "Stress")
    
    
    stress_dev = stress - (1./3)*tr(stress)*Identity(3)
    von_mises.assign(project(sqrt(3./2*inner(stress_dev, stress_dev)), Von))
    von_mises.rename("Von Mises Stress", "Von Mises Stress")
    
    vtkfile_temp << (T, t)
    vtkfile_disp << (u_sol, t)
    vtkfile_stress << (stress, t)
    vtkfile_vonmises << (von_mises, t)