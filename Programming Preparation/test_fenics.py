from fenics import *

# Define the problem
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
bc = DirichletBC(V, u_D, 'on_boundary')
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Output results
vtkfile = File('poisson_solution.pvd')
vtkfile << u
print('Solution saved to poisson_solution.pvd')