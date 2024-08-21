# Installing FEniCS 2019.1.0 on macOS using Anaconda

Follow these steps to install FEniCS, a popular open-source computing platform for solving partial differential equations (PDEs), on your macOS system using Anaconda.

## Reference
- https://fenicsproject.org/download/archive/

## Prerequisites

Ensure that Anaconda is installed on your system. If not, download and install it follow [our tutorial](/Markdown/Install%20Anaconda3%20on%20MacOS).

## Step 1: Open Terminal

Open the Terminal application on your macOS. You can find Terminal in `Applications/Utilities` or by searching for it using Spotlight.

## Step 2: Create a Conda Environment

It is recommended to create a new conda environment to avoid conflicts with existing Python packages:
```bash
conda create --n fenics python=3.9
conda activate fenics2019
```

## Step 3: Install FEniCS

- Install FEniCS from the conda-forge channel:
  ```bash
  conda install -c conda-forge fenics
  ```
    This step may take long time.

- Install required packages
  ```bash
  conda install matplotlib
  conda install numpy
  ```

## Step 4: Verify Installation
Check if FEniCS is installed correctly:

```bash
python -c "import fenics; print(fenics.__version__)"
```
This command should output 2019.1.0.

## Example Usage
Here is a simple example to test your FEniCS installation:

```python
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
```

If you see "Solution saved to poisson_solution.pvd" in command, your installation is success.