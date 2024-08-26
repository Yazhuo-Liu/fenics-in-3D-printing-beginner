
# Introduction to Python Syntax, NumPy, and FEniCS

## 1. Python Syntax

### 1.1 Basic Syntax

Python is known for its clean and readable syntax. Here are some fundamental aspects:

- **Variables**: Variables are dynamically typed, meaning you don’t need to declare their type.
    ```python
    x = 10
    name = "Alice"
    ```

- **Indentation**: Python uses indentation to define code blocks.
    ```python
    if x > 5:
        print("x is greater than 5")
    ```

- **Comments**: Use `#` for single-line comments and triple quotes for multi-line comments.
    ```python
    # This is a comment
    '''
    This is a multi-line comment
    '''
    ```

- **Functions**: Define functions using the `def` keyword.
    ```python
    def add(a, b):
        return a + b
    ```

- **Loops**: Python supports `for` and `while` loops.
    ```python
    for i in range(5):
        print(i)
    
    while x > 0:
        x -= 1
    ```

- **Data Structures**: Common structures include lists, dictionaries, tuples, and sets.
    ```python
    my_list = [1, 2, 3]
    my_dict = {'name': 'Alice', 'age': 25}
    my_tuple = (1, 2, 3)
    my_set = {1, 2, 3}
    ```

### 1.2 `continue` and `break` Keywords

#### `continue` Keyword

The `continue` keyword is used in loops (both `for` and `while`) to skip the current iteration and proceed with the next one. When Python encounters a `continue` statement, it immediately jumps to the next iteration of the loop, **bypassing** the remaining code within the loop for the current iteration.

**Example: Using `continue` in a `for` loop**
```python
for i in range(5):
    if i == 2:
        continue  # Skip the rest of the loop when i is 2
    print(i)
```
**Output:**
```
0
1
3
4
```

#### `break` Keyword

The `break` keyword is used to exit a loop prematurely. When Python encounters a `break` statement inside a loop, it immediately **terminates** the loop, and the program continues with the code following the loop.

**Example: Using `break` in a `while` loop**
```python
i = 0
while i < 5:
    if i == 3:
        break  # Exit the loop when i is 3
    print(i)
    i += 1
```
**Output:**
```
0
1
2
```

### Summary

- **`continue`**: Skips the current iteration of the loop and moves to the next iteration.
- **`break`**: Exits the loop entirely, skipping all subsequent iterations.


## 2. Introduction to NumPy

### 2.1 What is NumPy?

NumPy is a powerful Python library used for numerical computing. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

### 2.2 Installing NumPy

You can install NumPy using conda:
```bash
conda install numpy
```

### 2.3 Creating Arrays

NumPy arrays can be created from lists or generated using functions:
```python
import numpy as np

# Creating arrays from lists
a = np.array([1, 2, 3])

# Creating arrays with specific functions
b = np.zeros((2, 2))  # 2x2 array of zeros
c = np.ones((3, 3))  # 3x3 array of ones
d = np.arange(0, 10, 2)  # Array from 0 to 10 with step 2
e = np.linspace(0, 1, 5)  # Array of 5 equally spaced points from 0 to 1
```

### 2.4 Basic Operations

NumPy supports element-wise operations on arrays:
```python
f = a + 2  # Add 2 to each element of array a
g = b * 3  # Multiply each element of array b by 3
```

### 2.5 Array Indexing and Slicing

You can access elements using indices and slices:
```python
h = a[0]  # First element of array a
i = c[:, 1]  # Second column of array c
j = d[1:4]  # Elements from index 1 to 3 in array d
```

### 2.6 Advanced Features

- **Matrix Multiplication**:
    ```python
    k = np.dot(c, c.T)  # Matrix multiplication of c and its transpose
    ```

- **Random Numbers**:
    ```python
    l = np.random.rand(3, 3)  # 3x3 array of random numbers between 0 and 1
    ```

- **Einstein Summation**:
  - You can use `einsum` to perform matrix multiplication. For example, if you have two matrices `A` and `B`:

    ```python
    import numpy as np

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    # Regular matrix multiplication
    result = np.einsum('ij,jk->ik', A, B)
    print(result)
    ```

    **Output:**
    ```
    [[19 22]
    [43 50]]
    ```

    Here, `'ij,jk->ik'` indicates that you are summing over the index `j`, multiplying corresponding elements of `A` and `B`, and storing the result in a new matrix with subscripts `ik`.
  - You can compute the dot product of two vectors:

    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    # Dot product
    result = np.einsum('i,i->', a, b)
    print(result)
    ```

    **Output:**
    ```
    32
    ```

    In this case, `'i,i->'` tells NumPy to multiply corresponding elements of `a` and `b` and sum over the index `i`, resulting in a scalar.
  - To compute the outer product of two vectors:

    ```python
    # Outer product
    result = np.einsum('i,j->ij', a, b)
    print(result)
    ```

    **Output:**
    ```
    [[ 4  5  6]
    [ 8 10 12]
    [12 15 18]]
    ```

    Here, `'i,j->ij'` means that you're not summing over any indices, but rather creating a matrix where each element is the product of elements from `a` and `b`.
  - You can use `einsum` to sum the diagonal elements of a matrix (trace):

    ```python
    # Sum of diagonal elements (trace)
    result = np.einsum('ii->', A)
    print(result)
    ```

    **Output:**
    ```
    5
    ```

    Here, `'ii->'` selects the diagonal elements (where the row and column indices are the same) and sums them.

## 3. Introduction to FEniCS

### 3.1 What is FEniCS?

FEniCS is an open-source computing platform for solving partial differential equations (PDEs) using finite element methods. Python serves as the primary programming language for FEniCS, making it accessible for scientific computing.

### 3.2 Installing FEniCS

Please refer to [the previous section](/Markdown/Install%20FEniCS%202019.1.0%20using%20conda.md).

### 3.3 Basic Workflow in FEniCS

#### 3.3.1 Importing FEniCS Modules

Start by importing the FEniCS library:
```python
from fenics import *
```

#### 3.3.2 Defining a Mesh

Define the computational domain using a mesh:
```python
mesh = UnitSquareMesh(32, 32)  # Creates a 32x32 mesh over a unit square
```

#### 3.3.3 Function Spaces

Define function spaces where the solution and test functions will reside:
```python
V = FunctionSpace(mesh, 'P', 1)  # P1 elements, linear Lagrange basis functions
```

#### 3.3.4 Boundary Conditions

Set boundary conditions using Python functions:
```python
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)
```

#### 3.3.5 Variational Problem

Formulate the variational problem:
```python
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx
```

#### 3.3.6 Solving the Problem

Solve the PDE:
```python
u_sol = Function(V)
solve(a == L, u_sol, bc)
```

#### 3.3.7 Post-Processing

Visualize the results:
```python
plot(u_sol)
import matplotlib.pyplot as plt
plt.show()
```

### 3.4 Example: Solving Poisson's Equation

Here’s a complete example solving Poisson’s equation:
```python
from fenics import *

# Define mesh and function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define source term and variational problem
f = Constant(-6.0)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u_sol = Function(V)
solve(a == L, u_sol, bc)

# Plot solution and mesh
plot(u_sol)
plot(mesh)
import matplotlib.pyplot as plt
plt.show()
```

### 3.5 Advanced Topics

- **Nonlinear Problems**: Use `NonlinearVariationalProblem` and `NonlinearVariationalSolver` for nonlinear PDEs.
- **Time-Dependent Problems**: Implement time-stepping for transient problems.
- **Parallel Computing**: FEniCS supports parallel computations using MPI.
