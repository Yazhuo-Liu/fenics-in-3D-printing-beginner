# Installing FEniCS 2019.1.0 on macOS using Anaconda

Follow these steps to install FEniCS, a popular open-source computing platform for solving partial differential equations (PDEs), on your macOS system using Anaconda.

## What is FEniCS?

FEniCS is an open-source computing platform for solving partial differential equations (PDEs) using finite element methods. Python serves as the primary programming language for FEniCS, making it accessible for scientific computing.

## Reference
- https://fenicsproject.org/download/archive/

## Prerequisites

Ensure that Anaconda is installed on your system. If not, download and install it follow [our tutorial](/Markdown/Install%20Anaconda3%20on%20MacOS).

## Step 1: Open Terminal

Open the Terminal application on your macOS. You can find Terminal in `Applications/Utilities` or by searching for it using Spotlight.

## Step 2: Create a Conda Environment

It is recommended to create a new conda environment to avoid conflicts with existing Python packages:
```bash
conda create -n fenics python=3.9
conda activate fenics
```

## Step 3: Install FEniCS

- Install FEniCS from the conda-forge channel:
  ```bash
  conda install -c conda-forge fenics
  ```
    This step may take long time for solving the environment. (~20 min)

- Install required packages
  ```bash
  conda install matplotlib
  conda install numpy
  ```

## Step 4: Verify Installation
Check if FEniCS is installed correctly:

```bash
python -c "import dolfin as df; print(df.__version__)"
```
This command should output 2019.1.0.

If yes, you finished the installation.
