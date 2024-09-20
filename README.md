# fenics-in-3D-printing-beginner
Heat transfer and thermoelasticity in 3D metal printing with fenics programming tutorial. All required materials are uploaded.

![Last Commit](https://img.shields.io/github/last-commit/Yazhuo-Liu/fenics-in-3D-printing-beginner)

## Reference
1. Gibson, Ian, et al. *Additive manufacturing technologies*. Vol. 17. Cham, Switzerland: Springer, 2021.
2. Abali, Bilen Emek. *Computational reality*. Springer Verlag, Singapor, 2016.
3. Langtangen, Hans Petter, and Anders Logg. *Solving PDEs in python: the FEniCS tutorial I*. Springer Nature, 2017.

## Table of contents:

### 1. Basics of Additive Manufacturing
- Read _Additive Manufacturing Technologies_ book 3rd edition Chapter 1, 2, 5, and 21.
- [Quick starter](/Markdown/Quick%20Starter%20for%20metal%203D%20printing.md)

### 2. Programming Preparation
- [Install Anaconda3 and OpenMPI on MacOS](/Markdown/Install%20Anaconda3%20on%20MacOS.md) 
- [Install legacy FEniCS version 2019.1.0](/Markdown/Install%20FEniCS%202019.1.0%20using%20conda.md)
- (Optional) [Install ***VS Code*** as IDE](/Markdown/Installing%20Visual%20Studio%20Code.md)
- (Optional) Install Termius for SSH connection [(windows)](/Markdown/Install%20Termius%20and%20enable%20remoteX.md) (MacOS)
- Python basics
  - [Basic Syntax](/Markdown/Introduction%20to%20Python%20Syntax.md)
  - [Practice Project: Main color extraction by K-means](/Markdown/Project%20K-means.md)
  - Practice Project: 
### 3. Mathematical-Physical foundation
- Introduction to linear algebra / tensor Algebra
- Introduction to partial differential equations
- Introduction to weak form PDEs
- [Introduction to Heat transfer](/Markdown/1D%20heat%20conduction.pdf)
  - [Thermal conduction](/Markdown/3D%20heat%20conduction.pdf)
  - Thermal convection
  - Thermal radiation
- Introduction to elasticity
  - Stress and Strain
  - Hooke's Law
  - Eigenstrain

### 4. Introduction to Finite Element Method
- Why FEM?
- Revisit weak form PDE
- General Strategy for Finite Element Analysis
- FEniCS Programming
  - Introduction to FEniCS
  - Functions in FEniCS
  - Project: [Heat Transfer](/Markdown/1D%20heat%20conduction.pdf)
  - Project: Linear Elasticity
  - Project: Laser Scanning on a Mild Steel
  
### 5. Introduction to Machine Learning
