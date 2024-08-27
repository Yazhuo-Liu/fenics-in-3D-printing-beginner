# Installing Anaconda3 and OpenMPI on macOS

This guide provides step-by-step instructions on how to install Anaconda3 on a macOS system. Anaconda is a popular distribution of Python programming languages for scientific computing.

## Reference
- https://docs.continuum.io/anaconda/install/mac-os/
- https://docs.continuum.io/anaconda/install/verify-install/

## Step 1: Download the Anaconda Installer

1. Go to the Anaconda distribution page: [Anaconda Installers](https://www.anaconda.com/download/success)
2. Select the macOS version.
3. Make sure you select the right version for your CPU (Apple silicon or Intel chips)
4. Download the graphical installer.

## Step 2: Run the Installer

1. Once the download is complete, locate the downloaded file in your `Downloads` folder. The file should be named something like `Anaconda3-2024.06-1-MacOSX-arm64.pkg`.
2. Double-click the `.pkg` file to launch the installer.
3. Follow the on-screen instructions:
   - Click "Continue".
   - Review the software license agreement and click "Continue", then click "Agree" to accept.
   - Select an install location (it's recommended to use the default location).
   - Click "Install" to start the installation process.
   - Enter your administrator password if prompted and click "Install Software".
   - A successful installation displays the following screen:
    ![Image](../imgs/2024-08-21-19-22-46.png)

## Step 3: Verify Installation

To verify that Anaconda was installed correctly, you can use the terminal:

1. Open the Terminal application (you can find it in the `Applications/Utilities` folder or search for it using Spotlight).
2. You should see (base) in the command line prompt. This tells you that you’re in your base conda environment.
3. Type `conda list`. This command lists all the packages installed in the Anaconda environment. If Anaconda is installed correctly, you should see a list of installed packages.

## Step 4: Deactivate the base environment by default

1. In Terminal, use the following line to prevent Anaconda from activating the base environment on startup

    ```bash
    conda config --set auto_activate_base false
    ```
2. restart the Terminal, (base) in the command line prompt should gone.

## Step 5: Installing OpenMPI on macOS

This guide will walk you through the steps to install MPI (Message Passing Interface) on a Mac using Open MPI, along with the `mpi4py` Python bindings.

1. **Install Homebrew (if not already installed)**

    Homebrew is a package manager for macOS that simplifies the installation of software.

    Open a terminal and run the following command to install Homebrew:

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

    To verify that Homebrew is installed, run:

    ```bash
    brew --version
    ```

2. **Install Open MPI**

    Open MPI is a popular implementation of the MPI standard. You can install it using Homebrew.

    Run the following command in your terminal:

    ```bash
    brew install open-mpi
    ```

3. **Verify the Installation**

    After the installation completes, verify that Open MPI and `mpirun` are installed correctly:

    ```bash
    mpirun --version
    ```

    You should see the version information for Open MPI, confirming that it is installed.

4. **Install `mpi4py` in Your Conda Environment**

    `mpi4py` provides Python bindings for MPI, allowing you to write MPI programs in Python.

    First, activate your Conda environment (or create a new one):

    ```bash
    conda create --name mpi_env python=3.9
    conda activate mpi_env
    ```

    Then, install `mpi4py`:

    ```bash
    conda install -c conda-forge mpi4py
    ```

5. **Write and Run a Simple MPI Program**

    Create a simple Python script that uses `mpi4py` to run an MPI program.

    ```python
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print(f"Hello from process {rank}")
    ```

    Save this file as `mpi_hello.py`.

6. **Run the MPI Program Using `mpirun`**

    To run the MPI program on multiple processes, use the `mpirun` command:

    ```bash
    mpirun -n 4 python mpi_hello.py
    ```

    This command will run the program on 4 processes. Each process will print a message showing its rank.

