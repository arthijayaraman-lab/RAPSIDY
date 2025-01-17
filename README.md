# RAPSIDY - Rapid Analysis Polymer Structure and Inverse Design strategY
![toc_v2](https://github.com/user-attachments/assets/8e40d115-90a7-4bb1-8f47-dd59dff0905a)

# Installation
1. Compile the latest version of LAMMPS (https://www.lammps.org/). We recommend using Cmake to compile. You will need to compile using the custom capped Lennard-Jones potential, which we have included in the repository. 
2. Link your compiled version of LAMMPS with your Python environment. (https://docs.lammps.org/Python_install.html)
3. Include the path to the helper functions (RAPSIDY.py) in your PATH.
4. RAPSIDY is now ready to run.

# Usage
The helper functions provided in RAPSIDY.py contain all scripts needed to begin using RAPSIDY. 
1. Determine a target morphology. The initial densities are provided for the following canonical morphologies: lamellar, double gyroid, hexagonally packed cylinders, body-centered cubic spheres, face-centered cubic spheres
2. Prepare an initial LAMMPS data file with your target chain design at a specified density, box size, and bonded potentials.
3. Generate an initial random melt by allowing the system to evolve under NVT without a non-bonded potential. This will generate the random initial melt needed prior to biasing. 
4. The biasing step is performed using the function bias_step(lammps_file, N, kb), where lammps_file is the path to your initial LAMMPS file, where N is the number of timesteps for biasing, and kb is the strength of the biasing field.
5. You can edit the bias_step function in RAPSIDY.py to change the non-bonded interactions used during biasing. By default, we use the capped Lennard-Jones potential. 
6. You can follow the trajectory of the biasing via the bias_dump.lammpstrj file in your favorite software.
7. The final biased structure is written to final.data and can be used for relaxation. For the sake of flexibility of different computing architectures, we recommend users developing their own relaxation protocol.
8. After relaxing, compare the volatility-of-ratio (Vr) using the Vr_traj function.

You can follow along using the example in the EXAMPLE folder. 

If you use RAPSIDY in your research, please cite the following paper:
Liao, V., Myers, T., & Jayaraman, A. (2024). A computational method for rapid analysis polymer structure and inverse design strategy (RAPSIDY). Soft Matter, 20(41), 8246-8259.

Changelog:
1. 01/17/2025: Bug fix in initializing disordered melt prior to biasing that caused numerical errors. 
