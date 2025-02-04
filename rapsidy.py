
import numpy as np
import sys
import time
from rapsidy_helper import *
import rapsidy_helper as bias_force
import lammps
import os
import MDAnalysis as mda
from MDAnalysis import transformations



# Perform check to see if LAMMPS equilibration is running
if not os.path.isfile('initial.data'):
    
    # Set your initial mesh size
    grid_size = 20
    
    fa = 0.4
    taua = 0.5
    N = 50
    
    # Create initial data file
    initial_structure_density_3D(fa, taua, N, grid_size)
    
    # Create initial LAMMPS file
    get_bias_input(120, 60)
    
    # Run initial 
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
    # lmp = lammps.lammps()
    lmp.file("get_bias_input.in")
    
        
    # Run biased MD

    phiA, phiB, mesh_dim = L_density(fa, grid_size)
    
    
    # Get positions and forces
    x = lmp.numpy.extract_atom('x')
    f = lmp.numpy.extract_atom('f')
    ids = lmp.numpy.extract_atom('id')
    atom_type = lmp.numpy.extract_atom('type')
    
    # Get atom type ID
    indexA_parent = ids[np.where(atom_type==1)[0]].copy()-1
    indexB_parent = ids[np.where(atom_type==2)[0]].copy()-1
    
    # Get atom type positions
    min_dim = np.array([lmp.extract_global('boxxlo'),lmp.extract_global('boxylo'), lmp.extract_global('boxzlo')])
    max_dim = np.array([lmp.extract_global('boxxhi'),lmp.extract_global('boxyhi'), lmp.extract_global('boxzhi')])
    
    x_copy = x.copy()
    posA_parent = (x[indexA_parent].copy()-min_dim)%(max_dim-min_dim)
    posB_parent = (x[indexB_parent].copy()-min_dim)%(max_dim-min_dim)
    
    dimensions = max_dim-min_dim
    

    lmp.command('pair_style lj/cap 2.0 10')
    lmp.command('pair_coeff      1 1 1.00 1.00')
    lmp.command('pair_coeff       1 2 $(1-(60/50)/6) 1.00')
    lmp.command('pair_coeff      2 2 1.00 1.00')
    
s
    # Get new forces
    print('Start')
    start = time.time()
    for i in range(1000):
        # Get positions and forces
        x = lmp.numpy.extract_atom('x')
        f = lmp.numpy.extract_atom('f')
        ids = lmp.numpy.extract_atom('id')
        atom_type = lmp.numpy.extract_atom('type')
    
        # Get atom type ID
        indexA = ids[np.where(atom_type==1)[0]].copy()-1
        indexB = ids[np.where(atom_type==2)[0]].copy()-1
    
        # Get atom type positions
        min_dim = np.array([lmp.extract_global('boxxlo'),lmp.extract_global('boxylo'), lmp.extract_global('boxzlo')])
        max_dim = np.array([lmp.extract_global('boxxhi'), lmp.extract_global('boxyhi'), lmp.extract_global('boxzhi')])
        pos_child = x.copy()
    
        posA =  (pos_child[indexA]-min_dim)%(max_dim-min_dim)
        posB =  (pos_child[indexB]-min_dim)%(max_dim-min_dim)
        
        
        forceA, forceB, deltaA, deltaB = bias_force.get_force_normalized(posA, posB, phiA, phiB, mesh_dim, dimensions, 100)
        f[indexA] = f[indexA]+forceA
        f[indexB] = f[indexB]+forceB
        
        mean_delta = (np.sum(deltaA**2)+np.sum(deltaB**2))
        print(mean_delta)
        
        lmp.command('run 1 pre no post yes')
    
    end = time.time()
    print(end-start)
    lmp.command('pair_style lj/cut 2.0')
    lmp.command('pair_coeff      1 1 1.00 1.00')
    lmp.command('pair_coeff       1 2 $(1-(120/50)/6) 1.00')
    lmp.command('pair_coeff      2 2 1.00 1.00')
    lmp.command('minimize 1.0e-4 1.0e-6 100 1000')
    
    final_phiA, final_phiB = bias_force.get_density_mesh_normalized(posA, posB, mesh_dim, dimensions)
    bias_force.plot_3D(final_phiA, dimensions, mesh_dim)
    bias_force.plot_3D(final_phiB, dimensions, mesh_dim)
    lmp.command('write_data initial.data nocoeff')

