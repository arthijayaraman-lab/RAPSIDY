#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Force Bias Functions

Created on Tue Apr  2 17:27:11 2024

@author: liaov
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import multiprocessing as mp
import MDAnalysis as mda
from numba import jit, njit, prange
from itertools import product

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.rcParams['figure.dpi'] = 1200

def charge_density(x, x_p):
    W = np.zeros((5,5))
    W[0,:] = np.array([1, -8, 24, -32, 16])/384
    W[1,:] = np.array([19, -44, 24, 16, -16])/96
    W[2,:] = np.array([115, 0, -120, 0, 48])/192
    W[3,:] = np.array([19, 44, 24, -16, -16])/96
    W[4,:] = np.array([1, 8, 24, 32, 16])/384
    x_xp = np.asarray([(x-x_p)**i for i in range(5)])
    result = np.dot(W, x_xp)
    
    return result

def charge_density_derivative(x, x_p, spacing):
    dW = np.zeros((5,5))
    dW[0,:] = np.array([0, -8, 2*24, 3*-32, 4*16])/384
    dW[1,:] = np.array([0, -44, 2*24, 3*16, 4*-16])/96
    dW[2,:] = np.array([0, 0, 2*-120, 3*0, 4*48])/192
    dW[3,:] = np.array([0, 44, 2*24, 3*-16, 4*-16])/96
    dW[4,:] = np.array([0, 8, 2*24, 3*32, 4*16])/384
    x_xp = np.asarray([0]*x.shape[0])
    x_xp = np.vstack((x_xp, np.asarray([(x-x_p)**i for i in range(4)])))
    result = np.dot(dW, x_xp)
    return result/spacing 

@njit
def find_min_abs(X):
    min_arg = np.argmin(np.abs(X), axis=1)
    result = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        result[i] = X[i, min_arg[i]]
    return result

def xp(x, cell, mesh_dim):
    parent_mesh = np.linspace(0, cell, mesh_dim, endpoint=False)
    spacing = parent_mesh[1]- parent_mesh[0]
    distance_cell = np.subtract.outer(x,parent_mesh)
    distance_cell[distance_cell>(cell-spacing/2)] -= cell
    x_rescaled = (find_min_abs(distance_cell)/spacing)
    xp = np.array([0]*x_rescaled.shape[0])
    return xp, x_rescaled, spacing

def get_local_charge_3D(pt, cell, mesh_dim):
    # Need to do this for X, Y, and Z coordinates. 
    found_xp_x, found_x_rescaled_x, spacingx = xp(pt[:,0], cell[0], mesh_dim[0])
    found_xp_y, found_x_rescaled_y, spacingy = xp(pt[:,1], cell[1], mesh_dim[1])
    found_xp_z, found_x_rescaled_z, spacingz = xp(pt[:,2], cell[2], mesh_dim[2])
    
    # Get charge densities
    charge_dist_x = charge_density(found_x_rescaled_x, found_xp_x)
    charge_dist_y = charge_density(found_x_rescaled_y, found_xp_y)
    charge_dist_z = charge_density(found_x_rescaled_z, found_xp_z)
    
    # Get 3D densities
    charge_dist_xy = np.einsum('ij,kj->ikj', charge_dist_x, charge_dist_y)
    charge_dist_xyz = np.einsum('ijk, lk ->ijlk', charge_dist_xy, charge_dist_z)

    return np.transpose(charge_dist_xyz, [3, 0, 1, 2])

def get_local_charge_3D_gradient(pt, cell, mesh_dim):
    # Need to do this for X, Y, and Z coordinates. 
    found_xp_x, found_x_rescaled_x, spacingx = xp(pt[:,0], cell[0], mesh_dim[0])
    found_xp_y, found_x_rescaled_y, spacingy = xp(pt[:,1], cell[1], mesh_dim[1])
    found_xp_z, found_x_rescaled_z, spacingz = xp(pt[:,2], cell[2], mesh_dim[2])
    
    # Get charge densities
    charge_dist_x = charge_density(found_x_rescaled_x, found_xp_x)
    charge_dist_y = charge_density(found_x_rescaled_y, found_xp_y)
    charge_dist_z = charge_density(found_x_rescaled_z, found_xp_z)
    
    # Get 3D densities
    charge_dist_xy = np.einsum('ij,kj->ikj', charge_dist_x, charge_dist_y)
    charge_dist_xyz = np.einsum('ijk, lk ->ijlk', charge_dist_xy, charge_dist_z)
    
    # Get derivatives
    charge_dist_x_deriv = charge_density_derivative(found_x_rescaled_x, found_xp_x, spacingx)
    charge_dist_y_deriv = charge_density_derivative(found_x_rescaled_y, found_xp_y, spacingy)
    charge_dist_z_deriv = charge_density_derivative(found_x_rescaled_z, found_xp_z, spacingz)
    
    # Get partial derivatives
    dX = np.einsum('ij,kj->ikj', charge_dist_x_deriv, charge_dist_y)
    dX = np.einsum('ijk, lk ->ijlk', dX, charge_dist_z)
    
    dY = np.einsum('ij,kj->ikj', charge_dist_x, charge_dist_y_deriv)
    dY = np.einsum('ijk, lk ->ijlk', dY, charge_dist_z)
    
    dZ = np.einsum('ij,kj->ikj', charge_dist_x, charge_dist_y)
    dZ = np.einsum('ijk, lk ->ijlk', dZ, charge_dist_z_deriv)
    
    
    return np.transpose(charge_dist_xyz, [3, 0, 1, 2]), np.transpose(dX, [3, 0, 1, 2]), np.transpose(dY, [3, 0, 1, 2]), np.transpose(dZ, [3, 0, 1, 2])


def get_density_mesh(pt, mesh_dim, dimensions):

    density_mesh = np.zeros(np.prod(mesh_dim))
    x_mesh = np.linspace(0, dimensions[0], mesh_dim[0], endpoint=False)
    y_mesh = np.linspace(0, dimensions[1], mesh_dim[1], endpoint=False)
    z_mesh = np.linspace(0, dimensions[2], mesh_dim[2], endpoint=False)
    
    spacingx = x_mesh[1] - x_mesh[0]
    spacingy = y_mesh[1] - y_mesh[0]
    spacingz = z_mesh[1] - z_mesh[0]
    
    distance_matrix_x = np.subtract.outer(pt[:, 0], x_mesh)
    distance_matrix_y = np.subtract.outer(pt[:, 1], y_mesh)
    distance_matrix_z = np.subtract.outer(pt[:, 2], z_mesh)
    
    distance_matrix_x[distance_matrix_x>(dimensions[0]-spacingx/2)] -= dimensions[0]
    distance_matrix_y[distance_matrix_y>(dimensions[1]-spacingy/2)] -= dimensions[1]
    distance_matrix_z[distance_matrix_z>(dimensions[2]-spacingz/2)] -= dimensions[2]
    
    
    min_point_index = np.vstack([np.argmin(abs(distance_matrix_x), axis=1), np.argmin(abs(distance_matrix_y), axis=1), np.argmin(abs(distance_matrix_z), axis=1)]).T
    
    pm_index = convert_local_to_global(min_point_index, mesh_dim)
    pm_index_before_ravel = np.array([pm_index[:, :, 0].flatten(), pm_index[:, :, 1].flatten(), pm_index[:, :, 2].flatten()], dtype=int)
    local_mesh_index = np.ravel_multi_index(pm_index_before_ravel, mesh_dim)
    
    # Need this to add values inplace for all updates. += inplace operator does not work!
    np.add.at(density_mesh, local_mesh_index.flatten(), get_local_charge_3D(pt, dimensions, mesh_dim).flatten())
    # return density_mesh.reshape(mesh_dim)
    return density_mesh, local_mesh_index


@njit(parallel=True)
def add_at(A, indices, B):
    for i in prange(len(indices)):
        A[indices[i]] += B[i]
    return A

@njit(parallel=True)
def multiply_numba(a, b):
    c = np.empty_like(a)
    for i in prange(len(a)):
        c[i] = a[i] * b[i]
    return c

@njit(parallel=True)
def convert_local_to_global(indexes, mesh_dim):
    result = np.empty((np.shape(indexes)[0], 125, 3))
    pm_array = np.array([[a,b, c] for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)])
    for i in prange(np.shape(indexes)[0]):
        result[i, :, :] = (indexes[i] + pm_array)%mesh_dim
    return result
    
def get_density_derivative_mesh(pt, mesh_dim, dimensions):

    density_mesh = np.zeros(np.prod(mesh_dim))
    x_mesh = np.linspace(0, dimensions[0], mesh_dim[0], endpoint=False)
    y_mesh = np.linspace(0, dimensions[1], mesh_dim[1], endpoint=False)
    z_mesh = np.linspace(0, dimensions[2], mesh_dim[2], endpoint=False)
    
    spacingx = x_mesh[1] - x_mesh[0]
    spacingy = y_mesh[1] - y_mesh[0]
    spacingz = z_mesh[1] - z_mesh[0]
    
    distance_matrix_x = np.subtract.outer(pt[:, 0], x_mesh)
    distance_matrix_y = np.subtract.outer(pt[:, 1], y_mesh)
    distance_matrix_z = np.subtract.outer(pt[:, 2], z_mesh)
    
    distance_matrix_x[distance_matrix_x>(dimensions[0]-spacingx/2)] -= dimensions[0]
    distance_matrix_y[distance_matrix_y>(dimensions[1]-spacingy/2)] -= dimensions[1]
    distance_matrix_z[distance_matrix_z>(dimensions[2]-spacingz/2)] -= dimensions[2]
    
    
    min_point_index = np.vstack([np.argmin(abs(distance_matrix_x), axis=1), np.argmin(abs(distance_matrix_y), axis=1), np.argmin(abs(distance_matrix_z), axis=1)]).T

    pm_index = convert_local_to_global(min_point_index, mesh_dim)
    pm_index_before_ravel = np.array([pm_index[:, :, 0].flatten(), pm_index[:, :, 1].flatten(), pm_index[:, :, 2].flatten()], dtype=int)
    local_mesh_index = np.ravel_multi_index(pm_index_before_ravel, mesh_dim)
    
    phi, dX, dY, dZ = get_local_charge_3D_gradient(pt, dimensions, mesh_dim)
    
    np.add.at(density_mesh, local_mesh_index.flatten(), phi.flatten())
    
    return density_mesh, phi, dX, dY, dZ, local_mesh_index


def get_force(pt, density_parent, mesh_dim, dimensions, kb):
    #start = time.time()
    child_density_A, phiA, dXA, dYA, dZA, local_mesh_index_A = get_density_derivative_mesh(pt, mesh_dim, dimensions)
    delta_densityA = child_density_A - density_parent
    # end = time.time()
    # print(f'Density Derivative {end-start}')
    
    #start = time.time()
    w0 = pt.shape[0]/np.prod(mesh_dim)
    FXA = kb/w0*np.sum(np.multiply(delta_densityA[local_mesh_index_A].reshape(dXA.shape[0], -1), dXA.reshape(dXA.shape[0],-1)), axis=1)
    
    FYA = kb/w0*np.sum(np.multiply(delta_densityA[local_mesh_index_A].reshape(dYA.shape[0], -1), dYA.reshape(dYA.shape[0],-1)), axis=1)
    
    FZA = kb/w0*np.sum(np.multiply(delta_densityA[local_mesh_index_A].reshape(dZA.shape[0], -1), dZA.reshape(dZA.shape[0],-1)), axis=1)

    # end = time.time()
    # print(f'Force {end-start}')
    
    # return np.vstack((-FXA, -FYA, -FZA)).T, delta_densityA, local_mesh_index_A, dXA
    return np.vstack((-FXA, -FYA, -FZA)).T, delta_densityA,


def get_force_normalized(ptA, ptB, density_parentA, density_parentB ,mesh_dim, dimensions, kb):
    #start = time.time()
    child_density_A, phiA, dXA, dYA, dZA, local_mesh_index_A = get_density_derivative_mesh(ptA, mesh_dim, dimensions)
    child_density_B, phiB, dXB, dYB, dZB, local_mesh_index_B = get_density_derivative_mesh(ptB, mesh_dim, dimensions)


    total_atoms = ptA.shape[0] + ptB.shape[0]
    total_mesh_points = np.prod(mesh_dim)

    average_density = total_atoms/total_mesh_points

    delta_densityA = child_density_A/average_density- density_parentA
    delta_densityB = child_density_B/average_density- density_parentB
   


    FXA = kb*np.sum(np.multiply(delta_densityA[local_mesh_index_A].reshape(dXA.shape[0], -1), dXA.reshape(dXA.shape[0],-1)), axis=1)
    FYA = kb*np.sum(np.multiply(delta_densityA[local_mesh_index_A].reshape(dYA.shape[0], -1), dYA.reshape(dYA.shape[0],-1)), axis=1)
    FZA = kb*np.sum(np.multiply(delta_densityA[local_mesh_index_A].reshape(dZA.shape[0], -1), dZA.reshape(dZA.shape[0],-1)), axis=1)

    FXB = kb*np.sum(np.multiply(delta_densityB[local_mesh_index_B].reshape(dXB.shape[0], -1), dXB.reshape(dXB.shape[0],-1)), axis=1)
    FYB = kb*np.sum(np.multiply(delta_densityB[local_mesh_index_B].reshape(dYB.shape[0], -1), dYB.reshape(dYB.shape[0],-1)), axis=1)
    FZB = kb*np.sum(np.multiply(delta_densityB[local_mesh_index_B].reshape(dZB.shape[0], -1), dZB.reshape(dZB.shape[0],-1)), axis=1)

    return np.vstack((-FXA, -FYA, -FZA)).T, np.vstack((-FXB, -FYB, -FZB)).T, delta_densityA, delta_densityB

def get_density_mesh_normalized(ptA, ptB, mesh_dim, dimensions):
    phiA = get_density_mesh(ptA, mesh_dim, dimensions)[0]
    phiB = get_density_mesh(ptB, mesh_dim, dimensions)[0]
    
    total_atoms = phiA.sum() + phiB.sum()
    total_mesh_points = np.prod(mesh_dim)
    
    average_density = total_atoms/total_mesh_points

    return phiA/average_density, phiB/average_density
    
    
def sample_parent_density(phiA, mesh_dim, dimensions, num_atoms, mesh_dim_new):
    x_mesh = [np.linspace(0, dimensions[i], mesh_dim[i], endpoint=False) for i in range(3)]
    
    xx, yy, zz = np.meshgrid(x_mesh[0], x_mesh[1], x_mesh[2], indexing='ij')
    
    coordinates = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    # coordinates = np.array([[x_mesh[i][0], x_mesh[j][1], x_mesh[k][2]] for i in range(mesh_dim[0]) for j in range(mesh_dim[1]) for k in range(mesh_dim[2])])
    
    chosen_mesh_points = np.random.choice([i for i in range(phiA.shape[0])], size=num_atoms, replace=False)
    probability_sample = np.random.uniform(size = num_atoms)
    A_beads = np.where(probability_sample<phiA[chosen_mesh_points])[0]
    B_beads = np.where(probability_sample>phiA[chosen_mesh_points])[0]
    
    A_bead_pos = np.array(coordinates[chosen_mesh_points[A_beads]])
    B_bead_pos = np.array(coordinates[chosen_mesh_points[B_beads]])
    
    phiA = get_density_mesh(A_bead_pos, mesh_dim_new, dimensions)[0]
    phiB = get_density_mesh(B_bead_pos, mesh_dim_new, dimensions)[0]
    
    return phiA, phiB

def unravel_indices(indices, shape):
    """
    Unravel an array of flat indices into a list of tuples of indices for a 3D array.

    Parameters:
        indices (list): A list of flat index values.
        shape (tuple): The shape of the 3D array (e.g., (rows, columns, depth)).

    Returns:
        list: A list of tuples of indices corresponding to the given flat indices.
    """
    if len(shape) != 3:
        raise ValueError("Shape must be a tuple of length 3 for a 3D array.")

    size_yz = shape[1] * shape[2]
    unravelled_indices = []
    
    for i in range(indices.shape[0]):
        i = indices[i] // size_yz
        j = (indices[i] % size_yz) // shape[2]
        k = (indices[i] % size_yz) % shape[2]
        unravelled_indices.append((i, j, k))
        
    return np.array(unravelled_indices, dtype=int)

def sample_parent_density_fast(phiA, mesh_dim, dimensions, num_atoms, mesh_dim_new):
    x_mesh = [np.linspace(0, dimensions[i], mesh_dim[i], endpoint=False) for i in range(3)]
    xx, yy, zz = np.meshgrid(x_mesh, indexing='ij')
    coordinates = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    
    # coordinates = np.array([[x_mesh[0][i], x_mesh[1][j], x_mesh[2][k]] for i in range(mesh_dim[0]) for j in range(mesh_dim[1]) for k in range(mesh_dim[2])])
    
    # chosen_mesh_points = np.random.choice([i for i in range(phiA.shape[0])], size=num_atoms, replace=False)
    chosen_mesh_points = np.random.randint(low=0, high=phiA.shape[0], size=num_atoms)

    probability_sample = np.random.uniform(size = num_atoms)
    A_beads = np.where(probability_sample<phiA[chosen_mesh_points])[0]
    B_beads = np.where(probability_sample>phiA[chosen_mesh_points])[0]
    
    # Chosen coordinates
    print(chosen_mesh_points)
    chosen_coordinates = unravel_indices(chosen_mesh_points, mesh_dim)
    print(chosen_coordinates)
    coordinates = np.vstack((x_mesh[0][chosen_coordinates[:, 0]], x_mesh[1][chosen_coordinates[:, 1]], x_mesh[2][chosen_coordinates[:, 2]])).T
    
    A_bead_pos = np.array(coordinates[A_beads])
    B_bead_pos = np.array(coordinates[B_beads])
    
    phiA = get_density_mesh(A_bead_pos, mesh_dim_new, dimensions)[0]
    phiB = get_density_mesh(B_bead_pos, mesh_dim_new, dimensions)[0]
    
    return phiA, phiB

def sample_parent_density_hexagonal(phiA, coordinates, dimensions, num_atoms, mesh_dim_new):
    # x_mesh = np.array([np.linspace(0, dimensions[i], mesh_dim[i], endpoint=False) for i in range(3)]).T
    # coordinates = np.array([[x_mesh[i, 0], x_mesh[j, 1], x_mesh[k, 2]] for i in range(mesh_dim[0]) for j in range(mesh_dim[1]) for k in range(mesh_dim[2])])
    
    chosen_mesh_points = np.random.choice([i for i in range(phiA.shape[0])], size=num_atoms, replace=False)
    probability_sample = np.random.uniform(size = num_atoms)
    A_beads = np.where(probability_sample<phiA[chosen_mesh_points])[0]
    B_beads = np.where(probability_sample>phiA[chosen_mesh_points])[0]
    
    A_bead_pos = np.array(coordinates[chosen_mesh_points[A_beads]])
    B_bead_pos = np.array(coordinates[chosen_mesh_points[B_beads]])
    
    phiA = get_density_mesh(A_bead_pos, mesh_dim_new, dimensions)[0]
    phiB = get_density_mesh(B_bead_pos, mesh_dim_new, dimensions)[0]
    
    return phiA, phiB
    

def plot_3D(density, dimensions, mesh_dim):
    x_mesh = [np.linspace(0, dimensions[i], mesh_dim[i], endpoint=False) for i in range(3)]
    
    xx, yy, zz = np.meshgrid(x_mesh[0], x_mesh[1], x_mesh[2], indexing='ij')
    
    parent_mesh = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect(aspect=(1, 2, 3))

    p = ax.scatter(parent_mesh[:, 0], parent_mesh[:, 1], parent_mesh[:,2], c=density, vmax=1.0, vmin=0.0)
    cb = plt.colorbar(p, label='$\phi_A$', pad=0.11, extend='both')
    cb.set_ticks(np.linspace(0, 1, 5))
    ax.set_xlabel('x/x$_{max}$')
    ax.set_ylabel('y/y$_{max}$')
    ax.set_zlabel('z/z$_{max}$')
    plt.show()

# Example density functions
def gyroid1(x, y, z, a, fa):
    m = 2*np.pi/a # Box size
    term1 = np.cos(m * x) * np.sin(m * y)
    term2 = np.cos(m * y) * np.sin(m * z)
    term3 = np.cos(m * z) * np.sin(m * x)
    
    term4 = np.cos(2 * m * x) * np.cos(2 * m * y)
    term5 = np.cos(2 * m * y) * np.cos(2 * m * z)
    term6 = np.cos(2 * m * z) * np.cos(2 * m * x)
    
    # Combine terms to form the final expression
    result = 10 * (term1 + term2 + term3) - 0.5 * (term4 + term5 + term6)-(1-fa)/0.067
    
    return result

def gyroid2(x, y, z, a, fa):
    m = 2*np.pi/a # Box size
    term1 = np.cos(m * x) * np.sin(m * y)
    term2 = np.cos(m * y) * np.sin(m * z)
    term3 = np.cos(m * z) * np.sin(m * x)
    
    term4 = np.cos(2 * m * x) * np.cos(2 * m * y)
    term5 = np.cos(2 * m * y) * np.cos(2 * m * z)
    term6 = np.cos(2 * m * z) * np.cos(2 * m * x)
    
    # Combine terms to form the final expression
    result = -10 * (term1 + term2 + term3) - 0.5 * (term4 + term5 + term6)-(1-fa)/0.067
    
    return result

def gyroid_density(fa, grid_size):
    x = np.linspace(0, 1, int(np.ceil(grid_size)))
    y = np.linspace(0, 1, int(np.ceil(grid_size)))
    z = np.linspace(0, 1, int(np.ceil(grid_size)))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    a = 1
    g1 = gyroid1(X, Y, Z, a, fa)
    g2 = gyroid2(X, Y, Z, a, fa)
    g1[g1>0] = 1
    g1[g1<0] = 0
    g2[g2>0] = 1
    g2[g2<0] = 0
    
    phiA = g1+g2
    phiA = phiA.flatten()
    phiB = np.ones(phiA.shape)-phiA
    phiB = phiB.flatten()
    
    mesh_dim = np.array([int(np.ceil(grid_size)), int(np.ceil(grid_size)), int(np.ceil(grid_size))])
    
    return phiA, phiB, mesh_dim

def C6_density(fa, grid_size):
    r = (fa*(3**0.5)/(2*np.pi))**0.5
    centers = np.array([[0,0, 0], [1, 0, 0], [2, 0, 0], [1/2, 3**(0.5)/2, 0], [3/2, 3**(0.5)/2, 0], [0, 3**(0.5), 0], [1, 3**(0.5), 0], [2, 3**(0.5), 0]])
    
    x = np.linspace(0, 2, int(np.ceil(grid_size*2)))
    y = np.linspace(0, 3**0.5, int(np.ceil(grid_size*3**0.5)))
    z = np.linspace(0, 1, int(np.ceil(grid_size*2)))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    phiA = np.zeros(X.shape).flatten()
    
    for i in range(8):
        phiA += (np.linalg.norm(pts[:, 0:2]-centers[i, 0:2], axis=1)<r)*1
    phiB = np.ones(X.shape).flatten() - phiA
    mesh_dim = np.array(X.shape)
    
    return phiA, phiB, mesh_dim

def L_density(fa, grid_size):
    # Use 2 periods, stack in z direction
    x = np.linspace(0, 1, int(np.ceil(grid_size)))
    y = np.linspace(0, 1, int(np.ceil(grid_size)))
    z = np.linspace(0, 1, int(np.ceil(grid_size)))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    phiA = ((Z>0) & (Z<(fa/2)) | (Z>0.5) & (Z<(0.5+fa/2)))*1
    phiA = phiA.flatten()
    phiB = np.ones(X.shape).flatten() - phiA
    mesh_dim = np.array(X.shape)
    
    return phiA, phiB, mesh_dim

def bcc_density(fa, grid_size):
    r = (fa/(2*np.pi))**(1/3)
    centers = np.array([[0.5, 0.5, 0.5], [0,0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    x = np.linspace(0, 1, int(np.ceil(grid_size)))
    y = np.linspace(0, 1, int(np.ceil(grid_size)))
    z = np.linspace(0, 1, int(np.ceil(grid_size)))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    phiA = np.zeros(X.shape).flatten()
    
    for i in range(centers.shape[0]):
        phiA += (np.linalg.norm(pts-centers[i], axis=1)<r)*1
    phiB = np.ones(X.shape).flatten() - phiA
    mesh_dim = np.array(X.shape)
    
    return phiA, phiB, mesh_dim

def fcc_density(fa, grid_size):
    r = (3*fa/(16*np.pi))**(1/3)
    centers = np.array([[0,0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0], [0.5, 0.5, 1], [0, 0.5, 0.5], [1, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 1, 0.5]])
    x = np.linspace(0, 1, int(np.ceil(grid_size)))
    y = np.linspace(0, 1, int(np.ceil(grid_size)))
    z = np.linspace(0, 1, int(np.ceil(grid_size)))
    X, Y, Z = np.meshgrid(x, y, z)
    pts = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    phiA = np.zeros(X.shape).flatten()
    
    for i in range(centers.shape[0]):
        phiA += (np.linalg.norm(pts-centers[i], axis=1)<r)*1
    phiB = np.ones(X.shape).flatten() - phiA
    mesh_dim = np.array(X.shape)
    
    return phiA, phiB, mesh_dim
