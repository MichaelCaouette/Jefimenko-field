# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:02:47 2022

@author: mcaoue2
"""

from jefimenko_acsolver import *

from rotating_field import RotatingVector
from model_array_loop import ArrayLoop
from viewer_wire_field_3D import Show3DVectorField

import affine_transfo as afifi

import model_array_loop
model_array_loop._debug_enabled = True

import nibabel as nib # To create a nifty file with the 3D volume
from imageviewer.images import  GUIimage # To view the 3D volume with personal script

import numpy as np

# =============================================================================
# Get the reference Image to overal the loop on
# Especially to get the right FOV
# =============================================================================


def save_for_ITKsnap(volume, affine,
                     savename):
    """
    Explaination soon. 
    It is to save the volume in orther to meet the ITK-snap
    """
    # ITK-SNAP reverse the x and y from the numpy convention. 
    # So I need to swap them
    my3Dstuff = volume.transpose((1, 0, 2))
    
    # I should check the affine
    new_image = nib.Nifti1Image(my3Dstuff, 
                                affine=affine)
    nib.save(new_image, savename)

### Find the center of the MRI image

# ref_img = nib.load('affine_test_ori.nii')
ref_img = nib.load('NMT_v2.0_sym_fh_ORIGINAL.nii.gz')
# Get the translation to the center of FOV
img_data = ref_img .get_fdata()
vox_center = (np.array(img_data.shape) - 1) / 2.
ref_center = afifi.apply_affine(ref_img.affine, 
                          vox_center[0], # x and y are inverted in position from the shape. 
                          vox_center[1], 
                          vox_center[2])
# Get the position of the bounds of the image. 
print('WE ASSUM NO ROTATION !')
ref_Nx, ref_Ny, ref_Nz = np.array(img_data.shape)
ref_xmin, ref_ymin, ref_zmin = afifi.apply_affine(ref_img.affine, 
                                            0,  0, 0)
ref_xmax, ref_ymax, ref_zmax = afifi.apply_affine(ref_img.affine, 
                                            ref_Nx-1, ref_Ny-1, ref_Nz-1)
ref_dx = ref_xmax - ref_xmin
ref_dy = ref_ymax - ref_ymin
ref_dz = ref_zmax - ref_zmin
print('ref_dx = %.2f mm'%ref_dx)
print('ref_dy = %.2f mm'%ref_dy)
print('ref_dz = %.2f mm'%ref_dz)

# =============================================================================
# # Define the single loop, and the system
# =============================================================================
radius_loop = 0.024 # m, Radius of the loop

# Let's list a couple of axis on which we will check the B1 rotating component
# Axis of the bore hole, or the B0 field (for the rotating calculation)
list_axis_B0 = [[0, 0, 1  ],
                [0, 1, 0.5],
                [0, 1, 1  ],
                [0, 0.5, 1],
                [1, 0, 0  ]] 

self = ArrayLoop(R=radius_loop, 
                 R_bending=-1, # No bending
                 N_loop_ver=1, N_loop_hor=1, # Just one loop
                  pos_init = [0, 0, 0], # Centered
                 dist_btw_loop=1.5*1, # Doesn't matter, because we have just one loop.
                 n_vec = [0, 1, 0],  # Point in the y direction
                 I_loop=2.3, # The current should not matter. 
                 N_dscrt_pts_loop=105, 
                 f = 297.2*1e6)

# Check the antena in 3D space
vv = Show3DVectorField()
# Show all the elements of the antennas
vv.add_list_wire(self.list_wire, list_color=np.array(self.list_delay)*self.w/(np.pi*2), 
                      str_cmap='brg', label='My loop',
                      color_title='Delay (w/(2 pi))',
                      show_corner=False)
    
# =============================================================================
# # Define the spatial points to probe
# =============================================================================

# A volume
# Nx, Ny, Nz = 40, 40, 30 # Same ratio as reference image
# Nx, Ny, Nz = 40, 39, 28# Same ratio as reference image
Nx, Ny, Nz = 18, 19, 15# Same ratio as reference image

height_probe_min = 0.5*radius_loop # Minimum distance to probe above the loop
# ymin, ymax = -0.5*ref_dy*1e-3 , 0.5*ref_dy*1e-3
# zmin, zmax = -0.5*ref_dz*1e-3 , 0.5*ref_dz*1e-3
# xmin, xmax = height_probe_min, ref_dx*1e-3 + height_probe_min
xmin, xmax = -0.5*ref_dx*1e-3 , 0.5*ref_dx*1e-3
zmin, zmax = -0.5*ref_dz*1e-3 , 0.5*ref_dz*1e-3
ymin, ymax = height_probe_min, ref_dy*1e-3 + height_probe_min


# Now everything should be done with the following variable.
x = np.linspace(xmin, xmax, Nx) 
y = np.linspace(ymin, ymax, Ny) 
z = np.linspace(zmin, zmax, Nz) 

x_probe, y_probe, z_probe = np.meshgrid(x, y, z)


# =============================================================================
# # Compute the field (only spatial component)
# =============================================================================
print('Probing the field into space...')
(list_vcos, 
 list_vsin) = self.get_field_spatial(x_probe, 
                                     y_probe,
                                     z_probe,
                                     print_progress=True) 
print('Done')

# =============================================================================
# # Compute the rotating component with respect to a couple of orientation
# =============================================================================


# define the Affine
# Get the original affine (in the current referential)
# IN mm ! nibabel works with millimeters
A_ori = afifi.get_affine(min(x)*1e3, max(x)*1e3, len(x),
                         min(y)*1e3, max(y)*1e3, len(y),
                         min(z)*1e3, max(z)*1e3, len(z))

# Get my center
vc = (np.array(x_probe.shape) - 1) / 2.
# ITK-Snap OR the numpy convention as x and y axis reversed in the shape
my_center = afifi.apply_affine(A_ori, vc[1], vc[0], vc[2])

vtrans = ref_center  - my_center

# Now we know the center of the image. So we will translate up to there !
A_trans = afifi.add_translation(A_ori, vtrans)


my_affine = A_trans

# Check where is our center with respect to the reference
my_xmin, my_ymin, my_zmin = afifi.apply_affine(my_affine, 0, 0, 0)
my_xmax, my_ymax, my_zmax = afifi.apply_affine(my_affine, Nx-1, Ny-1, Nz-1)
print()
print('Comparison of the FOV of the ref and field')
print('my_center = ', my_center)
print('spa_center = ', ref_center)
print()
print('ref_xmin, ref_xmax = ', ref_xmin, ref_xmax, ref_dx)
print('ref_xmin, ref_xmax = ', my_xmin, my_xmax, my_xmax-my_xmin)  
print()
print('ref_xmin, ref_ymax = ', ref_ymin, ref_ymax, ref_dy)
print('ref_xmin, ref_ymax = ', my_ymin, my_ymax, my_ymax-my_ymin)  
print()
print('ref_zmin, ref_zmax = ', ref_zmin, ref_zmax, ref_dz)
print('ref_zmin, ref_zmax = ', my_zmin, my_zmax, my_zmax-my_zmin)  
print()

for i, axis_B0  in enumerate([list_axis_B0[0]]):
# for i, axis_B0  in enumerate(list_axis_B0):
    r_calc = RotatingVector()
    B_clock, _, B_antic, _ = r_calc.get_rotation_pure_osc(list_vcos,
                                                          list_vsin, 
                                                          axis_B0)
    # Make sure that it is an array of float
    B_clock = np.array(B_clock, dtype='float')
    B_antic = np.array(B_antic, dtype='float')
    
    # =============================================================================
    # Visualize that 
    # =============================================================================
    # Check the volume field
    slicy = GUIimage()
    slicy.show()
    slicy.set_list_image(B_clock)
   
    # =============================================================================
    # Save as a nifty file 
    # =============================================================================

    # Save the nifty file
    index_version = 2 
    
    filename = 'B1_CLOCKY v%d.nii'%index_version
    
    save_for_ITKsnap(B_clock, 
                     my_affine,
                     filename)    



























