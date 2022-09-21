# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:02:47 2022

@author: mcaoue2
"""

from rotating_field import RotatingVector
from model_array_loop import ArrayLoop
from viewer_wire_field_3D import Show3DVectorField

import model_array_loop
model_array_loop._debug_enabled = True

import nibabel as nib # To create a nifty file with the 3D volume
from imageviewer.images import  GUIimage # To view the 3D volume with personal script

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('default')    

# =============================================================================
# # Define the single loop, and the system
# =============================================================================
radius_loop = 0.024 # m, Radius of the loop

# Let's list a couple of axis on which we will check the B1 rotating component
# Axis of the bore hole, or the B0 field (for the rotating calculation)
list_axis_B0 = [[0, 1, 0  ],
                [0, 1, 0.5],
                [0, 1, 1  ],
                [0, 0.5, 1],
                [0, 0, 1  ]] 

self = ArrayLoop(R=radius_loop, 
                 R_bending=-1, # No bending
                 N_loop_ver=1, N_loop_hor=1, # Just one loop
                 pos_init = [0, 0, 0], # Centered
                 dist_btw_loop=1.5*1, # Doesn't matter, because we have just one loop.
                 n_vec = [0, 0, 1],  # Point in the z direction
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

# A line cute through the system
# Straight but along x and y
list_probe_x= np.linspace(-0.01, 0.035, 87)
list_probe_y = (0.02+0.01) - list_probe_x
list_probe_z = 0.02 + 0*list_probe_x

# A volume
x = np.linspace(-2*radius_loop, 2*radius_loop, 20)
y = np.linspace(-2*radius_loop, 2*radius_loop, 30)
z = np.linspace(0.5*radius_loop, 3*radius_loop, 10)

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

for i, axis_B0  in enumerate(list_axis_B0):
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
    slicy.set_list_image(B_clock.T)
    
    
    # =============================================================================
    # Save as a nifty file 
    # =============================================================================
    
    index_version = 6
    my3Dstuff = B_clock.T
    filename = 'B1_%dx%dx%d_index%d_B0axisX%.3dY%.3dZ%.3d_v%d_CLOCKY.nii'%(np.shape(my3Dstuff)[0], 
                                                             np.shape(my3Dstuff)[1],
                                                             np.shape(my3Dstuff)[2],
                                                             i,
                                                             axis_B0[0]*100,
                                                             axis_B0[1]*100,
                                                             axis_B0[2]*100,
                                                             index_version) 
    # I should check the affine
    new_image = nib.Nifti1Image(my3Dstuff, 
                                affine=np.eye(4))
    nib.save(new_image, filename )
    
    my3Dstuff = B_antic.T
    filename = 'B1_%dx%dx%d_index%d_B0axisX%.3dY%.3dZ%.3d_v%d_ANTICLOCKY.nii'%(np.shape(my3Dstuff)[0], 
                                                                 np.shape(my3Dstuff)[1],
                                                                 np.shape(my3Dstuff)[2],
                                                                 i,
                                                                 axis_B0[0]*100,
                                                                 axis_B0[1]*100,
                                                                 axis_B0[2]*100,
                                                                 index_version) 
    # I should check the affine
    new_image = nib.Nifti1Image(my3Dstuff, 
                                affine=np.eye(4))
    nib.save(new_image, filename )
    
    print('np.sum(np.abs(B_clock.T-B_antic.T))/np.sum(np.abs(B_clock.T))  = ', 
          np.sum(np.abs(B_clock.T-B_antic.T))/np.sum(np.abs(B_clock.T)) )
    
    
    
    



























