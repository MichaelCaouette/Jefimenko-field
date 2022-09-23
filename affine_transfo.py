# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:17:17 2022

Goal:
    Define the correct affine transformation. 
    And some cool function

@author: mcaoue2
"""



import numpy as np


def apply_affine(affine, i, j, k):
   """ 
   Return X, Y, Z coordinates for i, j, k 
   
   affine is the 4x4 matrix describing the affine transformation
   
   """
   M = affine[:3, :3]
   abc = affine[:3, 3]
   return M.dot([i, j, k]) + abc


def extract_affine(nifty):
    """
    ""

    Parameters
    ----------
    nifty : nifty file
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #TODO I should define this function
    return

def get_affine(xmin, xmax, Nx,
               ymin, ymax, Ny,
               zmin, zmax, Nz):
    # Define the affine matrix
    mx = (xmax - xmin)/(Nx - 1)
    my = (ymax - ymin)/(Ny - 1)
    mz = (zmax - zmin)/(Nz - 1)
    ax, ay, az = xmin, ymin, zmin
    
    affine_matrix = [[mx, 0 , 0 , ax],
                     [0 , my, 0 , ay],
                     [0 ,  0, mz, az],
                     [0 ,  0,  0,  1]]
    return np.array(affine_matrix)

def add_rotation(A, kvec, angle):
    """
    Add a rotation to the affine transformation. 

    Parameters
    ----------
    A : 4x4 matrix 
        Initial affine matrix
    kvec : 3x1 vector
         Unit vector around which we rotate.
    angle : float
        Angle, in radian, of how much we rotate around kvec

    Returns
    -------
    new_affine : 4x4 matrix 
        The new affine transformation

    """
    
    # To get the rotation matrix, I use the Rodriguess formulas  from:
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    # Make sure the rotation vector is normalized
    kx, ky, kz = np.array(kvec)/np.linalg.norm(kvec)
    
    # Matri representation of the vector cross product
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])
    # Rotation matrix
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K, K)
    
    
    # Add this to the affine transformation
    M = np.dot(A[:3, :3] ,R) # The rotation is added to the scaling part of the affine
    
    # Define the new affine
    # There might be cleaner way to do it, but this method makes explicit what is going on. 
    new_affine = np.array([[M[0][0], M[0][1], M[0][2], A[0][3]],
                           [M[1][0], M[1][1], M[1][2], A[1][3]],
                           [M[2][0], M[2][1], M[2][2], A[2][3]],
                           [0      , 0      , 0      , A[3][3]] ] )
    
    return new_affine
    
def add_translation(A, rvec):
    """
    Add a translation to the affine transformation

    Parameters
    ----------
    A : 4x4 matrix 
        Initial affine matrix
    rvec : 3x1 vector
        Translation to add to the affine

    Returns
    -------
    new_affine : 4x4 matrix 
        The new affine transformation

    """
    rnew = A[:3, 3] + np.array(rvec)
    
    
    new_affine = np.array([[A[0][0], A[0][1], A[0][2], rnew[0]],
                           [A[1][0], A[1][1], A[1][2], rnew[1]],
                           [A[2][0], A[2][1], A[2][2], rnew[2]],
                           [0      , 0      , 0      , 1      ] ])    
    return new_affine
                              


def sanitycheck_rotate(pts, n_vec, angle):
    """
    Another function to rotate, but using the explicit rotation matrix. 
    I put this function here just to check that the rotation works. 
    
    Apply a rotation from an axis and a angle. 
    Thanks to wikipedia:
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    Parameters
    ----------
    pts : list (x,y,z)
        3D point to rotate.
    n_vec : list(nx, ny, nz)
        Vectore pointing in the direction of the axis to rotate
    angle : float
        Angle to rotate

    """
    # From
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    
    # Get the normalized vector
    ux, uy, uz = np.array(n_vec)/np.linalg.norm(n_vec)
    
    cosA = np.cos(angle)
    sinA = np.sin(angle)
    
    # Build the rotation matrix
    R11 = cosA + ux**2*(1-cosA)
    R12 = ux*uy*(1-cosA) - uz*sinA
    R13 = ux*uz*(1-cosA) + uy*sinA
    
    R21 = uy*ux*(1-cosA) + uz*sinA
    R22 = cosA + uy**2*(1-cosA)
    R23 = uy*uz*(1-cosA) - ux*sinA
    
    R31 = uz*ux*(1-cosA) - uy*sinA
    R32 = uz*uy*(1-cosA) + ux*sinA
    R33 = cosA + uz**2*(1-cosA)
    
    rot_matrix = np.matrix([[R11, R12, R13],
                            [R21, R22, R23],
                            [R31, R32, R33]])
    
    my_dot = np.dot(pts, rot_matrix)
    
    return np.array(my_dot)[0]

if __name__ == '__main__':
    
    # Set numpy to print 3 decimal points and suppress small values
    np.set_printoptions(precision=3, suppress=True)
    
    ww = 20 # mm, Sets the size of our system. 
    Nx, Ny, Nz = 160, 175, 125
    x = np.linspace(-2.3*ww, 2*ww, Nx)
    y = np.linspace(-1.8*ww, 1.7*ww, Ny)
    z = np.linspace(-1*ww, 3*ww, Nz)
    xMesh, yMesh, zMesh = np.meshgrid(x, y, z)
    
    # Define a volume for these points
    # Two gaussian bumps
    d1 = ((xMesh-0.2*ww)**2+(yMesh+0.1*ww)**2+(zMesh-0.0*ww)**2)/(0.04*ww**2)
    d2 = ((xMesh-0.0*ww)**2+(yMesh+0.0*ww)**2+(zMesh-1.2*ww)**2)/(0.03*ww**2)
    my_vol = np.exp( -d1 ) + 0.8*np.exp( -d2 )
    
    
    # Original affine
    A_ori = get_affine(min(x), max(x), Nx, 
                        min(y), max(y), Ny, 
                        min(z), max(z), Nz)
    print('A_ori = \n', A_ori)
    
    # rotate it
    A_rot = add_rotation(A_ori, [0, 8, 0], 0.5*np.pi)
    print('A_rot = \n', A_rot)
    
    # Add a translation
    A_rtr = add_translation(A_rot, [ww, 0, 0.5*ww])
    print('A_rtr = \n', A_rtr)
    
    # Optional: Test the nifty file !
    import nibabel as nib # To create a nifty file with the 3D volume
    new_image = nib.Nifti1Image(my_vol, 
                                affine=A_ori)
    nib.save(new_image, 'affine_test_ori.nii' )
    new_image = nib.Nifti1Image(my_vol, 
                                affine=A_rot)
    nib.save(new_image, 'affine_test_rot.nii' )
    new_image = nib.Nifti1Image(my_vol, 
                                affine=A_rtr)
    nib.save(new_image, 'affine_test_rot_trans.nii' )    
    
    
    
    
    
    
    













