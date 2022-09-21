# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:40:20 2022

Goal: 
    Implement the array loop geomtery with the delay offset (if any)

@author: mcaoue2
"""


import numpy as np
from line_path import FieldLinePath


#Debug
_debug_enabled           = False
def _debug(*a):
    if _debug_enabled: 
        s = []
        for x in a: s.append(str(x))
        print(', '.join(s))

def get_pringle_pts(x0, y0, z0, n_vec, 
                    R_cylinder, R_loop,
                    phi_0=0, 
                    Npts=50):
    """
    Generate a loop that is bent onto a cylinder (ie a pringle chips 
                                                  perimeter). 

    Parameters
    ----------
    x0, y0, z0:
        Center of the loop in the cylinder. 
    n_vec : TYPE
        Vector perpendicular to the pringle center (toward the radial 
                                                    coordinate of the cylinder).
    R_cylinder : TYPE
        Radius of the cylinder
    R_loop : TYPE
        Radius of the loop
        
    phi_0:
        Initial angle around the cylinder (if zero, n_vec is perpendicular
                                           to the center)
        
    Npts : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    None.

    """

    # Determine the roating axis and angle
    n_hat = np.array(n_vec)/np.linalg.norm(n_vec)
    angle_rot = np.arccos(n_hat[1]) # Angle 
    axis_rot = np.cross(n_hat, [0, 1, 0]) # The reference is the y axis
    
    # Define the loop
    list_pts = [] # Will store the points defining the loop
    for i in range(Npts+1): # Include the last point !
        s = i/Npts    
        
        theta  = s*2*np.pi
        height = R_loop*np.cos(theta)
        phi     = R_loop/R_cylinder*np.sin(theta) + phi_0
        
        # Cylindrical coordinate
        x1 = R_cylinder*np.sin(phi) - R_cylinder*np.sin(phi_0)
        y1 = R_cylinder*np.cos(phi) - R_cylinder*np.cos(phi_0) 
        z1 = height
        
        # Rotate it
        x2, y2, z2 = rotate_pts([x1, y1, z1], 
                                axis_rot, 
                                angle_rot)    

        # Translate it
        x3, y3, z3 = (x2+x0, y2+y0, z2+z0)
        
        list_pts.append( (x3, y3, z3) )
        
    return list_pts


def get_loop_pts(x0, y0, z0, n_vec, R, Npts=50):
    """
    

    Parameters
    ----------
    R : Float
        Radius of the loop
    Npts : int, optional
        Number of point to define the loop. The default is 50.

    Returns
    -------
    None.

    """
    
    # Determine the roating axis and angle
    n_hat = np.array(n_vec)/np.linalg.norm(n_vec)
    
    # # Some way to get the angle from complex number. 
    # cx = n_hat[2]
    # cy = (1 - cx**2)**0.5
    # angle_rot = np.angle(cx+cy*1j) #  # Angle 
    angle_rot = np.arccos(n_hat[2]) # Angle of rotation
    
    
    axis_rot = np.cross(n_hat, [0, 0, 1]) # The reference is the z axis
    
    # Define the loop
    list_pts = [] # Will store the points defining the loop
    for i in range(Npts+1): # Include the last point !
        s = i/Npts
        
        # pts in x-y plane
        x1 = R*np.cos(s*2*np.pi)
        y1 = R*np.sin(s*2*np.pi)
        z1 = 0
        
        # Rotate it only if the axis of rotation is not null (in case the n_hat
        # was the reference)
        if not (np.linalg.norm(axis_rot) ==0):
            x2, y2, z2 = rotate_pts([x1, y1, z1], 
                                    axis_rot, 
                                    angle_rot)
        else:
            x2, y2, z2 = x1, y1, z1
        # Translate it
        x3, y3, z3 = (x2+x0, y2+y0, z2+z0)
        
        list_pts.append( (x3, y3, z3) )
        
    return list_pts



def rotate_pts(pts, n_vec, angle):
    """
    
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

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # From
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    
    # Get the normalized vector
    ux, uy, uz = np.array(n_vec)/np.linalg.norm(n_vec)
    
    # n_norm = (n_vec[0]**2 + 
    #           n_vec[1]**2 + 
    #           n_vec[2]**2   )**0.5
    # ux = n_vec[0] / n_norm
    # uy = n_vec[1] / n_norm
    # uz = n_vec[2] / n_norm
    
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
    

class ArrayLoop():
    """
    Magnetic field from Jefimenko equation from a loop array (used in MRI) 
    
    This is the true time dependant field from E&M solution. 
    
    
    """
    def __init__(self, 
                 R=1, R_bending=-1, 
                 N_loop_ver=3, N_loop_hor=4,
                 pos_init = [0, 0, 0], 
                 dist_btw_loop=1.5*1, # half overlap
                 n_vec = [0, 1, 0],  # y direction  
                 I_loop=1, 
                 N_dscrt_pts_loop=100,
                 f=1,
                 u0 = 4*np.pi*1e-7,
                 c = 299792458 ):
        """
        R:
            Meter
            Radius of the loop
            
        R_bending:
            Determine how bent are the loops. 
            This is the radius of the cylinder on which the loops are mapped. 
            For no bending, put -1 (the default is -1)
        
        N_loop_ver:
            (int)
            Number of loop in the vertical direction.

        N_loop_hor:
            (int)
            Number of loop in the horizontal direction.
            
        pos_init = [0, 0, 0]
            np.array of size 3, floats
            Optional. 
            Position of the first loop. All the other loop will be defined by
            a translation-rotation about this initial position.

        dist_btw_loop:
            (float) optional
            Meter, Distance between the loops.
            
        n_vec = [0, 1, 0]
            np.array of size 3, floats
            Optional. 
            Orientation of the loop. (axis perpendicular to the loop)
        
        I_loop:
            (Amp)
            Maximum electrical current in a loop.
            
        N_dscrt_pts_loop:
            (int)
            Number of element to discretize a loop (number of segment used
            to define it). The more the better (because smoother circle).
        
        f:
            (Hz)
            Frequency of the current oscillation in the loop. 
            Note that t is not w !  
            For example,  2*pi*f = w, such that gamma*B=w=2*pi*f 
            (For n magnetization dynamic)

        u0 = 4*np.pi*1e-7:
            H/m Magnetic constant
            
        c =  299792458:
             m/s Speed of light in vaccuum
        """
        _debug('ArrayLoop.__init__')
        
        self.R = R
        self.R_bending = R_bending
        self.N_loop_ver = N_loop_ver
        self.N_loop_hor = N_loop_hor
        self.I_loop = I_loop
        self.N_dscrt_pts_loop = N_dscrt_pts_loop
        self.dist_btw_loop = dist_btw_loop
        self.n_vec = n_vec
        self.pos_init = pos_init 
        
        self.f = f
        self.w = 2*np.pi*f
        self.u0 = u0
        self.c  = c 

        # Displacement vector for each horizontal or vertical direction
        # PLease don't change this before I make sure to understand how to generalire other angle
        self.disp_vec_hor = [self.dist_btw_loop, 0] # Horizontal displacement
        self.disp_vec_ver = [0, self.dist_btw_loop] # Vertical   displacement
        
        # =============================================================================
        #         # Prepare the geometry and delay. 
        # =============================================================================
        self.list_wire  = [] # Will hold the various wire geometry
        self.list_delay = [] # Will hold the corresponding delay
        # Each element consist of a line path and has an associated delay.
        
        self._update_wire()
        
    
        
    def _update_wire(self):
        """
        Define the set of wires that will make the current. 

        Returns
        -------
        None.

        """
        _debug('ArrayLoop._update_wire')
        
        if self.R_bending == -1:
            self._define_flat_loops()
        else:
            self._define_bent_loops()
        
    def _define_flat_loops(self):
        """
        Set the wires to be an array of flat loop
        
        """
        _debug('ArrayLoop._define_flat_loops')
        
        # Horizontal is the x axis
        for i_hor in range(self.N_loop_hor):
            # Vertical is the z axis
            for i_ver in range(self.N_loop_ver):
                # Define the loop centers 
                # Based on the displacement for each direction.
                x0 = i_hor*self.dist_btw_loop + self.pos_init[0]
                y0 = 0 + self.pos_init[1]
                z0 = i_ver*self.dist_btw_loop + self.pos_init[2]
                
                # Get the list of pts making the loop
                list_pts = get_loop_pts(x0, y0, z0, self.n_vec,
                                        self.R, 
                                        Npts=self.N_dscrt_pts_loop)
                
                # Create the wire
                wire = FieldLinePath(list_pts,
                                     I =self.I_loop , 
                                     f =self.f,
                                     u0=self.u0,
                                     c =self.c  )
                self.list_wire.append(wire)
                # Delay
                # This should be an input for the user
                self.list_delay.append(0.25*2*np.pi*i_hor/self.N_loop_hor)
        
        # Note the number of element that makes our field
        self.N_wire = len(self.list_wire)        

    def _define_bent_loops(self):
        """
        Set the wires to be an array of bend loop. 
        (Basically flat loops mapped on a cylinder. )
        
        """
        _debug('ArrayLoop._define_bent_loops')
        
        # Horizontal is the x axis
        for i_hor in range(self.N_loop_hor):
            # Vertical is the z axis
            for i_ver in range(self.N_loop_ver):
                # Define the loop centers 
                # Based on the displacement for each direction.
                # Since it is on the coordinate of a cylinder, we translate
                # the horizontal displacement into an angular one.
                angle = i_hor*self.dist_btw_loop/self.R_bending
                z = i_ver*self.dist_btw_loop
                                
                x0 = self.R_bending*np.cos(angle)
                y0 = self.R_bending*np.sin(angle)
                z0 = z
                
                n_dir = [x0, y0, 0]
                
                # Get the list of pts making the loop
                list_pts = get_pringle_pts(x0, y0, z0, n_dir, 
                                           self.R_bending, self.R, phi_0=0*np.pi/180,
                                           Npts=self.N_dscrt_pts_loop)
                
                # Create the wire
                wire = FieldLinePath(list_pts,
                                     I =self.I_loop , 
                                     f =self.f,
                                     u0=self.u0,
                                     c =self.c  )
                self.list_wire.append(wire)
                # Delay
                # Small delay (could be zero)
                self.list_delay.append(0.25*i_hor/(self.N_loop_hor-1)/self.f)
        
        # Note the number of element that makes our field
        self.N_wire = len(self.list_wire)         
        
    def get_field_from_spatial(self, vcos, vsin, t):
        """
        Convenient method to get the field when we already have the spatial 
        components. 
        """
        _debug('ArrayLoop.get_field_from_spatial')
        
        phase = self.w*t
        return vcos*np.cos(phase) + vsin*np.sin(phase)
        
    def get_field_spatial(self, x, y, z, print_progress=False):
        """
        Get only the spatial contribution of the field (ie the two vector:
            the cos and sin term)
         Useful for studying the rotating components   
         
         
         return self.vcos, self.vsin
        """
        _debug('ArrayLoop.get_field_spatial')
        
        for i in range(self.N_wire):
            # Important to add the delay
            wire = self.list_wire[i]
            out =  wire.get_field_spatial(x, y, z, print_progress=print_progress)
            # Add the effect of the phase
            phase = self.w*self.list_delay[i]
            A, D = out
            # The effect is a slight shift in the cos and sin component
            # (See my notebook 2022-07-13 for the trigonometry trick)
            vcos = A*np.cos(phase) - D*np.sin(phase)
            vsin = A*np.sin(phase) + D*np.cos(phase)
            
            # Add them
            if i == 0:
                self.vcos, self.vsin = vcos, vsin
            else:
                self.vcos += vcos
                self.vsin += vsin
        
        return self.vcos, self.vsin
        
    def get_field(self, x, y, z, t, print_progress=False):
        """
        Get the magnetic field a time t and position x,y,z. 
        
        """
        _debug('ArrayLoop.get_field')
        
        # We will add the field from each element plus the delays
        
        for i in range(self.N_wire):
            # Important to add the delay
            wire = self.list_wire[i]
            out =  wire.get_field(x, y, z, 
                                  t-self.list_delay[i],
                                  print_progress=print_progress)
            if i == 0:
                self.Bx, self.By, self.Bz = out
            else:
                self.Bx += out[0]
                self.By += out[1]
                self.Bz += out[2]
        # Done !
        return self.Bx, self.By, self.Bz
            
            
            
        
if __name__ == '__main__':
    import line_path
    line_path._debug_enabled = False
    _debug_enabled = True
       
    # from viewer_3D import plot_3D_cool
    from viewer_wire_field_3D import Show3DVectorField, list_pts_to_each_lines    
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('default')    
      
    # =============================================================================
    # Verify some geometry
    # =============================================================================
    # Test that our geometry is giving us what we expect
    v_geo = Show3DVectorField()
    v_geo.ax.set_title('Various shape for the pringle')

    # Center of loops
    x0, y0, z0 = 0.4, -0.08, 0.09
    # Vector perpendicular to the place of the center of the loop
    n_direction = [4, 0, 3.5]#[0, 0, 4] #[4, -5, 3.5]
    # Radius of the loop
    R_loop = 0.02 # Meter
    
    # Parameters related to the bending and radius
    R_cyl = 0.07 # Meter
    out = get_pringle_pts(x0, y0, z0, n_direction , 
                                          R_cyl, R_loop,
                                          Npts=50)
    xline, yline, zline = list_pts_to_each_lines(out) 

    v_geo.add_path(xline, yline, zline, color='C1', 
                   label='Pringle R_cyl=%.4f'%R_cyl, show_corner=True)

    # Parameters related to the bending and radius
    R_cyl = 0.02 # Meter
    out = get_pringle_pts(x0, y0, z0, n_direction , 
                                          R_cyl, R_loop,phi_0=0*45*np.pi/180,
                                          Npts=50)
    xline, yline, zline = list_pts_to_each_lines(out) 
    v_geo.add_path(xline, yline, zline, color='C2', 
                   label='Pringle R_cyl=%.4f'%R_cyl, show_corner=True)

    # Parameters related to the bending and radius
    R_cyl = 0.02 # Meter
    phi_0 = 45*np.pi/180 # rad
    out = get_pringle_pts(x0, y0, z0, n_direction , 
                                          R_cyl, R_loop,phi_0=phi_0,
                                          Npts=50)
    xline, yline, zline = list_pts_to_each_lines(out) 
    v_geo.add_path(xline, yline, zline, color='C3', 
                   label='Pringle R_cyl=%.4f phi_0=%.2f'%(R_cyl,phi_0), show_corner=True)
    
    # Parameters related to the bending and radius
    out = get_loop_pts(x0, y0, z0, n_direction, 
                                       R_loop,
                                       Npts=50)
    xline, yline, zline = list_pts_to_each_lines(out) 
    v_geo.add_path(xline, yline, zline, color='C3', 
                   label='Loop ', show_corner=True)
    
    # Add other stuff for comparision
    v_geo.ax.plot(x0, y0, z0, color='red', marker='o')
    v_geo.add_vector_field([x0], [y0], [z0],
                           [n_direction[0]], [n_direction[1]], [n_direction[2]],
                           scale=R_loop, label='v_perp', color='C0')    
    
    # =============================================================================
    # Test the geometry of the array of loops
    # =============================================================================
    # For cycling into any cmap
    import matplotlib
    # cmap = matplotlib.cm.get_cmap('rainbow')
    cmap = matplotlib.cm.get_cmap('brg')
    # Check up that it works
    rgba = cmap(0.5)
    print(rgba) # (0.99807766255210428, 0.99923106502084169, 0.74602077638401709, 1.0)



    self = ArrayLoop(R=0.01, R_bending=0.04,
                     N_loop_ver=4, N_loop_hor=4,
                     I_loop=2.3, 
                     dist_btw_loop=0.01*3, # No overlap, just to see that the grid works
                     N_dscrt_pts_loop=50,
                     f=1*297.2*1e6)
    
    
# =============================================================================
#     # Check if the field rotates at the center
# =============================================================================
    # List some times to probe. 
    list_t_probe = np.linspace(0, 1, 27)/self.f 
    x_probe = 0.02
    y_probe = 0.01
    z_probe = 0.02
    (list_Bx, 
     list_By, 
     list_Bz) = self.get_field(x_probe, 
                               y_probe, 
                               z_probe, 
                               list_t_probe,
                               print_progress=False)     
    list_probe_axis = list_t_probe
    plt.figure(tight_layout=True)
    plt.subplot(311)
    # plt.title('x, y = %.1f, %.1f radius'%(x/R, y/R))
    plt.ylabel('Bx (SI)')    
    plt.plot(list_probe_axis, list_Bx, label='',  linewidth=4)
    plt.subplot(312)
    plt.ylabel('By (SI)')    
    plt.plot(list_probe_axis, list_By, label='',  linewidth=4)
    plt.subplot(313)
    plt.ylabel('Bz (SI)')    
    plt.plot(list_probe_axis, list_Bz, label='',  linewidth=4)
    plt.xlabel('Time ')
    plt.legend()      
        
        
        
# =============================================================================
#     # Probe the field at various position and fixed time
# =============================================================================
    t_probe = 0.20/self.f # Fix the time to probe
    
    # A line cut
    # Staight cut along the z axis
    list_probe_z1 = np.linspace(-0.02, 0.06, 53)
    list_probe_y1 = 0.01   + 0*list_probe_z1
    list_probe_x1 = 0.02   + 0*list_probe_z1
    
    # Straight but along x and y
    list_probe_x2 = np.linspace(-0.01, 0.05, 87)
    list_probe_y2 = (0.02+0.01) - list_probe_x2
    list_probe_z2 = 0.02 + 0*list_probe_x2
    
    # Concatenate these lines
    list_probe_x = np.concatenate((list_probe_x1, list_probe_x2))
    list_probe_y = np.concatenate((list_probe_y1, list_probe_y2))
    list_probe_z = np.concatenate((list_probe_z1, list_probe_z2))
    
    list_Bz = []
    list_Bx = []
    list_By = []
    print('Probing the fiekld into space...')
    (list_Bx, 
      list_By, 
      list_Bz) = self.get_field(list_probe_x, 
                                list_probe_y, 
                                list_probe_z,
                                t_probe,
                                print_progress=False) 
    print('Done')
    
    
#     # =============================================================================
#     # Check this out
#     # =============================================================================
    
    vv = Show3DVectorField()
    # Show all the elements (to see all the wires)
    vv.add_list_wire(self.list_wire, list_color=self.list_delay, 
                     str_cmap='brg', label='Bent array of loop',
                     color_title='Delay (Rad)',
                     show_corner=False)
    # Show the field 
    vv.add_vector_field(list_probe_x, list_probe_y, list_probe_z, 
                        list_Bx, list_By, list_Bz,
                        scale=0.05, color='C1', label='B field')
 

# 
    # Also show the field individualy
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('default')    
    list_probe_axis = np.arange(len(list_probe_x))
    plt.figure(tight_layout=True)
    plt.subplot(311)
    # plt.title('x, y = %.1f, %.1f radius'%(x/R, y/R))
    plt.ylabel('Bx (SI)')    
    plt.plot(list_probe_axis, list_Bx, label='',  linewidth=4)
    plt.subplot(312)
    plt.ylabel('By (SI)')    
    plt.plot(list_probe_axis, list_By, label='',  linewidth=4)
    plt.subplot(313)
    plt.ylabel('Bz (SI)')    
    plt.plot(list_probe_axis, list_Bz, label='',  linewidth=4)
    plt.xlabel('Along line (arbr. unit) ')
    plt.legend()      

    
    
        
        
        
        