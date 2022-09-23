# -*- coding: utf-8 -*-
"""
Created on Sat May 28 15:28:02 2022

Goal:
    Time vayring magnetic field from a general line path. 
    The whole line is vayring 

@author: client
"""
from line_segment import FieldLineSegment
import numpy as np
import traceback
# Very usefull command to use for getting the last-not-printed error
# Just type _p() in the console 
_p = traceback.print_last

#Debug
_debug_enabled           = False


def _debug(*a):
    if _debug_enabled: 
        s = []
        for x in a: s.append(str(x))
        print(', '.join(s))
 


class FieldLinePath():
    """
    Magnetic field from Jefimenko equation from a line path. 
    
    This is the true time dependant field from E&M solution.
    We split the line path into linear segment and add the solution from these
    line segments. 
    
    It is assumed that the electric current on the line path is everywhere in 
    phase. 
    
    (see notebook 2022-05-23 for more details)
    
    """
    def __init__(self, list_vec,
                 I=1, f=1,
                 u0 = 4*np.pi*1e-7,
                 c = 299792458 ):
        """
        list_vec:
            List of positions that determines the line path. 
            Each element of the list is a tuple (x, y, z) that determines each 
            corner of the line path. 
            
        I:
            (Amp)
            Electrical current in the loop
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
        _debug('FieldLinePath.__init__')
        self.list_vec = list_vec
        self.I = I
        self.f=f
        self.u0 = u0
        self.c  = c 
        
        
        
        # Extract more info
        # Number of segment, assuming the last vector element is not connected 
        # to another segment. Hence the minus 1
        self.N_segment = len(self.list_vec) - 1 
        
    def get_field_spatial(self, x, y, z, print_progress=False):
        """
        Get only the spatial contribution of the field (ie the two vector:
            the cos and sin term)
         Useful for studying the rotating components
        """
        _debug('FieldLinePath.get_field_spatial')

        # We will add the terms separately, as a vector to encode the spatial dependency
        self.vec_cos_term, self.vec_sin_term = np.zeros((2,3), dtype=np.ndarray) 
        
        for i in range(self.N_segment):
            if print_progress:
                print('FieldLinePath.get_field_spatial --> Calculating %d/%d'%(i, self.N_segment))
            
            # Create the line segement
            self.vec_r1 = self.list_vec[i]
            self.vec_r2 = self.list_vec[i+1]
            self.field_segment = FieldLineSegment(self.vec_r1, 
                                                  self.vec_r2, 
                                                  I=self.I, 
                                                  f=self.f,
                                                  u0 = self.u0,
                                                  c = self.c )
            
            # Obtain the vector term of the field (without the cos and sin term)
            (new_cos_term, 
             new_sin_term) = self.field_segment.get_field_spatial(x, y, z, 
                                                                  cst='just_scale')
            # Add this, element per element
            for i in range(3):
                self.vec_cos_term[i] += new_cos_term[i]
                self.vec_sin_term[i] += new_sin_term[i]

        # We now have the vector component of the cos(wt) and sin(wt) term.
        # We just need to add the proper scaling constant and the vectors
        cst = self.u0*self.I/(4*np.pi) 
        
        return cst*self.vec_cos_term, cst*self.vec_sin_term
        

    def get_field(self, x, y, z, t, print_progress=False):
        """
        Get the magnetic field a time t and position x,y,z. 
        
        """
        _debug('FieldLinePath.get_field')
        
        # We will add the field from each line segment. 
        # Since the time dependency factors out in cos and sin terms, let's
        # Compute and add only the spatial term. With the constant
        self.vec_cos_term, self.vec_sin_term = self.get_field_spatial(x, y, z,
                                                                      print_progress=print_progress)
               
        # We now have the vector component of the cos(wt) and sin(wt) term.
        # We just need to add  the vectors
        self.w = 2*np.pi*self.f # Use the radial frequency
        time_cos, time_sin = np.cos(self.w*t), np.sin(self.w*t)
        self.Bx = (self.vec_cos_term[0]*time_cos + 
                   self.vec_sin_term[0]*time_sin )
        self.By = (self.vec_cos_term[1]*time_cos + 
                   self.vec_sin_term[1]*time_sin )    
        self.Bz = (self.vec_cos_term[2]*time_cos + 
                   self.vec_sin_term[2]*time_sin )  
        
        return self.Bx, self.By, self.Bz


if __name__ == '__main__':
    
    # Verify some solution

    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('default')    
    
    
    # =============================================================================
    # Sanity check with the circular loop    
    # =============================================================================
    # Define the circular loop of radiur R, on the x-y plane centered on the z
    # axis. 
    f = 1*297.2*1e3  # Rad/s Frequency of oscillating current.
    I = 2.3 # Amp, amplitude of the oscillating current in the line path
    R = 0.01 # meter, radius of the loop. 
    # Parametrix description
    N_element = 50 # Number of points define the loop
    list_pts_loop = [] # Will store the points defining the path
    for i in range(N_element+1): # Include the last point !
        s = i/N_element
        x = R*np.cos(s*2*np.pi)
        y = R*np.sin(s*2*np.pi)
        z = 0
        vec = (x, y, z)
        list_pts_loop.append(vec)
    
    self = FieldLinePath(list_pts_loop, I=I, f=f)
    
    # Compare with a simulation that as proof itself 
    from model_circular_loop import FieldLoop
    loop = FieldLoop(R=R, I=self.I, f=f, u0=self.u0, c=self.c)
    
    # =============================================================================
    #     # Check a single point in space far from the source
    # =============================================================================
#     x, y, z = (10*R, 10*R, 10*R)
#     list_t = np.linspace(0, 2/f, 100)
    
#     list_Bz, list_Bx, list_By  = [], [], []
#     list_Bz_loop, list_Bx_loop, list_By_loop  = [], [], []
#     print('Calculating all times at single pts in space...')
#     for i, t in enumerate(list_t):
# #        print('Calculating time %d/%d'%(i, len(list_t)))
#         # The linepath solution
#         Bx, By, Bz = self.get_field(x, y, z, t)
#         list_Bz.append(Bz)
#         list_Bx.append(Bx)
#         list_By.append(By)   
#         # The all-done script
#         Bx_loop, By_loop, Bz_loop = loop.trap_integrate(x, y, z, t)
#         list_Bz_loop.append(Bz_loop)
#         list_Bx_loop.append(Bx_loop)
#         list_By_loop.append(By_loop)         

#     plt.figure(tight_layout=True)
#     plt.title('Check dipolar soln')
#     plt.plot(list_t, list_Bz, label='Line path',  linewidth=4)
#     plt.plot(list_t, list_Bz_loop, '--', label='Loop script', linewidth=4)
#     plt.xlabel('t (s) ')
#     plt.ylabel('Bz (SI)')
#     plt.legend()       


# =============================================================================
#     # vary position into space at fixed time
# =============================================================================
    # list_pos_z = np.linspace(-1*R, 1*R, 87)
    # list_pos_x, list_pos_y = 2*R +0*list_pos_z, 0*R+0*list_pos_z
    list_pos_z = np.linspace(-1.5*R, 1*R, 87)
    list_pos_x, list_pos_y = 0.5*R +0*list_pos_z, 1.2*R+0*list_pos_z
    
    t = 0.2/f # Fix the time
    list_Bz = []
    list_Bx = []
    list_By = []
    print('Calculating along a line in space')
    (list_Bx, 
     list_By, 
     list_Bz) = self.get_field(list_pos_x, 
                               list_pos_y, 
                               list_pos_z, 
                               t,
                               print_progress=True) 
    (list_Bx_loop, 
     list_By_loop, 
     list_Bz_loop) = loop.trap_integrate(list_pos_x,
                                         list_pos_y, 
                                         list_pos_z, t, N=self.N_segment ) 
    print('Done')

    
    # =============================================================================
    #     # Check the vector field
    # ============================================================================
    from mpl_toolkits import mplot3d # Important for using the 3D projection
    # Create the list of x, y, z
    N_pts = len(self.list_vec)
    xline, yline, zline = np.zeros((3, N_pts))
    for i in range(N_pts):
        xline[i], yline[i], zline[i] = self.list_vec[i]
        
        
    # Show that !
    # Show that !
    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    
        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''
    
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
    
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
    
        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
    
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    ax = plt.subplot(projection='3d')
    ax.plot3D(xline, yline, zline, 'red', label='Wire Path')  
    
    # Add the vector field
    # Scale the b field to be easily seen on the geometry
    list_norm = (list_Bx**2 + 
                  list_By**2 +
                  list_Bz**2)**0.5
    # list_norm = (list_Bx_loop**2 + 
    #              list_By_loop**2 +
    #              list_Bz_loop**2)**0.5
    scaling_factor = 0.4*R/np.max(list_norm)
        
    ax.quiver(list_pos_x, list_pos_y, list_pos_z, 
              list_Bx*scaling_factor, 
              list_By*scaling_factor, 
              list_Bz*scaling_factor, color='C0', label='Line-Path')
    # ax.quiver(list_pos_x, list_pos_y, list_pos_z, 
    #           list_Bx_loop*scaling_factor, 
    #           list_By_loop*scaling_factor, 
    #           list_Bz_loop*scaling_factor, color='C1', label='Loop script')    
    ax.legend(loc='best')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    set_axes_equal(ax)
    plt.show()
    

    
    plt.figure(tight_layout=True)
    plt.subplot(311)
    plt.title('x, y = %.1f, %.1f radius'%(x/R, y/R))
    plt.ylabel('Bx (SI)')    
    plt.plot(list_pos_z*100, list_Bx, label='Line path',  linewidth=4)
    plt.plot(list_pos_z*100, list_Bx_loop, '--', label='Loop script', linewidth=4)
    plt.subplot(312)
    plt.ylabel('By (SI)')    
    plt.plot(list_pos_z*100, list_By, label='Line path',  linewidth=4)
    plt.plot(list_pos_z*100, list_By_loop, '--', label='Loop script', linewidth=4)
    plt.subplot(313)
    plt.ylabel('Bz (SI)')    
    plt.plot(list_pos_z*100, list_Bz, label='Line path',  linewidth=4)
    plt.plot(list_pos_z*100, list_Bz_loop, '--', label='Loop script', linewidth=4)
    plt.xlabel('z (cm) ')
    plt.legend()      
 
    
    
# 
#    
    
    
    

        
        
        