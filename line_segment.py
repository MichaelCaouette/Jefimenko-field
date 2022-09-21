# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:05:48 2022

@author: 
    Michael Caouette-Mansour
    Amir Shmuel and David Rudko laboratory

"""

import numpy as np
from integral_Jeffimenko_line import IntegralJeffimenko

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
        
        
        
class FieldLineSegment():
    """
    Magnetic field from Jefimenko equation from a line segment.
    
    This is the true time dependant field from E&M solution. 
    A numerical method is used to solved the integral. 
    
    (see notebook 2022-05-23 for more details)
    
    """
    def __init__(self, vec_r1, vec_r2, 
                 I=1, f=1,
                 u0 = 4*np.pi*1e-7,
                 c = 299792458 ):
        """
        vec_r1:
            Tuple (x, y, z) determining the first end of the line segment. 
        vec_r2:
            Tuple (x, y, z) determining the second end of the line segment. 
            
        I:
            (Amp)
            Electrical current in the segment
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
        _debug('FieldLineSegment: __init__')
        self.vec_r1 = vec_r1
        self.vec_r2 = vec_r2
        self.I = I
        self.w = 2*np.pi*f
        self.u0 = u0
        self.c  = c 
        
        # The integral calculator
        self.integrator = IntegralJeffimenko(w=self.w,
                                             c= self.c)
        
    def cross_prod(self, v, u):
        """
        Compute the cross product between two vector.
        
        v and u:
            Both tuple (x, y z) defining a 3D vector.
        
        return:
            The cross product v x u
        """
        _debug('FieldLineSegment: cross_prod')
        # Cross product between the lenght element along the loop and the 
        # relative position. 
        # Simpler to do it explicitely and element wise, because using 
        # np.cross() makes it hard to interprete multiple axis in x,y,z,t
        cross_x = v[1]*u[2] - v[2]*u[1]
        cross_y = v[2]*u[0] - v[0]*u[2]
        cross_z = v[0]*u[1] - v[1]*u[0]
        return  np.array([cross_x, cross_y, cross_z], dtype=np.ndarray)
    
    def magnitude(self, vec):
        """
        Compute the magnitude of the vector vec. 
        
        vec:
            Tuple (x, y, z) of the vector 
            
        return:
            The magnitude of vec. 
        """
        _debug('FieldLineSegment: magnitude')
        
        # Do it element wise, because this way it is easier if 'vec' contains
        # arrays.
        return (vec[0]*vec[0] + 
                vec[1]*vec[1] + 
                vec[2]*vec[2]   )**0.5

    def coordinate_mapping(self, vec_r, vec_r1, vec_r2 ):
        """
        Map the original coordinates into a more convenient coordinate system
        to solve the field. 
        
        vec_r:
            Tuple (rx, ry, rz) for the coordinate of the position vector. 
            It is the position where we measure the magnetic field. 
        
        vec_r1:
            Tuple (rx, ry, rz) for the coordinate of the position vector. 
            It is the position of the begining of the line segment. 

        vec_r2:
            Tuple (rx, ry, rz) for the coordinate of the position vector. 
            It is the position of the end of the line segment. 
            
        return:
            x1, y1, lenght, n_hat
            Where all these refer to the geometry defined in my notebook 
            of 2022-05-23. 
        """
        _debug('FieldLineSegment: coordinate_mapping')
        
        # Overall note for the code:
        # I define a the numpy array and make my operation element-wise.
        # This is to deal with when the user inputs list of positions, such 
        # that the vectors contains many positions at once. 
        
    
        # Define the side of the triangle which the apex are these 3 vectors. 
        
        # Start with the differences vector
        rr1  = np.array([vec_r [0]-vec_r1[0], 
                         vec_r [1]-vec_r1[1],
                         vec_r [2]-vec_r1[2]], dtype=np.ndarray )
        rr2  = np.array([vec_r [0]-vec_r2[0], 
                         vec_r [1]-vec_r2[1],
                         vec_r [2]-vec_r2[2]], dtype=np.ndarray  )       
        r2r1 = np.array([vec_r2[0]-vec_r1[0], 
                         vec_r2[1]-vec_r1[1],
                         vec_r2[2]-vec_r1[2]], dtype=np.ndarray  )
        # Side of the triangle
        a = self.magnitude(rr2 )
        b = self.magnitude(r2r1)
        c = self.magnitude(rr1 )
        
        # New coordinate from this triangle (see notebook 2022-05-23)
        # It comes from high-school triangle geometry
        c2 = c*c
        x1 = -(c2+ b*b - a*a)/(2*b)
        y1 = (c2 - x1*x1)**0.5
        lenght = b
        
        # Unit vector pointing in the direction perpendiculare to this plan. 
        # It is the direction of the magnetic field. 
        self.n_unscaled = self.cross_prod(r2r1, rr1)
        self.n_mag = self.magnitude(self.n_unscaled )
        n_hat_x = self.n_unscaled[0]  / self.n_mag
        n_hat_y = self.n_unscaled[1]  / self.n_mag
        n_hat_z = self.n_unscaled[2]  / self.n_mag
        n_hat = np.array([n_hat_x, n_hat_y, n_hat_z])
        return x1, y1, lenght, n_hat
        
    def get_field_spatial(self, x, y, z, cst='all'):
        """
        Get the spatial vector (cos and sin term) of the magnetic field. 
        This is useful to sum up many line segment, because we don't need the 
        time dependent term everytime
        
        no_cst = 'all':
            (str)
            Constant to put in front of the B-field
            
            if all:
                It will be the full unit u0*I/(4pi) times the correct scaling
            if none:
                It will be one (useful to have quickly just the vector, unscaled)
            if 'just_scale':
                It will have the right scaling, wihtout the factor u0*I/(4pi) 
                This is useful if we are adding lines together and want to 
                add the overall u0*I/(4pi) factor at the end. 
            The whole idea is to save some time in computation.
        """
        _debug('FieldLineSegment: get_field_spatial')
        
        # Map the coordinate to the convenient coordinate system.
        (self.x1, 
         self.y1, 
         self.lenght, 
         self.n_hat) = self.coordinate_mapping((x, y, z), 
                                               self.vec_r1, 
                                               self.vec_r2 )
        # Compute the magnitude of the field. 
        # Overall constant
        if   cst=='all':
            cst = self.u0*self.I*self.y1*lenght/(4*np.pi)
        elif cst == 'just_scale':
            cst = self.y1*self.lenght
        elif cst=='none':
            cst = 1
        else:
            print('ERROR in FieldLineSegment.get_field_spatial')
            print('The variable cst must be either all, non or just_scale')
            
        # Spatial terms
        # We take the maximum discretization. 
        self.out = self.integrator.optimal_discretization(self.x1, 
                                                          self.lenght, 
                                                          self.y1)
        self.Ndiscr = np.max(self.out)
        
        term_cos, term_sin = self.integrator.get_integral_terms(self.x1, 
                                                                self.lenght, 
                                                                self.y1,
                                                                N_integ=self.Ndiscr)
        # We can now construct the cos and sin vector
        vec_cos_term, vec_sin_term = np.zeros((2,3), dtype=np.ndarray) 
        for i in range(3):
            vec_cos_term[i] = cst*self.n_hat[i]*term_cos
            vec_sin_term[i] = cst*self.n_hat[i]*term_sin      
       
        return vec_cos_term, vec_sin_term
        
    def get_field(self, x, y, z, t, cst='all'):
        """
        Get the magnetic field a time t and position x,y,z. 
        
        no_cst = 'all':
            (str)
            Constant to put in front of the B-field
            
            if all:
                It will be the full unit u0*I/(4pi) times the correct scaling
            if none:
                It will be one (useful to have quickly just the vector, unscaled)
            if 'just_scale':
                It will have the right scaling, wihtout the factor u0*I/(4pi) 
                This is useful if we are adding lines together and want to 
                add the overall u0*I/(4pi) factor at the end. 
            The whole idea is to save some time in computation.
        """
        _debug('FieldLineSegment: get_field')
        
        # Since the time dependency factors out in cos and sin terms, let's
        # Compute and add only the spatial term. With the constant        
        self.vec_cos_term, self.vec_sin_term = self.get_field_spatial(x, y, z, cst=cst)
        
        # Add the time dependency
        time_cos, time_sin = np.cos(self.w*t), np.sin(self.w*t)
        self.Bx = (self.vec_cos_term[0]*time_cos + 
                   self.vec_sin_term[0]*time_sin )
        self.By = (self.vec_cos_term[1]*time_cos + 
                   self.vec_sin_term[1]*time_sin )    
        self.Bz = (self.vec_cos_term[2]*time_cos + 
                   self.vec_sin_term[2]*time_sin )  
        
        return self.Bx, self.By, self.Bz
   
    def get_short_wire(self, x, y, z, t, cst='all'):
        """
        Approximation for a short wire compared to the wavelenght and when
        x, y, z are far from the wire. 
        
        It is basically not doing the integration. 
        (Taking the integrant to be constant)
        
        """
        _debug('FieldLineSegment: get_short_wire')

        # Map the coordinate to the convenient coordinate system.
        x1, y1, lenght, n_hat = self.coordinate_mapping((x, y, z), 
                                                        self.vec_r1, 
                                                        self.vec_r2 )
        # Compute the magnitude of the field. 
        # Overall constant
        if   cst=='all':
            cst = self.u0*self.I*y1*lenght/(4*np.pi)
        elif cst == 'just_scale':
            cst = y1*lenght
        elif cst=='none':
            cst = 1
        else:
            print('ERROR in FieldLineSegment: get_short_wire')
            print('The variable cst must be either all, non or just_scale')
            
        # Spatial terms
        # We are no integrating. Just taking the mean, which correspond in this
        # case to the intermediate value. No overall multiplication constant,
        # because the integral is from 0 to 1 (hence multiply by 1)
        (self.term_cos, 
         self.term_sin) = self.integrator._integrant_cos_sin(x1, lenght, y1,
                                                             0.5)
        
        # Various terms makes the final answer
        self.B_mag = cst*(np.cos(self.w*t)*self.term_cos +
                          np.sin(self.w*t)*self.term_sin  )
        
        # Each component of the field is the unit vector times the magnitude 
        Bx = n_hat[0]*self.B_mag
        By = n_hat[1]*self.B_mag
        Bz = n_hat[2]*self.B_mag
        
        return Bx, By, Bz
    
    def get_quasistatic(self, x, y, z, t, cst='all'):
        """
        Approximation for low frequency. 
        It consist of taking the static solution and modulating the field by 
        cos(wt).
        
        It is expected that this works well close to the wire and when the 
        wavelenght is much larger than the wire.
        
        """
        _debug('FieldLineSegment: get_quasistatic')

        # Map the coordinate to the convenient coordinate system.
        x1, y1, lenght, n_hat = self.coordinate_mapping((x, y, z), 
                                                        self.vec_r1, 
                                                        self.vec_r2 )
        # Compute the magnitude of the field. 
        # Overall constant
        if   cst=='all':
            cst = self.u0*self.I/(4*np.pi)
        elif cst=='none':
            cst = 1
        else:
            print('ERROR in FieldLineSegment: get_quasistatic')
            print('The variable cst must be either all, none or just_scale')  
            
        # Static field, from solbing the integral in the static case. 
        # For this geometry (in the plane) we have an analitical solution. 
        # See, for example, Griffith !
        x2 = x1 + lenght
        self.term_x2 = x2 / (y1*y1 + x2*x2)**0.5
        self.term_x1 = x1 / (y1*y1 + x1*x1)**0.5
        B_static = cst * (self.term_x2 - self.term_x1) / y1
        
        # The quasistatic consist of modulating the field by the time varying
        # current. 
        self.B_mag = np.cos(self.w*t)*B_static
        
        # Vectorize all this
        Bx = n_hat[0]*self.B_mag
        By = n_hat[1]*self.B_mag
        Bz = n_hat[2]*self.B_mag
        
        return Bx, By, Bz


if __name__ == '__main__':
    _debug_enabled = True
    # Verify some solution

    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('default')    
    
    
    # =============================================================================
    # Small wire compared to y and wavelenght    
    # And also Quasistatic
    # =============================================================================
    f = 1*1*297.2*1e6 # Hz Frequency of oscillating current.
    # A short wire
    # vec_r1 = (0.01, -0.03, 0)   # Meter
    # vec_r2 = (0.005, 0.02, 0)   # Meter
    
    #  A medium wire
    vec_r1 = (0.1, -0.3, 0)   # Meter
    vec_r2 = (0.05, 0.2, 0)   # Meter
    
    # A huge wire, to be used with very low frequency and compare the quasistatic
    # vec_r1 = (1, -3, 0)   # Meter
    # vec_r2 = (0.5, 2, 0)   # Meter    
    
    # List some line cut
    list_probe_z = np.linspace(-0.4, 0.4, 87)
    list_probe_x = 0.1 + 0*list_probe_z
    list_probe_y = 0.0 + 0*list_probe_z
    t_probe = 0.2/f # Fix time TODO: change this for an animation
    
    self = FieldLineSegment(vec_r1, vec_r2, f=f, I=2.3)
    
    # Check some order of magnitude in the problem
    lenght = ((vec_r1[0]-vec_r2[0])**2 +
              (vec_r1[1]-vec_r2[1])**2 +
              (vec_r1[2]-vec_r2[2])**2  )**0.5
    wavelenght = 2*np.pi*self.c/self.w
    print('Lenght     = ', lenght)
    print('Wavelenght = ', wavelenght)
    print('Lenght/Wavelenght = ', lenght/wavelenght)
    
    # Compute the Field
    (list_Bx, 
     list_By,
     list_Bz) = self.get_field(list_probe_x, 
                               list_probe_y, 
                               list_probe_z, t_probe)
    (list_Bx_short, 
     list_By_short, 
     list_Bz_short) = self.get_short_wire(list_probe_x, 
                                          list_probe_y, 
                                          list_probe_z, t_probe)
    (list_Bx_quasi, 
     list_By_quasi, 
     list_Bz_quasi) = self.get_quasistatic(list_probe_x, 
                                           list_probe_y, 
                                           list_probe_z, t_probe)


    # =============================================================================
    #     # Check the vector field
    # ============================================================================
    list_vec =  (vec_r1, vec_r2)
    from mpl_toolkits import mplot3d # Important for using the 3D projection
    # Create the list of x, y, z
    N_pts = len(list_vec)
    xline, yline, zline = np.zeros((3, N_pts))
    for i in range(N_pts):
        xline[i], yline[i], zline[i] = list_vec[i]
        
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
    scaling_factor = 0.3*lenght/np.max(list_norm)
        
    ax.quiver(list_probe_x, list_probe_y, list_probe_z, 
              list_Bx*scaling_factor, 
              list_By*scaling_factor, 
              list_Bz*scaling_factor, color='C0', label='Line-Path')
    ax.plot(list_probe_x, list_probe_y, list_probe_z, 
            color='C1', label='Line probed', linewidth=5)
    # ax.quiver(list_probe_x, list_probe_y, list_probe_z, 
    #           list_Bx_loop*scaling_factor, 
    #           list_By_loop*scaling_factor, 
    #           list_Bz_loop*scaling_factor, color='C1', label='Loop script')    
    ax.legend(loc='best')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    set_axes_equal(ax)
    plt.show()
    

    # =============================================================================
    #     # Just plot the short wire and full solution
    # =============================================================================
    plt.figure(tight_layout=True)
    plt.title('Sanity check along z axis, fixed time')
    plt.plot(list_probe_z*100, list_By, label='Numerical',  linewidth=4)
    plt.plot(list_probe_z*100, list_By_short, '--', label='Short wire',  linewidth=4)
    plt.xlabel('z (cm) ')
    plt.ylabel('By (SI)')
    plt.legend()      
    
    # Plot everyhting
    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, 
                                           sharex=True, 
                                           tight_layout=True)
    # Bx
    ax_x.set_title('Sanity check along z axis, fixed time.'+
                   '\nQuasi-static works close to wire, long wavelenght')
    ax_x.plot(list_probe_z*100, list_Bx, label='Numerical',  linewidth=4)
    ax_x.plot(list_probe_z*100, list_Bx_short, '--', label='Short wire',  linewidth=4)
    ax_x.plot(list_probe_z*100, list_Bx_quasi, '-.', label='Quasistatic', linewidth=4)
    ax_x.set_xlabel('z (cm) ')
    ax_x.set_ylabel('Bx (SI)')
    ax_x.legend(loc='upper right')  
    # By    
    ax_y.plot(list_probe_z*100, list_By, label='Numerical',  linewidth=4)
    ax_y.plot(list_probe_z*100, list_By_short, '--', label='Short wire',  linewidth=4)
    ax_y.plot(list_probe_z*100, list_By_quasi, '-.', label='Quasistatic', linewidth=4)
    ax_y.set_xlabel('z (cm) ')
    ax_y.set_ylabel('By (SI)')    
    # Bz
    ax_z.plot(list_probe_z*100, list_Bz, label='Numerical',  linewidth=4)
    ax_z.plot(list_probe_z*100, list_Bz_short, '--', label='Short wire',  linewidth=4)
    ax_z.plot(list_probe_z*100, list_Bz_quasi, '-.', label='Quasistatic', linewidth=4)
    ax_z.set_xlabel('z (cm) ')
    ax_z.set_ylabel('Bz (SI)')

    
    # =============================================================================
    #      # Check up some order of magnitude.
    # =============================================================================
    # Take the peak distance
    min_diff_r = 99999999999999999999999
    for i in range(len(list_probe_z)):
        x1, y1, z1 = list_probe_x[i], list_probe_y[i], list_probe_z[i]
        # Compute the mean distance
        mean_dist = 0
        for jj in range(len(list_vec )):
            x, y, z = list_vec[jj]
            dist = ((x-x1)**2 + 
                    (y-y1)**2 + 
                    (z-z1)**2   )**0.5
            mean_dist += dist
        # Obtain the mean distance between the probed position and wire path
        mean_dist /=  len(list_vec )
        # Update the closest distance
        if mean_dist <= min_diff_r:
            min_diff_r = mean_dist
            
        
    print()
    print('min_diff_r = ', min_diff_r, ' meter')
    print('u0*I/min_diff_r/(4*np.pi) = ', self.u0*self.I/min_diff_r/(4*np.pi))
    print()
    print('max(list_Bz) = ', max(list_Bz))
    print()
    




   

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        