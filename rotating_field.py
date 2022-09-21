# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:41:41 2022

Goal:
    Extract the rotating component of an oscillating field
    
@author: Michael Caouette-Mansour
"""

import numpy as np

class RotatingVector():
    """
    This class will help to extract the rotating (clockwise and counter) 
    components of an oscillating vector field. 
    
    More specific, the user should provide the axis perpendicular to the plane
    of rotation of interest. Than the method would gives the amplitude of the 
    two components (maybe the phase also, we will see lol)
    
    It is meant to contain all the algebraic manipulation. 
    
    """
    def __init__(self):
        return

    def cross_prod(self, v, u):
        """
        Compute the cross product between two vector.
        
        v and u:
            Both tuple (x, y z) defining a 3D vector.
        
        return:
            The cross product v x u
        """
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
        
        # Do it element wise, because this way it is easier if 'vec' contains
        # arrays.
        return (vec[0]*vec[0] + 
                vec[1]*vec[1] + 
                vec[2]*vec[2]   )**0.5
                
    def dot(self, v, u):
        # Do it element wise, because this way it is easier if the input 
        # contains arrays.
        return v[0]*u[0] + v[1]*u[1] + v[2]*u[2]
            
    def get_perp_vec(self, v_3d, a_axis):
        """
        Get the projection of the 3D vector on a plane perpendicular to a given
        axis.
        """
        # Normalize the axis
        a_hat = a_axis / self.magnitude(a_axis)
        
        # Lenght of the projection along the axis
        the_dot = self.dot(v_3d, a_hat)
        
        # Remove the projection along the axis to get the other projection
        return v_3d - the_dot*a_hat


    def get_basis(self, a_axis):
        """
        Get 3 orthogonal and normalized vector, one which is the input axis. 
        The angle of the two other is a bit arbitrary.
        
        """
        a_axis = np.array(a_axis)
        
        a1, a2, a3 = a_axis
        
        # Find the second axis
        # The choose of the axis depends on which component of a_axis is 
        # non-null. 
        # At least one component of a_xis must be non zero
        if a1 != 0:
            b1 = -(a2+a3)/a1
            b2, b3 = b1*0+1, b1*0+1 #Not writting 1, in case the input as arrays. But anyway, I am already taking a condition on a1
        elif a2 != 0:
                b2 = -(a1+a3)/a2
                b1, b3 = b2*0+1, b2*0+1
        elif a3 != 0:
            b3 = -(a1+a2)/a3
            b1, b2 = b3*0+1, b3*0+1
        
        b_axis = np.array([b1, b2, b3], dtype=np.ndarray)
        
        # Get the third axis, which is perpendicular to the two first axis
        c_axis = self.cross_prod(a_axis, b_axis)
            
        # Normalize that
        a_hat = a_axis/self.magnitude(a_axis)
        b_hat = b_axis/self.magnitude(b_axis)
        c_hat = c_axis/self.magnitude(c_axis)
        
        return a_hat, b_hat, c_hat
        
        
        
    def get_rotation_pure_osc(self, v_cos, v_sin, a_axis):
        """
        Get the rotating component of the oscillating field, when the input 
        field as a known form: vector(A)*cos(wt) + vector(D)*sin(wt)
        (A pure oscillation, with one frequency)
        
        In this situation, we can extract the rotating component just from the 
        two vector A and D. 
        
        It returns the rotating components at the one frequency w. 
        
        Input:
            v_cos, v_sin:
                Both 3D vector. 
                Cosine and Sine component, such that the total field is 
                v_cos*cos(wt) + v_sin*sin(wt)
            a_axis:
                3D vector which is perpendicular to the plane on which we want
                the rotation. 
        
        """
        
        # Dertermine basis vector in the plane perpendicular to the rot axis
        self.a_hat, self.b_hat, self.c_hat = self.get_basis(a_axis)        
        
        # Get the projection of the vector in the rotating plane
        self.A1 = self.dot(v_cos, self.b_hat)
        self.A2 = self.dot(v_cos, self.c_hat)
        self.D1 = self.dot(v_sin, self.b_hat)
        self.D2 = self.dot(v_sin, self.c_hat)
        
        # Get each rotating component
        # See my notebook 2022-07-13 for the mathematical derivation.
        # The complex vector is assumed to be
        # clock*exp(iwt) + antic*exp(-iwt), where clock and antic are complexe cst

        # Real and imaginary part of each component
        self.clock_re = 0.5*(self.A1 + self.D2)
        self.clock_im = 0.5*(self.A2 - self.D1)
        self.antic_re = 0.5*(self.A1 - self.D2)
        self.antic_im = 0.5*(self.A2 + self.D1) 
        
        # Amplitude of these component
        self.clock_amp = (self.clock_re**2 + self.clock_im**2)**0.5
        self.antic_amp = (self.antic_re**2 + self.antic_im**2)**0.5
        # Angle, or relative phase
        # None for now, because I don`t know yet how to compute element wise
        # if the input contains arrays
        self.clock_pha = None # I don`t compute it for now
        self.antic_pha = None # I don`t compute it for now
        
        # We are done, we have done our job. I can have my Ph.D. now
        return self.clock_amp, self.clock_pha, self.antic_amp, self.antic_pha
    
    def get_rotation_integ(self, v_t, list_t, w, a_axis):
        """
        For time dependant input.
        """
        
        # Get the projection 
        # Dertermine basis vector in the plane perpendicular to the rot axis
        self.a_hat, self.b_hat, self.c_hat = self.get_basis(a_axis)    
        
        self.v1_t = self.dot(v_t, self.b_hat)
        self.v2_t = self.dot(v_t, self.c_hat)        
        
        # Basically Fourier integrate the thing. 
        
        # Complex formalism
        v_cplx = self.v1_t + 1j*self.v2_t
        
        # Clock-wise component
        # Some along the axis of time
        dt = np.mean(np.diff(list_t)) # Assuming linear spacing was used. 
        # Perform the sumation in the dumb way. 
        # Until I find a better way, with Python, to sum along the time axis
        integ = 0 
        for i, t in enumerate(list_t):
            integ += v_cplx[i]*np.exp(-1j*w*list_t[i]) 
        # Rescale properly
        A_clock = integ * dt * w/(2*np.pi)
        
        # Also check the other direction (counter-clockwise)
        integ = 0 
        for i, t in enumerate(list_t):
            integ += v_cplx[i]*np.exp(+1j*w*list_t[i]) 
        # Rescale properly
        A_count = integ * dt * w/(2*np.pi)  
        
        return A_clock, A_count
            
        
    

if __name__ == '__main__':
    # Test the class and verify that it works 
    
    from mpl_toolkits import mplot3d # Important for using the 3D projection
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('default')   
    
    # =============================================================================
    #     Check that the orthomalization works
    # =============================================================================
    
    my_v = [0.5, 0.9, -0.3]
    self = RotatingVector()
    a, b, c = self.get_basis(my_v)
    
    # Plot the three axis and check that it is perpendicular and normalized
    fig = plt.figure()
    ax = fig.gca(projection='3d')      
    ax.quiver(0, 0, 0, 
              *a, 
              color='C1', label='a_hat --> Norm=%f'%self.magnitude(a))  
    ax.quiver(0, 0, 0, 
              *b, 
              color='C2', label='b_hat --> Norm=%f'%self.magnitude(b)) 
    ax.quiver(0, 0, 0, 
              *c, 
              color='C3', label='c_hat --> Norm=%f'%self.magnitude(b))  
    ax.legend(loc='best')
    ax.set_title(  'a.b = %f'%self.dot(a, b)+
                 '\na.c = %f'%self.dot(a, c)+
                 '\nb.c = %f'%self.dot(b, c))
    
    # =============================================================================
    #     Test the rotating frame
    # =============================================================================
    # Define some vector for the problem investigated
    my_vcos = [0.7, -0.8, 0.1]
    my_vsin = [-0.9, 0.3, -0.7]
    my_axis = [-7, 0.9, 3]
    
    clock, _, antic, _ = self.get_rotation_pure_osc(my_vcos, my_vsin, my_axis)
    
    # Compare with an integral method
    w = 300*1e6 # Hz
    my_t = np.linspace(0, 2*np.pi/w, 4000)
    
    my_vt = np.zeros((3, len(my_t)))
    for i in range(3):
        my_vt[i] = (np.array(my_vcos[i])*np.cos(w*my_t)+
                    np.array(my_vsin[i])*np.sin(w*my_t))
        
#    my_vt = (np.array(my_vcos)*np.cos(w*my_t)+
#             np.array(my_vsin)*np.sin(w*my_t))
    
    a_cw, a_an = self.get_rotation_integ(my_vt, my_t, w, my_axis)
    
    print()
    print('np.abs(a_cw) = ', np.abs(a_cw))
    print('clock        = ', clock)
    print()
    print('np.abs(a_an) = ', np.abs(a_an))
    print('antic        = ', antic)
    print()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')      
    ax.quiver(0, 0, 0, 
              *my_vcos, 
              color='C1', label='vcos')  
    ax.quiver(0, 0, 0, 
              *my_vsin, 
              color='C2', label='vsin') 
    ax.quiver(0, 0, 0, 
              *my_axis, 
              color='C3', label='axis')  
    ax.legend(loc='best')
    ax.set_title('Clock = %f'%clock+
                 '\nAntic = %f'%antic)   
    
    
    
    
    
    
    
    
    
    
    
    
    