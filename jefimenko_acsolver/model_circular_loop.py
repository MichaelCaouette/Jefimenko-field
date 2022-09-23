# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:31:13 2022

@author: client
"""

import numpy as np

class FieldLoop():
    """
    Magnetic field from Jefimenko equation from a loop coil. 
    It is the analytical application of the integral formula to compute the 
    field in each point in space and time from the loop coil.
    
    This is the true time dependant field from E&M solution. 
    
    """
    def __init__(self, R=1, I=1, f=1,
                 u0 = 4*np.pi*1e-7,
                 c = 299792458 ):
        """
        R:
            Meter
            Radius of the loop coil. 
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
        self.R = R
        self.I = I
        self.w = 2*np.pi*f
        self.u0 = u0
        self.c  = c 

    def r_rel(self, x, y, z, s):
        """
        Relative position between the loop and the external. 
        The loop position is cylindrical. 
        The external postion is cartesian. 
            
        """
        x_rel = x - self.R*np.cos(s)
        y_rel = y - self.R*np.sin(s)
        z_rel = z # The loop lies on the z=0 plane
        
        return np.array([x_rel, y_rel, z_rel], dtype=np.ndarray)

    def integrant(self, x, y, z, t, s):
        """
        Integrant of the Jefimenko eqn at position 's' on the current loop. 
        For this circular loop, 's' is the angle from the x-axis and the loop 
        element (equivalent to 'theta' in the usual circle definition).
        
        Note that the overal sclaing factor will be given after computing the 
        integral, for saving calculation
        """
        # Unit element of lenght along the wire, point in the current direction
        v = [- np.sin(s), np.cos(s), 0] # Without the factor R, to save computation
        # Relative position between the loop element and external reference
        u = self.r_rel(x, y, z, s)

        # Cross product between the lenght element along the loop and the 
        # relative position. 
        # Simpler to do it explicitely and element wise, because using 
        # np.cross() makes it hard to interprete multiple axis in x,y,z,t
        cross_x = v[1]*u[2] - v[2]*u[1]
        cross_y = v[2]*u[0] - v[0]*u[2]
        cross_z = v[0]*u[1] - v[1]*u[0]
        cross = np.array([cross_x, cross_y, cross_z])
 
        # Amplitude of the relative position. 
        # Again use the elementwise explicit
        abs_rel = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])**0.5
        
        # Retarded time
        tr = t - abs_rel / self.c
        phase = np.float64(self.w*tr)
        # terms
        term1 =  np.cos(phase)       /abs_rel**3
        term2 = -np.sin(phase)*self.w/abs_rel**2/self.c
        term = term1 + term2
        
        # Make a 3D array for each spatial coordinate (Bx, By, Bz)
        # Doint it this way, explicitely, ensure that there will not 
        # be conflicts with the shape dimension if the input x, y or z are 
        # arrays
        vec = np.array([term*cross[0], term*cross[1], term*cross[2]])
        # Any other prefactor will be added after the numerical integration
        return vec
    
    def trap_integrate(self, x, y, z, t, N=100):
        """
        Compute the integral, numerical way. 
        
        Jeffimenko eqn from Griffith, p450 eqn 10.38
        
        N:
            # Number of integral element. 
            The larger this number, the more accurate is the integral. 
        """
        # Integrant element is a small fraction of the loop Radius
        self.ds = 2*np.pi/N
        # Super basic trapezoidal method
        self.integ = 0
        for i in range( N ):
            # We average the height of the function on each end
            if i == 0:                
                fa = self.integrant(x, y, z, t,  i   *self.ds)
                fb = self.integrant(x, y, z, t, (i+1)*self.ds)
            else:
                # Avoid double evaluation by taking the previous value
                fa = fb
                fb = self.integrant(x, y, z, t, (i+1)*self.ds)   
            self.integ += 0.5*(fa+fb)
        self.overall_cst = self.ds*self.u0*self.I*self.R/(4*np.pi)
        return self.overall_cst*self.integ
    
    def along_z(self, z, t):
        """
        Analytical solution on the z axis (x,y = 0, 0).
        In this situation, the integral is straigthfoward to solve. 
        """
        # Distance from the center of the loop
        abs_rel = np.sqrt(self.R*self.R + z*z)
        
        # Retarded time
        tz = t - abs_rel/self.c
        
        # Terms of the solutipon
        term1 =  np.cos(self.w*tz) / abs_rel**3
        term2 = -np.sin(self.w*tz) * self.w / abs_rel**2 / self.c
        
        # Overall constant
        cst = 0.5*self.u0*self.I*self.R**2
        
        # B is only along z
        Bz = cst*(term1 + term2)
        return Bz
        
    def dipole(self, x, y, z, t):
        """
        Dipolar radiation of the loop. 
        This is for comparing the numerical solution into the limits. 
        The solution is from my Griffiths book, eqn 11.37 p.476.  
        
        """
        # Overall prefactor
        sqrt_xy = np.sqrt(x*x + y*y)
        r = np.sqrt(sqrt_xy*sqrt_xy + z*z)
              
        cst = -self.u0*self.R**2*self.I*self.w**2/(4*self.c**2)
        pre_fac_coord = sqrt_xy / (r*r)*np.cos(self.w*(t - r/self.c)) 
        pre_fac = cst * pre_fac_coord
        
        # Unit vector in the theta direction (spherical coordinate)
        vx = z*x / (r * sqrt_xy)
        vy = z*y / (r * sqrt_xy)
        vz = -sqrt_xy / r # Note the minus sign ! It always point in the negative z
        
        # The B field
        Bx = pre_fac * vx
        By = pre_fac * vy
        Bz = pre_fac * vz
        
        return np.array([Bx, By, Bz])       
    
if __name__ == '__main__':


    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('default')    
    
    # Compute and show some field   
    I = 2.3
    R = 0.015 # m, radius of the loop
    f = 1*297.2*1e6 #62*1e9 #297.2*1e9 # Rad/s Frequency of oscillating current. 
    self = FieldLoop(R=R, f=f, I=I)
    # Compare the wavelenght with respect to the loop radius
    wvl = self.c/f
    print('R = %f   and wavelenght = %f'%(R, wvl))
    print('R / wavelenght = %f'%(R/wvl))
    
# =============================================================================
#     Sanity check with the dipole solution
    # Approximation that must be fullfilled:
    # R << c/w << r, where r is where we are looking
# =============================================================================
    # Check a single point in space far from the source
    x, y, z = (10*R, 10*R, 10*R)
    list_t = np.linspace(0, 4/f, 100)
    
    list_Bz = []
    list_Bx = []
    list_By = []
    print('Calculating all times at single pts in space...')
    for i, t in enumerate(list_t):
#        print('Calculating time %d/%d'%(i, len(list_t)))
        Bx, By, Bz = self.trap_integrate(x, y, z, t, N=150)
        list_Bz.append(Bz)
        list_Bx.append(Bx)
        list_By.append(By)   
    print('Done')
    print('Calculating the dipole solution...')
    list_dipBx, list_dipBy, list_dipBz = self.dipole(x, y, z, list_t)
    print('Done')
      

    plt.figure(tight_layout=True)
    plt.title('Check dipolar soln')
    plt.plot(list_t, list_Bz, label='Bz numerical',  linewidth=4)
    plt.plot(list_t, list_dipBz, '--', label='Bz dipole', linewidth=4)
    plt.xlabel('t (s) ')
    plt.ylabel('B (SI)')
    plt.legend()        
# =============================================================================
# Dipole again, but vary position into space at fixed time
# =============================================================================
  
    list_z = np.linspace(50*R, 1000*R, 1000)
    x, y = 100*R, 100*R
    t = 0.2/f # Fix the time
    list_Bz = []
    list_Bx = []
    list_By = []
    print('Calculating along a line in space')
    list_Bx, list_By, list_Bz = self.trap_integrate(x, y, list_z, t) 
    print('Done')
    print('Calculating the dipole solution...')
    # Note that the dipole field is zero on the z axis. 
    list_dipBx, list_dipBy, list_dipBz = self.dipole(x, y, list_z, t)
    print('Done')
    plt.figure(tight_layout=True)
    
    plt.subplot(311)
    plt.title('x, y = %.1f, %.1f radius'%(x/R, y/R))
    plt.ylabel('Bx (SI)')    
    plt.plot(list_z*100, list_Bx, label='Numerical',  linewidth=4)
    plt.plot(list_z*100, list_dipBx, '--', label='Dipole', linewidth=4)
    plt.subplot(312)
    plt.ylabel('By (SI)')    
    plt.plot(list_z*100, list_By, label='Numerical',  linewidth=4)
    plt.plot(list_z*100, list_dipBy, '--', label='Dipole', linewidth=4)
    plt.subplot(313)
    plt.ylabel('Bz (SI)')    
    plt.plot(list_z*100, list_Bz, label='Numerical',  linewidth=4)
    plt.plot(list_z*100, list_dipBz, '--', label='Dipole', linewidth=4)
    plt.xlabel('z (cm) ')
    plt.legend()      
 
    
## =============================================================================
## Sanity check the known solution along the z axis
## =============================================================================
    list_z = np.linspace(-3*R, 10*R, 300)
    x, y = 0, 0
    t = 0.2/f # Fix the time
    list_Bz = []
    list_Bx = []
    list_By = []
    print('Calculating along the z axis...')
    list_Bx, list_By, list_Bz = self.trap_integrate(x, y, list_z, t, N=3) 
    print('Done')
    print('Calculating the exact solution...')
    list_exaBz = self.along_z(list_z, t)
    print('Done')
    print('Calculating the dipole solution...')
    # Note that the dipole field is zero on the z axis. 
    list_dipBx, list_dipBy, list_dipBz = self.dipole(x, y, list_z, t)
    print('Done')
    plt.figure(tight_layout=True)
    plt.title('Sanity check along z axis, fixed time')
    plt.plot(list_z*100, list_Bz, label='Numerical',  linewidth=4)
    plt.plot(list_z*100, list_exaBz, '--', label='Exact soln', linewidth=4)
    plt.plot(list_z*100, list_dipBz, '--', label='Dipole', linewidth=4)
    plt.xlabel('z (cm) ')
    plt.ylabel('Bz (SI)')
    plt.legend()      
 


























