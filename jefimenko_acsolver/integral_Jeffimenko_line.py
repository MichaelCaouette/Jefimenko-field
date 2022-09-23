# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:28:35 2022

Goal:
    Calculate and study the Jeffimenko integral for a straghit line, with the 
    time dependency taken out (assuming the current source is sinuosidal)

@author: mcaoue2
"""

import numpy as np

_debug_enabled           = False
def _debug(*a):
    if _debug_enabled: 
        s = []
        for x in a: s.append(str(x))
        print(', '.join(s))
        
        
class IntegralJeffimenko():
    """
    Integral to get the magnetic field from the the Jefimenko equation, when
    the current source is along a straight line, and oscillates at a fixed 
    frequency w. 
    
    This class concern just the integral that I was not able to reduce further. 
    There might be better simplification (pole analysis ??) but I haven't 
    find any yet. 
    
    (see notebook 2022-05-23 for more details)
    
    """
    def __init__(self, 
                 w,
                 c):     
        """
        

        Parameters
        ----------
        w : float
            Oscillation frequency (rad/sec) NOT Hertz !!!
            
        c : float
            Speed of light (m/sec)

        Returns
        -------
        None.

        """
        _debug('IntegralJeffimenko.__init__')
        
        self.w = w
        self.c = c
        
    def _phase_relr(self, x1, lenght, y1, s):
        """
        Get the phase and relative distance for the integrant.

        Parameters
        ----------
        x1 : TYPE
            DESCRIPTION.
        lenght : TYPE
            DESCRIPTION.
        y1 : TYPE
            DESCRIPTION.
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        phase : TYPE
            DESCRIPTION.
        rel_r : TYPE
            DESCRIPTION.

        """
        
        xx = (x1+lenght*s)
        rel_r = ( y1*y1 + xx*xx )**0.5
        phase = np.float64(rel_r*self.w/self.c)
        
        return phase, rel_r
        
        
    def _integrant_cos_sin(self, x1, lenght, y1, s):
        """
        Integrant for the integral of the cos and sin terms. 
        
        s is the integration constant (meant to go from 0 to 1, unitless)
        
        The other inputs are coordinate variable in the convenient coord system
        """
        _debug('IntegralJeffimenko._integrant_cos_sin')
               
               
        # Compute some variables once, because they come over and over
        self.phase, rel_r = self._phase_relr(x1, lenght, y1, s)
        sin_phase = np.sin(self.phase)
        cos_phase = np.cos(self.phase)
        self.r3 = np.float64(rel_r*rel_r*rel_r)
        
        # We have everything to compute each term
        term_cos = (cos_phase + self.phase*sin_phase)/self.r3
        term_sin = (sin_phase - self.phase*cos_phase)/self.r3
        
        return term_cos, term_sin
        
    def get_integral_terms(self, x1, lenght, y1, N_integ):
        """
        Useful function to get only the spatial terms in the oscillating 
        solution. 
        (x1, lenght, y1) are the coordinate at which we want to compute these 
        terms, in the 'convenient' coord system. 
        If you start by a general cartesian coord system, transform the 
        coordinate with the method 'coordinate_mapping'.
        """ 
        _debug('IntegralJeffimenko.get_integral_terms')
        
        self.N_integ = np.ceil(N_integ)
        
        # The numerical method will be first order. Adding the area of bins. 
        self.ds = 1/self.N_integ # The parameter ranges from 0 to 1, so the width is 1/N
        self.integ_cos, self.integ_sin = 0, 0
        for i in range(int(self.N_integ)):
            # We average the height of the function on each end of the bin
            s2 = (i+1)*self.ds # position of the right edge of the bin
            if i == 0:             
                # The left edge of the first bin is s=0
                fa_cos, fa_sin = self._integrant_cos_sin(x1, lenght, y1, 0)
                fb_cos, fb_sin = self._integrant_cos_sin(x1, lenght, y1, s2)
            else:
                # Avoid double evaluation by taking the previous value
                # Because the bins share the left side with the right side of 
                # the previous bin. 
                fa_cos, fa_sin = np.copy(fb_cos), np.copy(fb_sin)
                # fa_cos, fa_sin = self._integrant_cos_sin(x1, lenght, y1, i*self.ds)  
                fb_cos, fb_sin = self._integrant_cos_sin(x1, lenght, y1, s2)  
            # Area of the bin, without the proper scaling.
            # The scaling (0.5*bin_width) will be added at the end, to save 
            # precious computation. 
            self.integ_cos += (fa_cos + fb_cos) 
            self.integ_sin += (fa_sin + fb_sin)  
        # We are done \m/
        # Just not forget the integration width and scaling
        self.bin_width_scale = 0.5*self.ds
        return (self.bin_width_scale *self.integ_cos, 
                self.bin_width_scale *self.integ_sin )
    
    def optimal_discretization(self,  x1, lenght, y1):
        """
        estimate the best number of discret of points to take for the integration. 
        It is based on the oscillation of the integral. 
        """
        _debug('IntegralJeffimenko.optimal_discretization')
        
        # Check how the phase evolves
        phi0, _ = self._phase_relr(x1, lenght, y1, 0)
        phi1, _ = self._phase_relr(x1, lenght, y1, 1)
        # Difference in phase
        dphi = phi1 - phi0
        # Check how many warp-around it is. 
        N_warp = np.ceil(np.abs(dphi / (2*np.pi)))
        # Just make sure to not have zero
        N_warp += 1
        
        # The optimal number of discret point is 20 times that. 
        # I choose 20 because it seems right for a sine, but maybe there is a 
        # better number. 
        return 20*N_warp
    
    
if __name__ == '__main__':
    _debug_enabled = True
    
    
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('default')        
    
    
    x_start  = 1
    L        = 30
    y_height =  5
    
    self = IntegralJeffimenko(w=297.2*1e6, 
                              c=299792458)
    
    s_smooth = np.linspace(0, 1, 1000)
    vc_smooth, vs_smooth = self._integrant_cos_sin(x_start, L, y_height, s_smooth)
    
    
    plt.figure(tight_layout=True)
    plt.plot(s_smooth, vc_smooth)
    plt.plot(s_smooth, vs_smooth)
    plt.xlabel('s (integration variable)')
    plt.ylabel('Integrant')
    
    
    
    
    N_opt = self.optimal_discretization(x_start, L, y_height)
    print('N_opt = ', N_opt)
    
    # Make some integrations with various discretization
    list_Ndiscr = [0.5*N_opt, N_opt, 10*N_opt, 100*N_opt]

    fig, (ax_c, ax_s) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                   tight_layout=True)
    
    for Ndiscr in list_Ndiscr:
        # Integrate
        intc, ints = self.get_integral_terms(x_start, L, y_height,
                                         N_integ=Ndiscr)
        # Plot that
        s_discr = np.linspace(0, 1, int(Ndiscr))
        vc_discr, vs_discr = self._integrant_cos_sin(x_start, L, y_height, 
                                                     s_discr)        

        ax_c.plot(s_discr, vc_discr, label='Ndiscr=%d --> Int=%f'%(Ndiscr, intc*1e5))
        ax_s.plot(s_discr, vs_discr, label='Ndiscr=%d --> Int=%f'%(Ndiscr, ints*1e5))
       
    # Make the plot beautiful
    ax_c.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_s.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_s.set_xlabel('s (integration variable)')
    ax_c.set_ylabel('Integrant Cos')    
    ax_s.set_ylabel('Integrant Sin') 
    ax_c.set_title('Verification of the integral terms VS the discretization.'+
                   '\nHere, N_opt = %d'%N_opt)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    