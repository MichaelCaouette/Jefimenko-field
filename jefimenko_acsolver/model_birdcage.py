# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:40:20 2022

Goal: 
    Implement a birdcage geomtery with the delay offset

@author: mcaoue2
"""


import numpy as np
from line_path import FieldLinePath

class BirdCage():
    """
    Magnetic field from Jefimenko equation from a birdcage (used in MRI) 
    
    This is the true time dependant field from E&M solution. 
    
    
    For the current distribution, I use chapter 11 of the book from 
    J. Thomas Vaughan; John R. Griffiths
    
    """
    def __init__(self, R=1, height=2, N_rung=8,
                 I_rung=1, 
                 N_smoothloop=100,
                 f=1,
                 u0 = 4*np.pi*1e-7,
                 c = 299792458 ):
        """
        R:
            Meter
            Radius of the Birdcage
            
        height:
            Meter
            Height of the birdcage (along the cylindrical axis)
        
        N_rung:
            (int)
            Number of rung of the birdcage. 
            
        I_rung:
            (Amp)
            Maximum electrical current in a rung. 
            
        N_smoothloop:
            (int)
            Number of element to have a smooth end ring (number of segment used
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
        self.R = R
        self.height = height
        self.N_rung = N_rung
        self.I_rung = I_rung
        
        self.f = f
        self.w = 2*np.pi*f
        self.u0 = u0
        self.c  = c 
        
        # Get other quantity from that
        self.I_ER = self.I_rung/(2*np.sin(np.pi/self.N_rung))
        
        
        
        # =============================================================================
        #         # Prepare the geometry and delay. 
        # =============================================================================
        self.list_wire  = [] # Will hold the various wire geometry
        self.list_delay = [] # Will hold the corresponding delay
        # Each element consist of a line path and has an associated delay.
        
        # Set the rung
        for i in range(self.N_rung):
            angle = (i/self.N_rung)*2*np.pi
            # 3D pts
            p1 = (self.R*np.cos(angle), 
                  self.R*np.sin(angle),
                  -self.height/2)
            p2 = (self.R*np.cos(angle), 
                  self.R*np.sin(angle),
                  +self.height/2)
            list_vec = [p1, p2]
            wire = FieldLinePath(list_vec,
                                 I =self.I_rung, 
                                 f =self.f,
                                 u0=self.u0,
                                 c =self.c  )
            self.list_wire.append(wire)
            # Add the delay
            self.list_delay.append(angle/self.w)
            
        # Set the end loop
        list_pts_top_ring = [] # Will store the points defining the top end ring
        list_pts_bot_ring = [] # Will store the points defining the bottom end ring
        for i in range(N_smoothloop+1): # Include the last point !
            s = i/N_smoothloop
            x = self.R*np.cos(s*2*np.pi)
            y = self.R*np.sin(s*2*np.pi)
            z_top = +self.height/2
            z_bot = -self.height/2
            list_pts_top_ring.append( (x, y, z_top) )     
            list_pts_bot_ring.append( (x, y, z_bot) ) 
        # Add these wires
        wire = FieldLinePath(list_pts_top_ring,
                             I =self.I_ER , 
                             f =self.f,
                             u0=self.u0,
                             c =self.c  )        
        self.list_wire.append(wire)
        # VERY IMPORTANT the current is minus the top one !
        wire = FieldLinePath(list_pts_bot_ring,
                             I = -self.I_ER ,  
                             f =self.f,
                             u0=self.u0,
                             c =self.c  )        
        self.list_wire.append(wire)
        # Add the delay (no delay in the end rings, 
        # see J. Thomas Vaughan; John R. Griffiths)
        self.list_delay.append(0)        
        self.list_delay.append(0)  
        
        # Note the number of element that makes our field
        self.N_wire = len(self.list_wire)
    
    def get_field_from_spatial(self, vcos, vsin, t):
        """
        Convenient method to get the field when we already have the spatial 
        components. 
        """
        phase = self.w*t
        return vcos*np.cos(phase) + vsin*np.sin(phase)
        
    def get_field_spatial(self, x, y, z):
        """
        Get only the spatial contribution of the field (ie the two vector:
            the cos and sin term)
         Useful for studying the rotating components   
         
         
         return self.vcos, self.vsin
        """
        
        for i in range(self.N_wire):
            # Important to add the delay
            wire = self.list_wire[i]
            out =  wire.get_field_spatial(x, y, z)
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
        
        
    def get_field(self, x, y, z, t):
        """
        Get the magnetic field a time t and position x,y,z. 
        
        """
        
        # We will add the field from each element plus the delays
        
        for i in range(self.N_wire):
            # Important to add the delay
            wire = self.list_wire[i]
            out =  wire.get_field(x, y, z, 
                                  t-self.list_delay[i])
            if i == 0:
                self.Bx, self.By, self.Bz = out
            else:
                self.Bx += out[0]
                self.By += out[1]
                self.Bz += out[2]
        # Done !
        return self.Bx, self.By, self.Bz
            
            
            
        
        
        
        
if __name__ == '__main__':
    # Check that the geometry is right        
        
    # from viewer_3D import plot_3D_cool
    from viewer_wire_field_3D import Show3DVectorField    
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('default')    
      
    
    
    self = BirdCage(R=0.3, 
                    height=0.5, 
                    N_rung=16,
                    I_rung=2.3, 
                    N_smoothloop=50,
                    f=1*297.2*1e6)
    
# =============================================================================
#     # Check if the field rotates at the center
# =============================================================================
    list_t_probe = np.linspace(0, 1, 27)/self.f 
    (list_Bx, 
     list_By, 
     list_Bz) = self.get_field(0.2*self.R, 
                               0, 
                               0, 
                               list_t_probe)     
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
    t_probe = 0.50/self.f # Fix the time to probe
    
    # A line cute into the cage
    # Staight cut along the z axis
    list_probe_x = np.linspace(-1.5*self.R, 1.5*self.R, 87)
    list_probe_y = 0.1*self.R      + 0*list_probe_x
    list_probe_z = 0*self.height + 0*list_probe_x
    
    list_Bz = []
    list_Bx = []
    list_By = []
    print('Probing the fiekld into space...')
    (list_Bx, 
     list_By, 
     list_Bz) = self.get_field(list_probe_x, 
                               list_probe_y, 
                               list_probe_z,
                               t_probe) 
    print('Done')
    
    
    # =============================================================================
    # Check this out
    # =============================================================================
    
    # For cycling into any cmap
    import matplotlib
    # cmap = matplotlib.cm.get_cmap('rainbow')
    cmap = matplotlib.cm.get_cmap('brg')
    # Check up that it works
    rgba = cmap(0.5)
    print(rgba) # (0.99807766255210428, 0.99923106502084169, 0.74602077638401709, 1.0)

    vv = Show3DVectorField()
    
    # Show all the elements
    for ii, wire in enumerate(self.list_wire):
        N_pts = len(wire.list_vec)
        xline, yline, zline = np.zeros((3, N_pts))
        for i in range(N_pts):
            xline[i], yline[i], zline[i] = wire.list_vec[i]        
        
        # Choose a color that matches the delay
        value = (self.list_delay[ii] - min(self.list_delay)) / (max(self.list_delay) - min(self.list_delay) )
        my_color = cmap(value)
        vv.add_path(xline, yline, zline, color=my_color, 
                    label='Current Path', show_corner=True)
        
    # Show the field 
    vv.add_vector_field(list_probe_x, list_probe_y, list_probe_z, 
                        list_Bx, list_By, list_Bz,
                        scale=0.4*self.R, color='C1', label='B field')
        
    
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
    # plt.ylabel('Bz (SI)')    
    # plt.plot(list_probe_axis, list_Bz, label='',  linewidth=4)
    # plt.xlabel('Along line (arbr. unit) ')
    # plt.legend()      


    plt.ylabel('|B| (SI)')    
    plt.plot(list_probe_axis, 
             (list_Bx**2+list_By**2+list_Bz**2)**0.5, 
             label='',  linewidth=4)
    plt.xlabel('Along line (arbr. unit) ')
    plt.legend()              
        
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        