# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:18:57 2022

Goal:
    Define a function por class to visualize the geometry of the system. 

@author: client
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # Important for using the 3D projection



    
# Show that !
# Show that !
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    
    This function it taken from the internet

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
    
    
def plot_3D_cool(xpath, ypath, zpath,
                 list_pos_x, list_pos_y, list_pos_z,
                 list_Bx, list_By, list_Bz, scale=1):
    """
    Description coming soon !
    
    TODO: Make this a class. And optionto add stuff in the plot
    And better organize the input, make verything clear etc. 
    Include the class
    
    For the class: a method to add a line path, a vector field, with label. Etc. 
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot3D(xpath, ypath, zpath, 'red', label='Wire Path')  
    
    # Add the vector field
    # Scale the b field to be easily seen on the geometry
    list_norm = (list_Bx**2 + 
                 list_By**2 +
                 list_Bz**2)**0.5
    # list_norm = (list_Bx_loop**2 + 
    #              list_By_loop**2 +
    #              list_Bz_loop**2)**0.5
    scaling_factor = scale/np.max(list_norm)
        
    ax.quiver(list_pos_x, list_pos_y, list_pos_z, 
              list_Bx*scaling_factor, 
              list_By*scaling_factor, 
              list_Bz*scaling_factor, color='C0', label='Line-Path') 
    ax.legend(loc='best')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    set_axes_equal(ax)
    plt.show()
    
 
    
def  list_pts_to_each_lines(list_pts):
    N_pts = len(list_pts)
    
    xline, yline, zline = np.zeros((3, N_pts))
    for i in range(N_pts):
        xline[i], yline[i], zline[i] = list_pts[i]   
    
    return xline, yline, zline
        
        
    
class Show3DVectorField():
    """
    This class shows the vector field and some path trajectory.
    """
    def __init__(self):
        
        # Initiate the viewer
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')    

        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_zlabel('z (m)')
        
        # Variables for setting the limits of the plot
        self.min_x, self.max_x = 0,0
        self.min_y, self.max_y = 0,0
        self.min_z, self.max_z = 0,0
        
    def _adjust_new_bound(self, 
                          min_x, max_x,
                          min_y, max_y,
                          min_z, max_z):
        """
        Check the extrem and ajust the limit of the plot accordingly. 

        Returns
        -------
        None.

        """
        
        print('THis can just grow. What About initial plot ?')
        
        # Reajsut the new limits
        # Do it for x
        if min_x < self.min_x:
            self.min_x = min_x
        if max_x > self.max_x:
            self.max_x = max_x  
        # Do it for y
        if min_y < self.min_y:
            self.min_y = min_y
        if max_y > self.max_y:
            self.max_y = max_y 
        # Do it for z
        if min_z < self.min_z:
            self.min_z = min_z
        if max_z > self.max_z:
            self.max_z = max_z        
        
        # Now set the axis equal
        x_range = abs(self.max_x - self.min_x)
        x_middle = (self.max_x + self.min_x)/2
        y_range = abs(self.max_y - self.min_y)
        y_middle = (self.max_y + self.min_y)/2
        z_range = abs(self.max_z - self.min_z)
        z_middle = (self.max_z + self.min_z)/2
    
        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
    
        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])   
        
    def add_vector_field(self, 
                         list_pos_x, list_pos_y, list_pos_z,
                         list_Vx, list_Vy, list_Vz,
                         scale=1, label='Cool vector', color='C0'):
        """
        DESCRIPTION COMMING SOON

        Parameters
        ----------
        list_pos_x : TYPE
            DESCRIPTION.
        list_pos_y : TYPE
            DESCRIPTION.
        list_pos_z : TYPE
            DESCRIPTION.
        list_Vx : TYPE
            DESCRIPTION.
        list_Vy : TYPE
            DESCRIPTION.
        list_Vz : TYPE
            DESCRIPTION.
        scale : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        # Numpy array all that
        list_Vx = np.array(list_Vx)
        list_Vy = np.array(list_Vy)
        list_Vz = np.array(list_Vz)
        
        list_norm = (list_Vx**2 + 
                     list_Vy**2 +
                     list_Vz**2)**0.5
        
        scaling_factor = scale/np.max(list_norm)
            
        self.ax.quiver(list_pos_x, list_pos_y, list_pos_z, 
                       list_Vx*scaling_factor, 
                       list_Vy*scaling_factor, 
                       list_Vz*scaling_factor, color=color, label=label)    
           
        # Usual stuff
        set_axes_equal(self.ax)
        self.ax.legend(loc='best')
        plt.show()
        
    def add_path(self,
                 xpath, ypath, zpath, 
                 label='', color='red', show_corner=False):
        """
        DESCRIPTION COMMING SOON

        Parameters
        ----------
        xpath : TYPE
            DESCRIPTION.
        ypath : TYPE
            DESCRIPTION.
        zpath : TYPE
            DESCRIPTION.
        label : TYPE, optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        None.

        """
        
        if label == '':
            self.ax.plot3D(xpath, ypath, zpath, color=color)
        else:
            self.ax.plot3D(xpath, ypath, zpath, color=color, label=label)  
        
        if show_corner:
            self.ax.plot3D(xpath, ypath, zpath, '.', color='k')  
            
        
        # Usual stuff
        # set_axes_equal(self.ax)
        self._adjust_new_bound(np.max(xpath), np.min(xpath),
                               np.max(ypath), np.min(ypath),
                               np.max(zpath), np.min(zpath))
        self.ax.legend(loc='best')
        plt.show()             
        
            
        
    def add_list_wire(self, list_wire, list_color=[], 
                      str_cmap='brg', label='', color_title='',
                      show_corner=False):
        """
        Add a list of wire (this is specific to the application of B field from
                            line of current)
        
        If there is a non-nul list_color the color will be adjusted with
        the delay.

        Parameters
        ----------
        str_cmap:
            String
            Can be 'rainbow', or anything cool

        Returns
        -------
        None.

        """
        
        
        
        
        # cmap = matplotlib.cm.get_cmap('rainbow')
        cmap = matplotlib.cm.get_cmap(str_cmap)
               
        for ii, wire in enumerate(list_wire):
            # Transform the data properly
            xline, yline, zline = list_pts_to_each_lines(wire.list_vec)        
            
            # Choose a color that matches the delay
            # The linear map works only for at least two elements in the list
            if len(list_color) != 1:
                value = (( list_color[ii] - min(list_color) ) / 
                         (max(list_color) - min(list_color) )  )
            else:
                value = list_color[0] 
            my_color = cmap(value)
            
            # Just put the label once
            if ii == 0:
                self.add_path(xline, yline, zline, color=my_color, 
                             label=label, show_corner=show_corner) 
            else:
                self.add_path(xline, yline, zline, color=my_color, 
                                 label='', show_corner=show_corner)     
            
        # Add a cool colorbar
        # Check https://matplotlib.org/stable/api/cm_api.html#matplotlib.cm.ScalarMappable
        norm = matplotlib.colors.Normalize(vmin=min(list_color), 
                                           vmax=max(list_color), 
                                           clip=False)
        # Some Version issue
        try:
            self.fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                              ax=self.ax,
                              label=color_title)
        except:
            # Solution found on https://stackoverflow.com/questions/63754046/when-adding-colorbar-for-a-matplotlib-tricontourf-typeerror-you-must-first-se
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            self.fig.colorbar(sm,
                              ax=self.ax,
                              label=color_title)            
        # Done !
        return 
        
        
        
        
        
        
        
    
    
    
    
    
    
