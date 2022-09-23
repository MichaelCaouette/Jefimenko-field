# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:29:22 2022

Goal:
    Have an image viewer. 
    Adapted to visualize a list of Matrix (NxM)
@author: client
"""
# The super cool 2D mapper

from gui_map_2D import Map2D


import numpy as np
from spinmob import egg
import traceback

#The following is for plotting matplotlib object in the gui
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import *
from PyQt5.QtGui import *

_p = traceback.print_last #Very usefull command to use for getting the last-not-printed error


# Debug stuff.
_debug_enabled     = False

def _debug(*a):
    if _debug_enabled: 
        s = []
        for x in a: s.append(str(x))
        print(', '.join(s))



"""
The following function is no longer used, but I keep it there because it may 
found to be useful one day.
"""
def get_minmax(self):
    """
    Get the minimum and maximum value of the signal among all the slices. 
    
    Note:
        Require the slices to be already 'loaded'.
    """
    _debug('Grid:get_minmax')

    # Updates the variables that depends on user-defined attributes
    self.N_slice = len(self.list_image)   
    self.N_plots = self.n_rows*self.n_cols # How many plot in total

    # Compute the overall max and min. 
    # Check all element that we will plot
    for j in range(self.N_plots):
        i_to_show = j*int(self.N_slice/self.N_plots) # Indice of the slice to show.
        z = self.list_image[i_to_show]      
        # Note the maxima in this file
        z_min = np.min(z)
        z_max = np.max(z)
        # Readjust the bounds if it worth
        if j == 0:
            best_min = z_min
            best_max = z_max
        else:
            if z_min < best_min:
                best_min = z_min
            if z_max > best_max:
                best_max = z_max
    return best_min, best_max

    
class SliceBySlice(egg.gui.Window):
    """
    GUI to see slice by slice
    """
    def __init__(self):
        """
        Hello
        """
        _debug('SliceBySlice:__init__')

        #Run the basic stuff for the initialization
        egg.gui.Window.__init__(self)
        
        # We use the map2D to image the slice
        self.map = Map2D()
        self.place_object(self.map) 
        
        # I will track the maximum index with this attribute instead of using bound
        self.slice_index_max = 99
        
        # Add elements in the tree dict
        self.map.treeDic_settings.add_parameter('slice', 0, 
                                                type='int', step=1, 
                                                bounds=[0,None],
                                                tip='Which slice to show')  
        self.map.treeDic_settings.connect_signal_changed('slice', 
                                                         self._slice_changed)
        
    def _slice_changed(self):
        """
        Update the slice shown
        """
        self.slice_index = self.map.treeDic_settings['slice']
        _debug('SliceBySlice: _slice_changed', self.slice_index)
        
        # Change the index only if we are still within the bounds
        if self.slice_index <= self.slice_index_max:
            self.prev_slice_index = self.slice_index
            self._update_plot()
        else:
            # Do not change the index (keep the previous value)
            self.slice_index = self.prev_slice_index
        
               

    def _update_plot(self):
        """
        Function that updates the plot with the current attribute. 
        It is important that the function do not depend on any parameters, 
        because it will be called after update single or few attributes at a 
        time. 
        """
        _debug('SliceBySlice:_update_plot')

        # Updates the variables that depends on user-defined attributes
        self.slice = self.list_image[self.slice_index]
        self.map.set_data(self.slice)
        
        
    def set_list_image(self, 
                       list_image):
        """
        Send a list to the plot and update all corresponding attribute
        
        list_image:
            list of slices to plot
        """
        _debug('SliceBySlice:set_list_image')
        
        self.list_image = list_image 
        
        # Update attributes that depends on the list of slice
        
        self.N_slice = len(self.list_image)  
        self.slice_index_max = self.N_slice - 1
        self.slice_index  = int(self.slice_index_max/2)     
        self.prev_slice_index = self.slice_index     
        # Show the plot with all these attributes
        self._update_plot()     
        
        
        
class Grid(egg.gui.Window):
    """
    A GUI to view N by M slices 
    """
    
    def __init__(self): 
        """
        """
        _debug('Grid:__init__')
        _debug('Everyone thinks of changing the world, but no one thinks of changing himself. – Leo Tolstoy')

        #Run the basic stuff for the initialization
        egg.gui.Window.__init__(self)
        
        #Run the basic stuff for the initialization
        egg.gui.Window.__init__(self)
        
        # We use the map2D to image the slice
        self.map = Map2D()
        self.place_object(self.map) 
        
        # Add elements in the tree dict and connect the corresponding signals
        self.map.treeDic_settings.add_parameter('nrows', 2, 
                                                type='int', step=1, 
                                                bounds=[1,None],
                                                tip='Number of row to show')  
        self.map.treeDic_settings.connect_signal_changed('nrows', 
                                                         self._update_plot)
        self.map.treeDic_settings.add_parameter('ncols', 2, 
                                                type='int', step=1, 
                                                bounds=[1,None],
                                                tip='Number of column to show')  
        self.map.treeDic_settings.connect_signal_changed('ncols', 
                                                         self._update_plot) 
        self.map.treeDic_settings.add_parameter('Normalize_all', False, 
                                                type='bool',
                                                tip='Normalize each image to range from 0 to 1.')  
        self.map.treeDic_settings.connect_signal_changed('Normalize_all', self._normalize_all_changed)
        
        # Note down the list of text to show the slice index 
        # Necessary for removing the preivous text
        self.list_textitem_slice = []
        
    def _clean_textitem_slice(self):
        """
        Remove all existing text
        """
        _debug('Grid:_clean_textitem_slice')
        for textitem in self.list_textitem_slice:
            self.map.plot_image.removeItem(textitem)
        # Reinitiate the item list
        self.list_textitem_slice = []
    
    def _normalize_all_changed(self):
        """
        Normalize or rmeove the normalization to all image

        Returns
        -------
        None.

        """
        # We update the plot according to the wish of the user.
        value = self.map.treeDic_settings['Normalize_all']
        
        if value:
            # Create a normalized list of image
            self.list_image_normal = []
            self.N_slice = len(self.list_image)
            for i in range(self.N_slice):
                # The max and min of the normalization will be from 0 to 1
                maximum = np.max(self.list_image[i])
                minimum = np.min(self.list_image[i])
                the_range = maximum - minimum                 
                image_norm = (self.list_image[i]-minimum)/the_range
                self.list_image_normal.append(image_norm)
            # Assigne this to be the new image
            self.list_image = self.list_image_normal
        else:
            # Remove the normalization
            self.list_image = self.list_image_unnormal 
            
        # Update all that
        self._update_plot()
            
    def _choose_image(self):
        """
        Choose the index of the image to show, based on the index of the loop. 
        This function have the purpose of making the label and slice shown 
        consistent, because they are made in separated loop
        

        Returns
        -------
        None.

        """
        
        # Make the image goes from 0 to the last existing slice
        return int(self.j*(self.N_slice-1)/(self.N_plots-1)) 
    
    def _update_plot(self):
        """
        Function that updates the plot with the current attribute. 
        It is important that the function do not depend on any parameters, 
        because it will be called after update single or few attributes at a 
        time. 
        """
        _debug('Grid:_update_plot')
 
        # Generate the grid map by concatenating the list to show  
        # TO make some test in the console, test this code for example:
#        m1 = np.array([[4,5],[-1,2]])
#        m2 = 2*m1
#        mc1 = np.concatenate((m1, m2), axis=0)
#        mc2 = 12*mc1
#        print(np.concatenate((mc1, mc2), axis=1))     
        # Get all relevant info
        self.nrows = self.map.treeDic_settings['nrows'] 
        self.ncols = self.map.treeDic_settings['ncols']
        self.N_slice = len(self.list_image)   
        self.N_plots = self.nrows*self.ncols # How many plot in total
        
        # A checkup before going further. 
        if self.N_plots > self.N_slice:
            print("Warning in 'Grid:_update_plot'. Too many plot for the number of images (%d)"%self.N_slice)
            # Don't even try lol 
            return

        # =============================================================================
        #         Create the grid 
        # =============================================================================
        # The elements of the grid are floats
        self.map_grid = np.array([], dtype=float)
        self.j = 0 # To determine which n'th slice to choose
        for row in range(self.nrows):
            self.one_row = np.array([], dtype=float)
            for col in range(self.ncols):
                # Indice of the slice to show.
                self.i_to_show = self._choose_image()
                self.slice = self.list_image[self.i_to_show]
                # Constructe the row by concatening slices
                if self.j%self.ncols == 0:
                    # The first element must be initiated for subsequent concatenation
                    self.one_row = self.slice
                else:
                    self.one_row = np.concatenate((self.one_row, self.slice), axis=0)
                # Update the number of slice that we have (starting from 0)               
                self.j += 1
            # At this point we built one row of the full grid    
            # Add the row in the total map
            if self.j == self.ncols: 
                # If we are at the first row
                # The first element must be initiated for subsequent concatenation
                self.map_grid = self.one_row
            else:
                self.map_grid =  np.concatenate((self.map_grid, self.one_row), axis=1)
        # At this point we constructed the whole grid     
        
        # =============================================================================
        #         Add the grid to the viewer
        # =============================================================================
        # The whole map is what we show
        self.map.set_data(self.map_grid)
 
        # =============================================================================
        #         # Let's add the slice indices of label
        # =============================================================================
        # First clean up what exists
        self._clean_textitem_slice()
        # Okay to go. 
        # We use the same type of loop that we used for creating the grid
        self.j = 0 # To determine which n'th slice is choosen
        for self.i_row in range(self.nrows):
            for self.i_col in range(self.ncols):
                
                self.i_to_show = self._choose_image()
                
                self.pos_x = self.map.xmin + self.i_row/self.nrows*(self.map.xmax - 
                                                                    self.map.xmin)
                self.pos_y = self.map.ymin + self.i_col/self.ncols*(self.map.ymax - 
                                                                    self.map.ymin)
                # Create a textitem
                if self.list_label == -1:
                    text = '%d/%d'%(self.i_to_show+1, self.N_slice)
                else:
                    text = self.list_label[self.i_to_show]
                self.textitem_slice = egg.pyqtgraph.TextItem(text=text, 
                                                             color=(200, 200, 255),
                                                             fill=(0, 0, 255, 100)) 
                # Add it to the list
                # Will be useful for cleaning up when refreshing
                self.list_textitem_slice.append(self.textitem_slice)
                self.map.plot_image.addItem(self.textitem_slice)
#                self.textitem_slice.setText(self.label_slice_date)
                self.textitem_slice.setPos(self.pos_x, self.pos_y)  
                # Update the number of slice that we have (starting from 0)  
                self.j += 1
               
        
    def set_list_image(self, 
                       list_image, 
                       list_label=-1):
        """
        Send a list to the plot and update all corresponding attribute
        
        list_image:
            list of slices to plot
        list_label:
            List of string. 
            Must be the same size as the list of slice. 
            Label for each slice. 
            If nothing, it will be a number. 
        Note:
            The number of image shown will be nrows*cols. This may not be the
            total number of slices. So only an evenly spaces element of 
            list_image will be shown.
        """
        _debug('Grid:set_list_image')
        # Get the list
        self.list_image = list_image   
        self.list_label = list_label
        # Note this for when using normalization
        self.list_image_unnormal = self.list_image
        
        
        # We have enough to go and update the plot
        self._update_plot()
        
class GUIimage(egg.gui.Window):
    """
    A convenient GUI to view slices. :D
    
    """
    
    def __init__(self, name="Best Slice viewer", size=[1800,1000]): 
        """
        """
        _debug('GUIimage:__init__')
        _debug('Everyone thinks of changing the world, but no one thinks of changing himself. – Leo Tolstoy')
        
        # Run the basic stuff for the initialization
        egg.gui.Window.__init__(self, title=name, size=size)        

        # We will have a tabe for each way to inspect the slices
        self.tabs = self.place_object(egg.gui.TabArea(), alignment=0)
        
        # Tab for the grid of images
        self.tab_grid = self.tabs.add_tab('Grid view')
        self.grid = Grid()
        self.tab_grid.place_object(self.grid )

        # Tab for the slice by slice image
        self.tab_single = self.tabs.add_tab('Slice by Slice')
        self.single= SliceBySlice()
        self.tab_single.place_object(self.single)
        
    def set_list_image(self, 
                       list_image,
                       list_label=-1):
        """
        Send a list to the plot and update all corresponding attribute
        
        list_image:
            list of slices to plot
            
        list_label:
            List of string. 
            Must be the same size as the list of slice. 
            Label for each slice. 
            If nothing, it will be a number. 
            
        """
        _debug('GUIimage:set_list_image')
        # Add the slice in each tabs
        self.grid.set_list_image(list_image, list_label)
        self.single.set_list_image(list_image)
        
                
if __name__ == '__main__':  
    _debug_enabled     = True
    # 'Calibrate' the slice viewer 
    # In other word, verify that it shows what it should 
    
    # Do any thing
    x = np.linspace(-20, 20, 104)
    y = np.linspace(-20, 20, 103)
    list_z = np.linspace(6, 20, 25)
    X, Y = np.meshgrid(x, y)
    Nz = len(list_z)
    list_image = []
    list_label = []
    for i in range(Nz):
        print('Defining slice %d/%d'%(i, Nz))
        z = list_z[i]
        # A cool plot
        ts = np.linspace(5, z, 50)
        f = 0
        for t in ts:
            x1 = t*np.cos(t)
            y1 = t*np.sin(t)        
            Gauss_t = np.exp(-( (X-x1)**2+(Y-y1)**2 )/4)
            f += Gauss_t        
#        f = np.cos(Y+X*z)*(Y**2+z**2)*np.exp(-(x**2/2+z**2/5))
        list_image.append(f)
        list_label.append( '%d best'%i )
        
    #The grid viewer
    self = Grid()
    self.show() # Important to see all the GUI
    self.set_list_image(list_image, list_label=list_label)
    # self.set_list_image(list_image)
    # Just Slice by slice
#    self = SliceBySlice()
#    self.show()
#    self.set_list_image(list_image)
    
    # The full GUI
#    self = GUIimage()
#    self.show()
#    self.set_list_image(list_image)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
