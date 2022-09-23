# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:29:22 2022

Goal:
    Have a slice viewer. 
    Adapted to visualize a list of Matrix (NxM)
@author: client
"""

from spinmob import egg
import traceback
# The super cool 2D mapper
from gui_map_2D import Map2D
#The following is for plotting matplotlib object in the gui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
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
        self.slice = self.list_slice[self.slice_index]
        self.map.set_data(self.slice)
        
        
    def set_list_slice(self, 
                       list_slice):
        """
        Send a list to the plot and update all corresponding attribute
        
        list_slice:
            list of slices to plot
        """
        _debug('SliceBySlice:set_list_slice')
        
        self.list_slice = list_slice 
        
        # Update attributes that depends on the list of slice
        
        self.N_slice = len(self.list_slice)  
        self.slice_index_max = self.N_slice - 1
        self.slice_index  = int(self.slice_index_max/2)     
        self.prev_slice_index = self.slice_index     
        # Show the plot with all these attributes
        self._update_plot()     
        
        
        
class NbyM(egg.gui.Window):
    """
    A GUI to view N by M slices 
    """
    
    def __init__(self): 
        """
        """
        _debug('NbyM:__init__')
        _debug('Everyone thinks of changing the world, but no one thinks of changing himself. – Leo Tolstoy')

        #Run the basic stuff for the initialization
        egg.gui.Window.__init__(self)
        
        #Create a figure and canvas
        self.fig = plt.figure(tight_layout=False, figsize=(7,7)) #tight_layout is for labels to be visible or note
        # this is the Canvas Widget that displays the `figure`. It takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.fig) 
        self.canvas.draw()
        #Add it to the GUI
        self.place_object(self.canvas,
                          row=0, column=0)
        # Slider for the color scale
        # Two sliders. One for the min and one for the max. 
        self.slider_cmin = egg.gui.Slider(autosettings_path='slider_viewer_cmin',
                                          steps=50)
        self.slider_cmax = egg.gui.Slider(autosettings_path='slider_viewer_cmax',
                                          steps=50)
        self.place_object(self.slider_cmin,  
                          row=1, column=0, column_span=1)
        self.place_object(self.slider_cmax,  
                          row=2, column=0, column_span=1)
        self.slider_cmin.event_changed = self._slider_cmin_changed     
        self.slider_cmax.event_changed = self._slider_cmax_changed    

        # Some stretching. Always important for a good physical health.
        self.set_row_stretch   (row=2, stretch=10)
        self.set_column_stretch(column=5, stretch=1000)

        
    def _slider_cmin_changed (self, value):
        """
        Change attributes and update the plot
        """
        _debug('NbyM:_slider_cmin_changed')

        # Define the extrema of the slider to make sure the value make sense
        if ((value > self.possible_min) and 
            (value < self.slider_cmax.get_value() ) ):   
            _debug('yes ', value, ' and ', self.slider_cmin.get_value())
            # Note the value for subsequent verification
            self.slider_prev_cmin = value
            # Update the value and the plot
            self.cmin = value
            self._update_plot()
            
        else:
            # Trick to make sure the slider doesn't go past where it shouldn't
            self.slider_cmin.set_value(self.slider_prev_cmin,
                                       block_signals=True)
                    
    def _slider_cmax_changed (self, value):
        """
        Change attributes and update the plot
        """
        _debug('NbyM:_slider_cmax_changed')
        
        # Define the extrema of the slider to make sure the value make sense
        if ((value < self.possible_max) and 
            (value > self.slider_cmin.get_value()) ):
            _debug('yes ', value, ' and ', self.slider_cmax.get_value())
            # Note the value for subsequent verification
            self.slider_prev_cmax = value
            # Update the value and the plot
            self.cmax = value
            self._update_plot()        
        else:
            # Trick to make sure the slider doesn't go past where it shouldn't
            self.slider_cmax.set_value(self.slider_prev_cmax,
                                       block_signals=True)        
    def _update_plot(self):
        """
        Function that updates the plot with the current attribute. 
        It is important that the function do not depend on any parameters, 
        because it will be called after update single or few attributes at a 
        time. 
        """
        _debug('NbyM:_update_plot')

        # Updates the variables that depends on user-defined attributes
        self.N_slice = len(self.list_slice)   
        self.N_plots = self.n_rows*self.n_cols # How many plot in total
        
        # Define the NxM image
        self.axes = self.fig.subplots(ncols=self.n_cols, nrows=self.n_rows,
                                       sharex=True, sharey=True)
        # Reduce spacing between plots
        # If not sure, search 'subplots_adjust' in the doc here:
        # https://matplotlib.org/3.3.4/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
        # We set left/bottom/etc to reduce unused spacing around the plot
        self.fig.subplots_adjust( wspace=0, hspace=0,
                                 left=0, bottom=0, right=1, top=1)
#        self.fig.set_constrained_layout_pads(w_pad=1, h_pad=0, wspace=1, hspace=0.0)
        # Scan each slice to plot
        self.axs = self.axes.flat #Flat the axis in order to acces them with a single index
        for j in range(self.N_plots):
            i_to_show = j*int(self.N_slice/self.N_plots) # Indice of the slice to show
            # Not that it is not wokring. k_element should be 0 and nothing more. 
            self.axs[j].imshow(self.list_slice[i_to_show], 
                                cmap='gray', origin='lower',
                                vmin=self.cmin, vmax=self.cmax)
            self.axs[j].axis('off')
            self.axs[j].text(0.1,0.8,"Slice %d/%d"%(i_to_show, self.N_slice), 
                    transform=self.axs[j].transAxes, color='lime', fontsize=11)  
            # Together with wspace=0 and hspace=0, this removes the space between the plots
            self.axs[j].set_xticklabels([])
            self.axs[j].set_yticklabels([])
            
        
        #The following update the plot. 
        self.fig.canvas.draw_idle()            
        
    def get_minmax(self):
        """
        Get the minimum and maximum value of the signal among all the slices. 
        
        Note:
            Require the slices to be already 'loaded'.
        """
        _debug('NbyM:get_minmax')

        # Updates the variables that depends on user-defined attributes
        self.N_slice = len(self.list_slice)   
        self.N_plots = self.n_rows*self.n_cols # How many plot in total

        # Compute the overall max and min. 
        # Check all element that we will plot
        for j in range(self.N_plots):
            i_to_show = j*int(self.N_slice/self.N_plots) # Indice of the slice to show.
            z = self.list_slice[i_to_show]      
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
               
        
    def set_list_slice(self, 
                       list_slice, 
                       scale_color=False,
                       n_rows=3,
                       n_cols=2):
        #TODO Remove the other parameters and put them into three-dict
        #TODO Concacanate the slice instead of using matplotlib
        """
        Send a list to the plot and update all corresponding attribute
        
        list_slice:
            list of slices to plot
        scale_color=False:
            (bool)
            If True, all the images will have the same scaling
        n_rows:
            Number of row to make in the N by M images. 
        n_cols:
            Number of columns to make in the N by M images. 
            
        Note:
            The number of image shown will be nrows*cols. This may not be the
            total number of slices. So only an evenly spaces element of 
            list_slice will be shown.
        """
        _debug('NbyM:set_list_slice')
        # Note down all attributes (important for the update function)
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.list_slice = list_slice   
        
        # Note the extrema of the intensity and fix the slider
        self.possible_min, self.possible_max = self.get_minmax()
        if scale_color:
            self.cmin, self.cmax = self.possible_min, self.possible_max
        else:
            self.cmin, self.cmax = None, None
            # Fix the extrem of the slider
        self.slider_prev_cmin = self.possible_min 
        self.slider_prev_cmax = self.possible_max 
        self.slider_cmin.number_lower_bound.set_value(self.possible_min, block_signals=True)
        self.slider_cmin.number_upper_bound.set_value(self.possible_max, block_signals=True)
        self.slider_cmax.number_lower_bound.set_value(self.possible_min, block_signals=True)
        self.slider_cmax.number_upper_bound.set_value(self.possible_max, block_signals=True) 
        # Fix the default value of the slider
        self.slider_cmin.set_value(self.possible_min, block_signals=True)
        self.slider_cmax.set_value(self.possible_max, block_signals=True)



            
        # Show the plot with all these attributes
        self._update_plot()
        
class GUISlices(egg.gui.Window):
    """
    A convenient GUI to view slices
    
    """
    
    def __init__(self, name="Best Slice viewer", size=[1800,1000]): 
        """
        """
        _debug('GUISlices:__init__')
        _debug('Everyone thinks of changing the world, but no one thinks of changing himself. – Leo Tolstoy')
        
        # Run the basic stuff for the initialization
        egg.gui.Window.__init__(self, title=name, size=size)        

        # We will have a tabe for each way to inspect the slices
        self.tabs = self.place_object(egg.gui.TabArea(), alignment=0)
        
        # Tab for the grid of images
        self.tab_nbym = self.tabs.add_tab('Grid view')
        self.nbym = NbyM()
        self.tab_nbym.place_object(self.nbym)

        # Tab for the slice by slice image
        self.tab_single = self.tabs.add_tab('Slice by Slice')
        self.single= SliceBySlice()
        self.tab_single.place_object(self.single)
        
    def set_list_slice(self, 
                       list_slice):
        """
        Send a list to the plot and update all corresponding attribute
        
        list_slice:
            list of slices to plot
        """
        _debug('GUISlices:set_list_slice')
        # Add the slice in each tabs
        self.nbym.set_list_slice(list_slice)
        self.single.set_list_slice(list_slice)
        
                
if __name__ == '__main__':  
    _debug_enabled     = True
    # 'Calibrate' the slice viewer 
    # In other word, verify that it shows what it should   
    import numpy as np
    
    # Do any thing
    x = np.linspace(-1, 7, 104)
    y = np.linspace(-12, 12, 103)
    list_z = np.linspace(-10, 10, 57)
    X, Y = np.meshgrid(x, y)
    Nz = len(list_z)
    list_slice = []
    for i in range(Nz):
        z = list_z[i]
        f = np.cos(Y+X*z)*(Y**2+z**2)*np.exp(-(x**2/2+z**2/5))
        list_slice.append(f)
        
#    self = NbyM()
#    self.show() # Important to see all the GUI
#    self.set_list_slice(list_slice, n_rows=3, n_cols=3, scale_color=False)
    
#    self = SliceBySlice()
#    self.show()
#    self.set_list_slice(list_slice)
    
    self = GUISlices()
    self.show()
    self.set_list_slice(list_slice)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
