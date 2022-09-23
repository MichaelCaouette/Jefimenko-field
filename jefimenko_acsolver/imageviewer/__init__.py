# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:58:18 2022

@author: mcaoue2
"""

# The goal of the following gymnastic is to add the path of the package to the 
# script that will use it
import sys
import os
# getcwd gives the current working directory. 
# We add the current folder, assuming that the user of the package put it in
# one folder. 
pkg_path = os.getcwd() + '\jefimenko_acsolver\imageviewer' 
if not (pkg_path in sys.path):
    # Prevent to add many time the same path, if called multiple time
    sys.path.insert(0, pkg_path)

# Make the module to work. Maybe not required
# from . import gui_map_2D as gui_map_2D
# from . import slices as slices
# from . import sliceviewer_mpbForGrid

print()
print('Thanks for using imageviewer. Note that it is taken into this directory:')
print('pkg_path = ', pkg_path)
print()

