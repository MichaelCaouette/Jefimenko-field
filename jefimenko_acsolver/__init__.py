# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:20:25 2022

@author: mcaoue2
"""

# The goal of the following gymnastic is to add the path of the package to the 
# script that will use it
import sys
import os
# getcwd gives the current working directory. 
# We add the current folder, assuming that the user of the package put it in
# one folder. 
pkg_path = os.getcwd() + '\jefimenko_acsolver' 
if not (pkg_path in sys.path):
    # Prevent to add many time the same path, if called multiple time
    sys.path.insert(0, pkg_path)

print()
print('Thanks for using jefimenko_acsolver. Note that it is taken into this directory:')
print('pkg_path = ', pkg_path)
print()

