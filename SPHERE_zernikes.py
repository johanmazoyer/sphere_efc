# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:55:05 2023

@author: apotier
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import poppy
from astropy.io import fits
import SPHERE_EFC_Func as func


def add_sum_zernike(vect):
    npix = 240
    a = 0
    outside = 0 #np.nan
    vect = np.array(vect)
    for i in np.arange(len(vect)):
        a += poppy.zernike.zernike1(vect[i,0], npix=npix, outside = outside ) * vect[i,1]
    
    return a

def record_slope_from_2Darray(MatrixDirectory, dir, file, refslope, name):
    """ --------------------------------------------------
    Creation of the phase shape (in slope) to apply on the DM    
    
    Parameters:
    ----------
    amptopushinnm:
    dir:
    refslope:

    -------------------------------------------------- """
    # Read SAXO calibrations
    # static calibrations
    rad_632_to_nm_opt = 632/2/np.pi
    IMF_inv = fits.getdata(MatrixDirectory + 'SAXO_DM_IFM_INV.fits', ignore_missing_end = True)
    

    coe = (file.flatten())@IMF_inv
    coe = coe / rad_632_to_nm_opt
    slopetopush = func.VoltToSlope(MatrixDirectory, coe)
    func.recordslopes(slopetopush, dir, refslope, name)
    return 0  

def extract_zernike_from_string(my_array_str):
    # Split the string into a list of lines
    lines = my_array_str.strip().split('\n')

    # Group the lines by two in a new list
    output_list = []
    for i in range(0, len(lines), 2):
        pair = [int(lines[i]), int(lines[i+1])] if i+1 < len(lines) else [int(lines[i]), None]
        output_list.append(pair)

    return output_list


# Get the exported array as a string
my_array_str = os.environ.get('MY_ARRAY')
WORK_PATH = os.environ['WORK_PATH']
MATRIX_PATH = os.environ['MATRIX_PATH']
slope_ini = os.environ['SLOPE_INI']

output_list = extract_zernike_from_string(my_array_str)
# Print the output list
#print(output_list)
  


a = add_sum_zernike(output_list)


record_slope_from_2Darray(MATRIX_PATH, WORK_PATH, a, slope_ini, 'Zernikes')

print('Zernike slopes recorded')
# # print(np.nanstd(a))
# plt.imshow(a)
# plt.show()
# plt.pause(5)
    