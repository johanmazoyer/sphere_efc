#%% Extracting the data
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:34:51 2024

@author: axel

Note: make sure you have all the relevant background and psf images in the same folder as the experiment files in ImageDirectory
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
#import seaborn as sns
import Performance_function as perf
import SPHERE_EFC_Func as SPHERE
import Definitions_for_matrices as matrices
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import fmin_powell as fmin_powell
import glob
from natsort import natsorted

from astropy.time import Time
import astropy.units as u

import importlib

importlib.reload(SPHERE)


onsky=1 #Set 0 for internal pup ; 1 for an on sky correction
#Coronagraph that is used
coro='APLC'
#Dark hole size : param namemask in CreateMatrixfromModelEFConSPHERE.py
dhsize=1

gain=0 # Keep 0 for CDI postprocessing!!

#Algorithm for estimation. Should be either PWP or BTW
estim_algorithm='PWP'

#Number of probing actuator
zone_to_correct='FDH' #vertical #horizontal #FDH

#Type of probes used for PWP
probe_type='individual_act' #'sinc' #'individual_act'

#SizeProbes : can be 296, 400 or 500 (in nm)
size_probes=400

#Do you want to center your image at each iteration. Set 1 for yes, 0 for no.
centeringateachiter=0
#Do you want to rescale the coherent intensity to match the total intensity in the DH?
rescaling=0

#Data
#'HR4796A' - Experiment 05 - cdi
#'CPD-366759' - Experiment 06 - cdi 2
# 'HD169142' - Experiment 10 - cdi 3
#'HD163264' - Experiment 11 - cdi 4 - minus sign for ADI!
target_name = 'HD163264'
if target_name == 'HR4796A':
    nb_experiment = '05'
    fold = 'cdi/'
    rotation_sign = 1
    estimated_onsky_PA_of_the_planet = 0
    remove = [14,12,9,8,7,6,1] #Added 6,7,8,9
elif target_name == 'CPD-366759':
    nb_experiment = '06'
    fold = 'cdi 2/'
    rotation_sign = 1
    estimated_onsky_PA_of_the_planet = 0
    remove = [8,6,4,3,2]
elif target_name == 'HD169142':
    nb_experiment = '10'
    fold = 'cdi 3/'
    rotation_sign = 1
    remove = [4, 3, 2, 1, 0]
    estimated_onsky_PA_of_the_planet = 0
elif target_name == 'HD163264':
    nb_experiment = '11'
    fold = 'cdi 4/'
    rotation_sign = -1
    estimated_onsky_PA_of_the_planet = 0
    remove = [12]
    

# Path 
WORK_PATH0='/Users/apoitier/Documents/Research/CDI/SPHERE/cdi-jun04-2024/'
WORK_PATH1='/Users/apoitier/Documents/Research/Softwares/sphere_efc/'
MatrixDirectory=WORK_PATH1+'MatricesAndModel/'
ImageDirectory=WORK_PATH0 + fold #+'SlopesAndImages/cdi-jun04-2024/' + fold

exp_name = 'Experiment00' + nb_experiment + '_'


centerx, centery = fits.getdata(ImageDirectory + exp_name + 'centerxy.fits')
which_nd = 'ND_2.0'

ModelDirectory = WORK_PATH1 + 'Model/'

if onsky == 0:
    lightsource_estim = 'InternalPupil_'
else:
    lightsource_estim = 'VLTPupil_'

lightsource_estim = lightsource_estim + coro + '_'


if coro == 'APLC':
    if zone_to_correct == 'vertical':
        posprobes = [678 , 679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
    elif zone_to_correct == 'horizontal':
        posprobes = [893 , 934]
    elif zone_to_correct == 'FDH':
        posprobes = [678 , 679, 720]

    
elif coro == 'FQPM':
    if zone_to_correct == 'vertical':
        posprobes = [678 , 679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
    elif zone_to_correct == 'horizontal':
        posprobes = [1089 , 1125] #FQPM
    elif zone_to_correct == 'FDH':
        raise ValueError('This setting is not available for FQPM yet')

param = {
  "ImageDirectory": ImageDirectory,
  "MatrixDirectory": MatrixDirectory,
  "exp_name": exp_name,
  "dhsize": dhsize,
  "zone_to_correct":zone_to_correct,
  "centerx": centerx,
  "centery": centery,
  "which_nd": which_nd,
  "onsky": onsky,
  "size_probes": size_probes,
  "amplitudeEFCMatrix": 8, #required for the code to run but useless
  "corr_mode": 1, #required for the code to run but useless
  "centeringateachiter": centeringateachiter,
  "dimimages": 200,
  "gain": gain,
  "rescaling": rescaling,
  "estim_algorithm": estim_algorithm,
  "probe_type": probe_type,

  "coro": coro,
  "live_matrix_measurement": False,
  "wave": 1.667e-6,
  "ModelDirectory": ModelDirectory,
  "lightsource_estim": lightsource_estim,
  "lightsource_corr": lightsource_estim, #required for the code to run but useless
  "posprobes": posprobes

}


processed_directory = ImageDirectory + 'processed_data/PCA_on_diff/'
if not os.path.exists(processed_directory):
        os.makedirs(processed_directory)

cube_co = []
cube_inco = []
cube_tot = []
PA = []


nbiter = 2
for file_raw in natsorted(glob.glob(ImageDirectory+exp_name+'*Probe_0001*.fits')):
    param['nbiter'] = nbiter
    coherent_signal, incoherent_signal, imagecorrection, Images_to_display, pentespourcorrection  = SPHERE.resultEFC(param)
    cube_co.append(coherent_signal)
    cube_inco.append(incoherent_signal)
    cube_tot.append(imagecorrection)
    time_now = fits.getval(file_raw,'DATE-OBS')
    PA.append(perf.PA_on_detector(target_name, time_now, estimated_onsky_PA_of_the_planet, verbose=False))
    nbiter = nbiter + 1
    
cube_co = np.array(cube_co)
cube_inco = np.array(cube_inco)
cube_tot = np.array(cube_tot)
PA = rotation_sign * np.array(PA)
    
fits.writeto(processed_directory+'signal_coh.fits',cube_co,overwrite=True)
fits.writeto(processed_directory+'signal_inc.fits',cube_inco,overwrite=True)
fits.writeto(processed_directory+'signal_tot.fits',cube_tot,overwrite=True)
fits.writeto(processed_directory+'PA.fits',-PA,overwrite=True)



#%% Normal ADI ---------------------------------------

import Performance_function as perf

#Filter unwanted images from the cube; remove corresponds to their index
#remove = []#np.append(np.arange(3,len(cube_co))[::-1],1)

#Create a mask to apply to each CDI and total intensity images. Can adjust the parameters. Was 10,65
#mask = matrices.creatingMaskDH(200,'circle',choosepixDH=[-70, 70, 5, 70], circ_rad=[12, 65], circ_side="Full", circ_offset=0, circ_angle=0)

print('Total intensity')
cube_tot_removed = cube_tot.copy() #* mask
newPA = PA.copy()
for i in remove:
    cube_tot_removed = np.delete(cube_tot_removed, i, axis = 0)
    newPA = np.delete(newPA, i, axis = 0)
print(len(cube_tot_removed))

total_rotated_removed = perf.rotate_cube(cube_tot_removed, - newPA)
fits.writeto(processed_directory +'Rotated_tot.fits', total_rotated_removed, overwrite=True)


# ADI of cube tot
u,s,vh = perf.get_cube_svd(cube_tot_removed)
vector = np.arange(len(cube_tot_removed))
ADI_tot_result = perf.reduction_ADI(u, s, vh, vector, - newPA)

ADI_tot_filtered = []
for high_pass_filter_cut in np.arange(1,11):
    ADI_tot_filtered.append(perf.high_pass_filter(ADI_tot_result[0], high_pass_filter_cut))
ADI_tot_filtered = np.array(ADI_tot_filtered)
fits.writeto(processed_directory +'Rotated_stacked_hpfiltered_tot.fits', ADI_tot_filtered, overwrite=True)
fits.writeto(processed_directory +'Rotated_stacked_tot.fits', ADI_tot_result, overwrite=True)

fits.writeto(processed_directory+'ADI_tot.fits', ADI_tot_result, overwrite=True)

    

#%% Test CDI + rotation + stack + high pass filter -------------------------------------
import Performance_function as perf

#Filter unwanted images from the cube; remove corresponds to their index
#remove = []#np.append(np.arange(3,len(cube_co))[::-1],1)

#Create a mask to apply to each CDI and total intensity images. Can adjust the parameters. Was 10,65
mask = matrices.creatingMaskDH(200,'circle',choosepixDH=[-70, 70, 5, 70], circ_rad=[12, 65], circ_side="Full", circ_offset=0, circ_angle=0)

print('INCOHERENT COMPONENT')
cube_inco_removed = cube_inco.copy()* mask

#Filter pixel with bad estimation. Can adjust the parameter
cube_inco_removed[np.where(np.abs(cube_inco_removed)>1e-2)]=0

#Removing worse images from the cube
newPA = PA.copy()
for i in remove:
    cube_inco_removed = np.delete(cube_inco_removed, i, axis = 0)
    newPA = np.delete(newPA, i, axis = 0)

# ADI of cube inco
u,s,vh = perf.get_cube_svd(cube_inco_removed)
vector = np.arange(len(cube_inco_removed))
ADI_inco_result = perf.reduction_ADI(u, s, vh, vector, - newPA)
cube_inco_rotated = perf.rotate_cube(cube_inco_removed, - newPA)

#ADI_hide = remove_center_cube(ADI_result, 30)

# High-pass filtering of ADI[0] (where the cube has simply been rotated and stacked)
ADI_inco_filtered = []
for high_pass_filter_cut in np.arange(1,11):
    ADI_inco_filtered.append(perf.high_pass_filter(ADI_inco_result[0], high_pass_filter_cut))
ADI_inco_filtered = np.array(ADI_inco_filtered)
fits.writeto(processed_directory +'Rotated_stacked_hpfiltered_inc.fits', ADI_inco_filtered, overwrite=True)
fits.writeto(processed_directory +'Rotated_stacked_inc.fits', ADI_inco_result[0], overwrite=True)

#ADI_hide_and_filtered = remove_center_cube(ADI_filtered, 30)

fits.writeto(processed_directory +'ADI_inc.fits', ADI_inco_result, overwrite=True)
fits.writeto(processed_directory +'Rotated_inc.fits', cube_inco_rotated, overwrite=True)


#%% Test PCA CDI --------------------------------------------
newPA = PA.copy()

#Remove unwanted data in cube_co and rotate
cube_co_removed = cube_co.copy() * mask
for i in remove: 
    cube_co_removed = np.delete(cube_co_removed, i, axis = 0)
    newPA = np.delete(newPA, i, axis = 0)
cube_co_rotated = perf.rotate_cube(np.array(cube_co_removed), - newPA)    
fits.writeto(processed_directory+'Rotated_coh.fits', cube_co_rotated, overwrite=True)

#Calculating the principle components of cube_co
u,s,vh = perf.get_cube_svd(cube_co_removed)
print(s.shape)
princ_comp = (s[:, np.newaxis]*vh).reshape((len(vh), 200, 200))

for i in np.arange(len(princ_comp)):
    princ_comp[i] = princ_comp[i]/np.sign(np.sum(princ_comp[i]))

fits.writeto(processed_directory+'signal_coh_principle_components.fits', princ_comp, overwrite=True)


def extract_modal_coefficients(OPDs, basis, pix_in_aperture):
    """
    Extract modal coefficients from time series

    Parameters
    ----------
    OPDs : 3D array
        OPD time series.
    basis : 3D array
        Modal decomposition.
    pix_in_aperture : INT
        Number of pixels that build the aperture.

    Returns
    -------
    coeff : 1D array
        Modal coefficients.

    """
    N = len(OPDs)
    nterms_tot = len(basis)
    coeff=np.zeros((nterms_tot,int(N)))
    for j in np.arange(int(N)):
        if j%100 == 0 : print('Calculating OPD nb', j)
        for i in np.arange(nterms_tot):
            product = (OPDs[j]*basis[i]).flatten()
            coeff[i,j] = np.sum(product[np.where(np.isnan(product)==0)])/pix_in_aperture
    return coeff



def reconstruct_OPD_from_coeff(basis, coeff, OPD_nb):
    """
    Reconstruct an optical path difference image from modal coefficients

    Parameters
    ----------
    basis : 3D array
        Modal decomposition.
    coeff : 1D array
        Modal coefficients.
    OPD_nb : INT
        OPD index we want to reconstruct.

    Returns
    -------
    reconstruct : TYPE
        DESCRIPTION.

    """
    nterms_tot = basis.shape[0]
    npix_basis = basis.shape[1]
    #np.reshape(coeff[:,lequel]@flattened_PCA,(256,256))
    reconstruct=np.zeros((npix_basis,npix_basis))
    for i in np.arange(nterms_tot):
        reconstruct=reconstruct+coeff[i,OPD_nb]*basis[i]
    return reconstruct


# coeffs = extract_modal_coefficients(cube_co_test, princ_comp, 1)

# reco = []
# for i in np.arange(len(cube_co_test)):
#     reco.append(reconstruct_OPD_from_coeff(princ_comp, coeffs, i))

# reco = np.array(reco)#cube_tot_filtered - np.array(reco)    
# fits.writeto(processed_directory+'projection.fits', reco, overwrite=True)

# Filter the highest principal components in cube_co and subtract cube_co from cube_tot
scan_modes = np.arange(len(cube_co_removed)+1)
cube_inco_mfiltered_rotated = []
cube_inco_mfiltered_save = []
for i in scan_modes:
    print(i)
    nb_images,isz,isz = vh.shape[0], int(np.sqrt(vh.shape[1])), int(np.sqrt(vh.shape[1])) 
    filtered = s.copy()
    filtered[i:]=0
    #filtered[20-nb_modes_filtered:]=0
    cube_co_mfiltered = (u * filtered @ vh).reshape(nb_images,isz,isz)
    cube_inco_mfiltered = cube_tot_removed - cube_co_mfiltered
    cube_inco_mfiltered_save.append(cube_inco_mfiltered)
    cube_inco_mfiltered_rotated.append(perf.rotate_cube(np.array(cube_inco_mfiltered), - newPA))

cube_inco_mfiltered_rotated = np.array(cube_inco_mfiltered_rotated)
cube_inco_mfiltered_save = np.array(cube_inco_mfiltered_save)

fits.writeto(processed_directory +'Rotated_mfiltered_inc.fits', cube_inco_mfiltered_rotated, overwrite=True)


# ADI of cube cube
Rotated_stacked_hpfiltered_mfiltered_inc = []
Rotated_stacked_mfiltered_inc = []
for i in scan_modes:
    u,s,vh = perf.get_cube_svd(cube_inco_mfiltered_save[i])
    vector = np.arange(len(cube_inco_mfiltered_save[i]))
    ADI_result = perf.reduction_ADI(u, s, vh, vector, - newPA) #[0] instead of vector
    Rotated_stacked_mfiltered_inc.append(ADI_result)
    #ADI_hide = remove_center_cube(ADI_result, 30)

    # High-pass filtering of ADI[0] (where the cube has simply been rotated and stacked)
    hp_filtered = []
    for high_pass_filter_cut in np.arange(1,11):
        hp_filtered.append(perf.high_pass_filter(ADI_result[0], high_pass_filter_cut))
    hp_filtered = np.array(hp_filtered)
    Rotated_stacked_hpfiltered_mfiltered_inc.append(hp_filtered)

Rotated_stacked_mfiltered_inc = np.array(Rotated_stacked_mfiltered_inc) * mask
Rotated_stacked_hpfiltered_mfiltered_inc = np.array(Rotated_stacked_hpfiltered_mfiltered_inc) * mask
fits.writeto(processed_directory +'Rotated_stacked_hpfiltered_mfiltered_inc.fits', Rotated_stacked_hpfiltered_mfiltered_inc, overwrite=True)
fits.writeto(processed_directory +'Rotated_stacked_mfiltered_inc.fits', Rotated_stacked_mfiltered_inc, overwrite=True)


#%% Test least-mean-squared Khi2# for PCA parameters -----------------------------------------------

mask_khi = matrices.creatingMaskDH(200,'circle',choosepixDH=[-70, 70, 5, 70], circ_rad=[45, 75], circ_side="Full", circ_offset=0, circ_angle=0)

scan_modes = np.arange(len(princ_comp))
cube_inco_khisquare_rotated = []
cube_inco_khisquare_save = []

for filt in scan_modes:
#filt = 5
    princ_comp_filtered = princ_comp[:filt].reshape(filt, 200 * 200).T

    Y=[]
    for k in np.arange(len(cube_tot_removed)):
        B = cube_tot_removed[k].flatten()
        #B -= np.mean(B) 
        Bmasked = B * mask_khi.flatten()
        Amasked = princ_comp_filtered * mask_khi.flatten()[:, None]
        X,res,rank,sprime = np.linalg.lstsq(Amasked, Bmasked, rcond=5)
        print(X)
        X[1:]=0
        #plt.imshow(cube_tot_filtered[0])
        solut = (B-princ_comp_filtered@X).reshape(200,200)
        Y.append(solut)

    cube_inco_khisquare_save.append(Y)
    cube_inco_khisquare_rotated.append(perf.rotate_cube(np.array(Y), - newPA))


cube_inco_khisquare_rotated = np.array(cube_inco_khisquare_rotated)
cube_inco_khisquare_save = np.array(cube_inco_khisquare_save)

fits.writeto(processed_directory +'Rotated_khisquare_inc.fits', cube_inco_khisquare_rotated, overwrite=True)    

# ADI of cube cube
Rotated_stacked_hpfiltered_khisquare_inc = []
Rotated_stacked_khisquare_inc = []
for i in scan_modes:
    u,s,vh = perf.get_cube_svd(cube_inco_khisquare_save[i])
    vector = np.arange(len(cube_inco_khisquare_save[i]))
    ADI_result = perf.reduction_ADI(u, s, vh, vector, - newPA) #0 instead of vector
    Rotated_stacked_khisquare_inc.append(ADI_result)
    #ADI_hide = remove_center_cube(ADI_result, 30)

    # High-pass filtering of ADI[0] (where the cube has simply been rotated and stacked)
    hp_filtered = []
    for high_pass_filter_cut in np.arange(1,11):
        hp_filtered.append(perf.high_pass_filter(ADI_result[0], high_pass_filter_cut))
    hp_filtered = np.array(hp_filtered)
    Rotated_stacked_hpfiltered_khisquare_inc.append(hp_filtered)

Rotated_stacked_khisquare_inc = np.array(Rotated_stacked_khisquare_inc) * mask
Rotated_stacked_hpfiltered_khisquare_inc = np.array(Rotated_stacked_hpfiltered_khisquare_inc) * mask
fits.writeto(processed_directory +'Rotated_stacked_hpfiltered_khisquare_inc.fits', Rotated_stacked_hpfiltered_khisquare_inc, overwrite=True)
fits.writeto(processed_directory +'Rotated_stacked_khisquare_inc.fits', Rotated_stacked_khisquare_inc, overwrite=True)

#fits.writeto(processed_directory +'test_svd.fits', ((u * s @ vh).reshape(vh.shape[0], int(np.sqrt(vh.shape[1])), int(np.sqrt(vh.shape[1])) )[0]), overwrite=True)







#%% Test performance CDI-PCA vs nb_iter
mask = matrices.creatingMaskDH(200,'circle',choosepixDH=[-70, 70, 5, 70], circ_rad=[12, 65], circ_side="Full", circ_offset=0, circ_angle=0)


processed_directory = ImageDirectory + 'processed_data_testiter/'
if not os.path.exists(processed_directory):
        os.makedirs(processed_directory)

cube_tot_removed_ini = cube_tot.copy() #* mask
cube_co_removed_ini = cube_co.copy() #* mask
newPA_ini = PA.copy()
for i in remove:
        cube_tot_removed_ini = np.delete(cube_tot_removed_ini, i, axis = 0)
        cube_co_removed_ini = np.delete(cube_co_removed_ini, i, axis = 0)
        newPA_ini = np.delete(newPA_ini, i, axis = 0)

ADI_tot_save = []
ADI_tot_filtered_save = []
ADI_inco_save = []
ADI_inco_filtered_save = []

for nb_iter in np.arange(len(cube_tot_removed_ini)):
    scan_iter = np.arange(1,len(cube_tot_removed_ini)-1-nb_iter)[::-1]

    cube_tot_removed = cube_tot_removed_ini.copy() #* mask
    cube_co_removed = cube_co_removed_ini.copy() * mask

    newPA = newPA_ini.copy()
    for i in scan_iter:
        cube_tot_removed = np.delete(cube_tot_removed, i, axis = 0)
        cube_co_removed = np.delete(cube_co_removed, i, axis = 0)
        newPA = np.delete(newPA, i, axis = 0)

    # ADI of cube tot
    u,s,vh = perf.get_cube_svd(cube_tot_removed)
    ADI_tot_result = perf.reduction_ADI(u, s, vh, [0], - newPA)[0]
    ADI_tot_save.append(ADI_tot_result*mask)

    #Filtering cube_co to its first component
    u,s,vh = perf.get_cube_svd(cube_co_removed)
    nb_images,isz,isz = vh.shape[0], int(np.sqrt(vh.shape[1])), int(np.sqrt(vh.shape[1])) 
    filtered = s.copy()
    filtered[1:]=0
    cube_co_mfiltered = (u * filtered @ vh).reshape(nb_images,isz,isz)
    cube_inco_mfiltered = cube_tot_removed - cube_co_mfiltered

    # ADI of cube inco mfiltered
    u,s,vh = perf.get_cube_svd(cube_inco_mfiltered)
    ADI_inco_result = perf.reduction_ADI(u, s, vh, [0], - newPA)[0] #[0] instead of vector
    ADI_inco_save.append(ADI_inco_result*mask)

    # High-pass filtering of ADI[0] (where the cube has simply been rotated and stacked)
    
    ADI_tot_filtered = []
    ADI_inco_filtered = []
    for high_pass_filter_cut in np.arange(1,11):
        ADI_tot_filtered.append(perf.high_pass_filter(ADI_tot_result, high_pass_filter_cut)*mask)
        ADI_inco_filtered.append(perf.high_pass_filter(ADI_inco_result, high_pass_filter_cut)*mask)
    ADI_tot_filtered_save.append(ADI_tot_filtered)
    ADI_inco_filtered_save.append(ADI_inco_filtered)


ADI_tot_save = np.array(ADI_tot_save)
ADI_tot_filtered_save = np.array(ADI_tot_filtered_save)
ADI_inco_save = np.array(ADI_inco_save)
ADI_inco_filtered_save = np.array(ADI_inco_filtered_save)



fits.writeto(processed_directory +'Rotated_stacked_tot_vs_iter.fits', ADI_tot_save, overwrite=True)
fits.writeto(processed_directory +'Rotated_stacked_hpfiltered_tot_vs_iter.fits', np.swapaxes(ADI_tot_filtered_save,0,1), overwrite=True)
fits.writeto(processed_directory +'Rotated_stacked_mfiltered_inco_vs_iter.fits', ADI_inco_save, overwrite=True)
fits.writeto(processed_directory +'Rotated_stacked_hpfiltered_mfiltered_inco_vs_iter.fits', np.swapaxes(ADI_inco_filtered_save,0,1), overwrite=True)



#%% Test PCA on probe images for super coherent intensity
importlib.reload(SPHERE)
mask = matrices.creatingMaskDH(200,'circle',choosepixDH=[-70, 70, 5, 70], circ_rad=[12, 65], circ_side="Full", circ_offset=0, circ_angle=0)

#Create cubes of probe difference images
cube_I1plus = SPHERE.reduce_cube_image(ImageDirectory+'*iter*_Probe_0001', param)
cube_I1moins = SPHERE.reduce_cube_image(ImageDirectory+'*iter*_Probe_0002', param)
diff1 = cube_I1plus - cube_I1moins
cube_I2plus = SPHERE.reduce_cube_image(ImageDirectory+'*iter*_Probe_0003', param)
cube_I2moins = SPHERE.reduce_cube_image(ImageDirectory+'*iter*_Probe_0004', param)
diff2 = cube_I2plus - cube_I2moins
cube_I3plus = SPHERE.reduce_cube_image(ImageDirectory+'*iter*_Probe_0005', param)
cube_I3moins = SPHERE.reduce_cube_image(ImageDirectory+'*iter*_Probe_0006', param)
diff3 = cube_I3plus - cube_I3moins
cube_image_coro = SPHERE.reduce_cube_image(ImageDirectory+'*iter*_coro_image*', param)

#Remove unwanted data in cube diff and image time series
diff1_removed = diff1.copy() * mask
diff2_removed = diff2.copy() * mask
diff3_removed = diff3.copy() * mask
cube_image_coro_removed = cube_image_coro.copy()
newPA = PA.copy()

for i in remove: 
    diff1_removed = np.delete(diff1_removed, i, axis = 0)
    diff2_removed = np.delete(diff2_removed, i, axis = 0)
    diff3_removed = np.delete(diff3_removed, i, axis = 0)
    cube_image_coro_removed = np.delete(cube_image_coro_removed, i, axis = 0)

    newPA = np.delete(newPA, i, axis = 0)

fits.writeto(processed_directory+'cube_diff1.fits', diff1, overwrite=True)
fits.writeto(processed_directory+'cube_diff2.fits', diff2, overwrite=True)
fits.writeto(processed_directory+'cube_diff3.fits', diff3, overwrite=True)

# Compute the PCA and filter modes in 
scan_modes = np.arange(1,len(diff1_removed)+10)
filtered_diff = []
j = 0
for cube_diff in [diff1_removed, diff2_removed, diff3_removed]:

    filtered_diff_i = []
    filtered_diff_cube = []

    u,s,vh = perf.get_cube_svd(cube_diff)
    princ_comp = (s[:, np.newaxis]*vh).reshape((len(vh), 200, 200))

    
    fits.writeto(processed_directory+'diff'+str(j)+'_principle_components.fits', princ_comp, overwrite=True)

    for i in scan_modes:
        print(i)
        nb_images,isz,isz = vh.shape[0], int(np.sqrt(vh.shape[1])), int(np.sqrt(vh.shape[1])) 
        filtered = s.copy()
        filtered[i:]=0
        a = (u * filtered @ vh).reshape(nb_images,isz,isz)
        #Save the filtered diff cube
        filtered_diff_cube.append(a)
        #Save the mean of filtered diff cube along the various iterations
        filtered_diff_i.append( np.mean(a, axis=0 )) #mean
    
    filtered_diff.append(filtered_diff_i)
    j=j+1

#Saved the three filtered mean cube_diff
filtered_diff = np.array(filtered_diff)
#Currently, save only filtered cube diff_3
filtered_diff_cube = np.array(filtered_diff_cube)

fits.writeto(processed_directory+'filtered_diff.fits', filtered_diff, overwrite=True)
fits.writeto(processed_directory+'filtered_diff_cube.fits', filtered_diff_cube, overwrite=True)

# The CDI part now starts here
#filename = probe_type + '_' + zone_to_correct + '_' + str(int(size_probes/37*37)) + 'nm' + '_'
filename = MatrixDirectory + lightsource_estim + filename + 'VecteurEstimation' + '.fits'
vectoressai = fits.getdata(filename)
print(filtered_diff.shape)
intensity_co = []
# For each filtered mode in diff1, diff2 and diff3, compute 1 super coherent intensity
for i in scan_modes:
    diff = filtered_diff[:,i-1]
    resultatestimation = SPHERE.estimateEab(diff, vectoressai)
    intensity_co.append(np.abs(resultatestimation)**2)


intensity_co = np.array(intensity_co)

fits.writeto(processed_directory+'super_intensity_co.fits', intensity_co, overwrite=True)

Rotated_stacked_mfiltered_inc = []
Rotated_stacked_hpfiltered_mfiltered_inc=[]
Cube_mfiltered_inc=[]

for i in scan_modes:
    Cube_mfiltered_inc.append(cube_image_coro_removed - intensity_co[i-1])
    u,s,vh = perf.get_cube_svd(cube_image_coro_removed - intensity_co[i-1])
    ADI_result = perf.reduction_ADI(u, s, vh, [0], - newPA) #[0] instead of vector
    Rotated_stacked_mfiltered_inc.append(ADI_result)
    #ADI_hide = remove_center_cube(ADI_result, 30)

    # High-pass filtering of ADI[0] (where the cube has simply been rotated and stacked)
    hp_filtered = []
    for high_pass_filter_cut in np.arange(1,11):
        hp_filtered.append(perf.high_pass_filter(ADI_result[0], high_pass_filter_cut))
    hp_filtered = np.array(hp_filtered)
    Rotated_stacked_hpfiltered_mfiltered_inc.append(hp_filtered)

Rotated_stacked_mfiltered_inc = np.array(Rotated_stacked_mfiltered_inc) * mask
Rotated_stacked_hpfiltered_mfiltered_inc = np.array(Rotated_stacked_hpfiltered_mfiltered_inc) * mask
Cube_mfiltered_inc = np.array(Cube_mfiltered_inc) * mask


fits.writeto(processed_directory+'Rotated_stacked_mfiltered_inc.fits', Rotated_stacked_mfiltered_inc, overwrite=True)
fits.writeto(processed_directory+'Rotated_stacked_hpfiltered_mfiltered_inc.fits', Rotated_stacked_hpfiltered_mfiltered_inc, overwrite=True)
fits.writeto(processed_directory+'Cube_mfiltered_inc.fits', Cube_mfiltered_inc, overwrite=True)





# cube = np.array(unfilt_cube_CDI)#cube_tot_unfiltered#cube_tot
# u,s,vh = get_cube_svd(cube)
# vector = np.arange(len(cube))
# ADI_result = reduction_ADI(u, s, vh, vector, angles)

# #ADI_hide = remove_center_cube(ADI_result, 30)

# #ADI_filtered = gaussian_filter_cube(ADI_result, 3)

# #ADI_hide_and_filtered = remove_center_cube(ADI_filtered, 30)

# fits.writeto(directory1+'CDI_ADI_reduced_unfilt.fits', ADI_result, overwrite=True)




#fits.writeto(directory+'cube_reduced_hidecenter.fits', ADI_hide, overwrite=True)
#fits.writeto(directory+'cube_reduced_nohidecenter.fits', ADI_result, overwrite=True)
#fits.writeto(directory+'cube_reducedandfiltered.fits', ADI_hide_and_filtered, overwrite=True)



""" u,s,vh = perf.get_cube_svd(cube_tot)
vector = np.arange(len(cube_tot))
ADI_result = perf.reduction_ADI(u, s, vh, vector, -PA)

ADI_filtered = []
for i in np.arange(len(ADI_result)):
    ADI_filtered.append(perf.high_pass_filter(ADI_result[i], high_pass_filter_cut))
ADI_filtered = np.array(ADI_filtered)
fits.writeto(processed_directory +'ADI_filtered.fits', ADI_filtered, overwrite=True)

fits.writeto(processed_directory+'ADI_reduced_nohidecenter.fits', ADI_result, overwrite=True)



 """



#%%


# lightsource_estim = 'VLTPupil_'+'FQPM'
# estimated_onsky_PA_of_the_planet = 210.532#31.6  # degree https://doi.org/10.1051/0004-6361/201834302



# date = 'apr07-2023'#'feb15'
# Series='02'
# #limit=(15, 5, None)
# limit=(None,)
# i=0
# directory1='/home/apotier/Documents/Recherche/DonneesTHD/EFConSPHERE2023/efc-'+date+'/Exp'+Series+'/'
# files=['*[0-9]Coherent*','*Incoherent*','*coro_image*0001*']

# #dark = fits.getdata(directory1 +'/../Calibration/SPHERE_BKGRD_EFC_32s_047_0001.fits')[0]
# #dark = fits.getdata(directory +'SPHERE_BKGRD_EFC_1s_045_0001.fits')[0]
# #dark = fits.getdata(directory1 +'/SPHERE_BKGRD_EFC_32s_047_0001.fits')[0]
# center = fits.getdata(directory1 +'Experiment00'+Series+'_centerxy.fits')

# if Series == '02':
#     maskDH=matrices.creatingMaskDH(200,
#                    'circle',
#                    choosepixDH=[8, 35, -35, 35],
#                    circ_rad=[7, 50],#
#                    circ_side="Bottom",
#                    circ_offset=7,
#                    circ_angle=0)
#     mask = maskDH #* bad_pix
#     ND = 1/0.00105
#     target_name = "Beta pictoris"


# cube_co = perf.CubeFits(directory1+files[0])[:limit[i]]
# cube_inco = perf.CubeFits(directory1+files[1])[:limit[i]]

# angles = []
# cube_tot = np.zeros_like(cube_co)#[:5]
# cube_tot_unfiltered = np.zeros_like(cube_co)
# for i in np.arange(len(cube_co))+2:
#     print(i)
#     cube_tot[i-2], PA = perf.process_image(directory1, 'Experiment00'+Series+'_', center, int(i), ND)
#     cube_tot_unfiltered[i-2] = perf.process_image(directory1, 'Experiment00'+Series+'_', center, int(i), ND,high_pass_filter=False)[0]
#     angles.append(-PA)


# concatenate = False
# if concatenate:
#     Series2 = '03'
#     directory2='C:/Users/apotier/Documents/Research/SPHERE/Data/efc-'+date+'/Exp'+Series2+'/'
#     center2 = fits.getdata(directory2 +'Experiment00'+Series2+'_centerxy.fits')
#     cube_co2 = perf.CubeFits(directory2+files[0])
#     cube_inco2 = perf.CubeFits(directory2+files[1])


#     cube_tot2 = np.zeros_like(cube_co2)
#     cube_tot_unfiltered2 = np.zeros_like(cube_co2)
#     for i in np.arange(len(cube_co2))+2:
#         print(i)
#         cube_tot2[i-2], PA = perf.process_image(directory2, 'Experiment00'+Series2+'_', center2, int(i), ND)
#         cube_tot_unfiltered2[i-2] = perf.process_image(directory2, 'Experiment00'+Series2+'_', center2, int(i), ND,high_pass_filter=False)[0]
#         angles.append(-PA)

#     cube_co = np.append(cube_co, cube_co2, axis = 0)
#     cube_inco = np.append(cube_inco, cube_inco2, axis = 0)
#     cube_tot = np.append(cube_tot, cube_tot2, axis = 0)
#     cube_tot_unfiltered = np.append(cube_tot_unfiltered, cube_tot_unfiltered2, axis = 0)


# wave = 1.667e-6
# pupsizeinmeter=8 #Pupsizeinmeter

# #Raccourcis conversions angles
# d2rad    = np.pi / 180.0 # degree to radian conversion factor
# d2arcsec = 3600
# arcsec2rad = d2rad/d2arcsec  # radian to milliarcsecond conversion factor


# #SPHERE detector resol
# resolinarcsec_pix = 12.25e-3  #arcsec/pix #12.27
# resolinrad_pix = resolinarcsec_pix*arcsec2rad  #rad/pix
# resolinpix_rad = 1 / resolinrad_pix     #pix/rad

# ld_rad = wave / pupsizeinmeter #lambda/D en radian
# ld_p = ld_rad * resolinpix_rad  #lambda/D en pixel
# ld_mas = ld_rad / arcsec2rad *1e3 #lambda/D en milliarcsec


# #%%
# maskCDI = matrices.creatingMaskDH(200,
#                'circle',
#                choosepixDH=[8, 35, -35, 35],
#                circ_rad=[7, 50],#
#                circ_side="Bottom", #bottom for exp 19!!
#                circ_offset=12, #12 for exp 19!! 18 for exp 20!!
#                circ_angle=0) #* bad_pix
# maskCDI[:,:115]=0


# maskvisu =  (matrices.creatingMaskDH(200,
#                               'square',
#                               choosepixDH=[-70, 70, 5, 70],
#                               circ_rad=[18, 60],#
#                               circ_side="Bottom",
#                               circ_offset=5,
#                               circ_angle=0) + matrices.creatingMaskDH(200,
#                                                             'square',
#                                                             choosepixDH=[-70, 70, -70, -70],
#                                                             circ_rad=[18, 60],#
#                                                             circ_side="Bottom",
#                                                                 circ_offset=5,
#                                                             circ_angle=0))*(1 - matrices.creatingMaskDH(200,
#                                                                                                         'circle',
#                                                                                                         choosepixDH=[-60, 60, 6, 55],
#                                                                                                         circ_rad=[0, 8],#
#                                                                                                         circ_side="Full",
#                                                                                                         circ_offset=5,
#                                                                                                         circ_angle=0) )*matrices.creatingMaskDH(200,
#                                                                                                                                                     'circle',
#                                                                                                                                                     choosepixDH=[-60, 60, 6, 55],
#                                                                                                                                                     circ_rad=[0, 65],#
#                                                                                                                                                     circ_side="Full",
#                                                                                                                                                     circ_offset=5,
#                                                                                                                                                     circ_angle=0) 
# #maskvisu = maskvisu + np.rot90(maskvisu,2)
# combien = 0
# maskvisu[:,100-combien:100+combien]=0


    

# cube_CDI = []
# new_cube_co = []

# for lequel in np.arange(len(cube_co)):
#     signal_co = cube_co.copy()[lequel]#SPHERE.noise_filter(cube_co.copy()[lequel], 3, 0.5)
#     signal_tot = cube_tot_unfiltered.copy()[lequel]#cube_tot.copy()[lequel]#

#     signal_co = perf.rescale_radial_CDI(signal_co, signal_tot, maskvisu, 3.5*1)
#     new_cube_co.append(perf.high_pass_filter(signal_co,2))  
#     cube_CDI.append(perf.high_pass_filter(signal_tot-signal_co,2))
    

# cube_CDI = np.array(cube_CDI*maskvisu)
# cube_tot_crop = np.array(cube_tot*maskvisu)
# fits.writeto(directory1+'CDI_iters.fits',cube_CDI,overwrite=True)
# fits.writeto(directory1+'CDIno_iters.fits',cube_tot_crop,overwrite=True)
   

# #%%
# vmin = -6e-6*1e6
# vmax = 6e-6*1e6
# linthresh = 1e-7*1e6
# linscale = 0.03
# cmap='inferno'
# fontsize = 18




# #plt.imshow(np.median(cube_CDI,axis=0)*maskvisu, vmin = -1e-5, vmax = 1e-5)im1 = ax1.imshow(high_pass_filter(cube_tot.copy()[4], 2)*maskDH*1e6 + np.median(cube_CDI,axis=0)*maskvisu*1e6, cmap = cmap, vmin= vmin, vmax = vmax)#, norm=SymLogNorm(linthresh=linthresh, linscale=linscale ,vmin= vmin, vmax = vmax))
# fig = plt.figure(figsize=(17,8))#,tight_layout=True)
# ax1 = fig.add_subplot(121)
# im1 = ax1.imshow(((cube_tot[0]*maskvisu))[30:170,30:170]*1e6, cmap = cmap, vmin= vmin, vmax = vmax)#, norm=SymLogNorm(linthresh=linthresh, linscale=linscale ,vmin= vmin, vmax = vmax))
# #ax1.contour(maskDH)
# ax1.set_title('Initial image', fontsize = fontsize)
# ax1.set_axis_off()

# ax2 = fig.add_subplot(122)
# im2 = ax2.imshow((np.median(cube_CDI[:-2]*maskvisu,axis=0))[30:170,30:170]*1e6, cmap = cmap, vmin= vmin, vmax = vmax)#, norm=SymLogNorm(linthresh=linthresh, linscale=linscale ,vmin= vmin, vmax = vmax))
# #ax2.contour(maskDH)
# ax2.set_title('Median incoherent', fontsize = fontsize)
# ax2.set_axis_off()


# fig.tight_layout()
# fig.subplots_adjust(right=0.92)
# fig.subplots_adjust(left=0.08)
# cbar_ax = fig.add_axes([0.93, 0.05, 0.03, 0.88])#[0.03, 0.18, 0.945, 0.08]#[0.03, 0.08, 0.945, 0.08]
# #cbar_ax = fig.add_axes([0.93, 0.02, 0.03, 0.935])#[0.03, 0.18, 0.945, 0.08]#[0.03, 0.08, 0.945, 0.08]
# fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
# cbar_ax.tick_params(labelsize=16)
# cbar_ax.xaxis.offsetText.set_fontsize(16)
# #ax2.text(105,83,'1e-6',fontsize=16)
# plt.savefig(directory1+'CDI.png')
# plt.savefig(directory1+'CDI.pdf', format='pdf')
# plt.show()


# #%%
# sorted_image=[]

# #sorted_image.append((cube_tot)*1e6)#*maskvisu [100:160,40:160]
# #for i in np.arange(6):
# #sorted_image.append(np.array(new_cube_co)*1e6)
# #for i in np.arange(6):
# #sorted_image.append(np.array(cube_CDI)*1e6)
# sorted_image = np.append(cube_tot,np.array(new_cube_co),axis=0)
# sorted_image = np.append(sorted_image,np.array(cube_CDI),axis=0)
# sorted_image = sorted_image*1e6

# ncol = len(cube_co)

# f1, ax = plt.subplots(nrows = 3, ncols = ncol, sharex = True, sharey = True, figsize = (20,10),
#                         subplot_kw = {'xticks': [], 'yticks': []})
# k = 0
# i=0
# #per=0
# for axes in ax.flat:
#     im1 = axes.imshow((sorted_image[k]*maskvisu)[30:170,30:170], cmap = cmap , vmin = vmin, vmax = vmax)
#     #axes.contour(maskDH[30:170,30:170])
#     #axes.contour(maskCDI[30:170,30:170], colors = 'blue')
#     # if (k%ncol == 0 and i==0) : 
#     #     axes.text(-200,100,'Total Intensity', fontsize = fontsize, rotation = 90) 
#     #     i=i+1
#     # if (k%ncol == 0 and i==1) : 
#     #     axes.text(-200,400,'Reference image', fontsize = fontsize, rotation = 90) 
#     #     i=i+1
#     # if (k%ncol == 0 and i==2) : 
#     #     axes.text(-200,650,'CDI result', fontsize = fontsize, rotation = 90) 
#     #     i=i+1
#     #if k == 1 : axes.set_title('Reference image', fontsize = fontsize)
#     #if k == 2 : axes.set_title('CDI result', fontsize = fontsize)
#     if k in np.arange(ncol) : 
#         print(k)
#         axes.set_title('Iteration '+str(int(k)), fontsize=fontsize)
#     k = k + 1
# plt.tight_layout()
# f1.subplots_adjust(bottom=0.05)
# cbar_ax = f1.add_axes([0.045, 0.02, 0.92, 0.02])#[0.03, 0.18, 0.945, 0.08]#[0.03, 0.08, 0.945, 0.08]
# #cbar_ax = fig.add_axes([0.93, 0.02, 0.03, 0.935])#[0.03, 0.18, 0.945, 0.08]#[0.03, 0.08, 0.945, 0.08]
# f1.colorbar(im1, cax=cbar_ax, orientation='horizontal')
# cbar_ax.tick_params(labelsize=16)
# cbar_ax.xaxis.offsetText.set_fontsize(16)
# plt.savefig(directory1+'CDI_steps.png')
# plt.savefig(directory1+'CDI_steps.pdf', format='pdf')



# #%%
# maskCDI2 = matrices.creatingMaskDH(200,
#                'circle',
#                choosepixDH=[8, 35, -35, 35],
#                circ_rad=[14, 60],#12 for Exp19! 14 or more for Exp20
#                circ_side="Bottom",
#                circ_offset=14, #12 for Exp19! 14 or more for Exp20
#                circ_angle=0)

# Niter=5
# first_image = 0
# div = 2
# rad_contrast_tot = perf.extract_contrast_radial(cube_tot, maskCDI2, ld_p/div, Niter)[1]
# rad_contrast = perf.extract_contrast_radial(cube_CDI, maskCDI2, ld_p/div, Niter)[1]
# Separation = np.arange(rad_contrast.shape[0])/div*ld_mas
# plt.close()
# fig, (ax,ax2) = plt.subplots(2,1, figsize=(6,8), gridspec_kw={'height_ratios': [4, 2]}, sharex=True)

# #ax.fill_between(mas, mini, maxi, alpha=0.5)
# #ax.plot(mas,medi)

# #ax.set_prop_cycle(color=sns.light_palette('orange',Niter+3))


# # for i in np.arange(len(rad_contrast.T)-1):
# #     if i==len(rad_contrast.T)-2: ax.plot(Separation, rad_contrast.T[i], linewidth = 2,label='Iteration '+str(i))
# #     else: ax.plot(Separation, rad_contrast.T[i],linewidth = 2)

# ax.set_prop_cycle(color=sns.light_palette('green',Niter+1))
# for i in np.arange(Niter):
#     if i==Niter-1: ax.plot(Separation, rad_contrast_tot.T[i], color = 'green', linewidth = 2, label = 'Total intensity')
#     else: ax.plot(Separation, rad_contrast_tot.T[i],linewidth = 2)


# ax.set_prop_cycle(color=sns.light_palette('blue',Niter+1))
# for i in np.arange(Niter):
#     if i==Niter-1: ax.plot(Separation, rad_contrast.T[i], color = 'blue', linewidth = 2, label = 'CDI result')
#     else: ax.plot(Separation, rad_contrast.T[i],linewidth = 2)
    

# ax.set_yscale('log')
# ax.set_ylabel('1-$\sigma$ normalized intensity',size=14)
# #ax.grid(which='major', linewidth=1)
# #ax.grid(which='minor', linewidth=0.5)
# ax.legend(fontsize=14)
# ax.tick_params(labelsize=16)
# ax.xaxis.offsetText.set_fontsize(16)

# ax2.set_prop_cycle(color=sns.light_palette('red',Niter+1))
# #ax2.plot(Separation, rad_contrast.T[0]/rad_contrast.T[-2], color = 'red', linewidth=2, label = 'Final Gain')
# for i in np.arange(Niter):
#     if i==Niter-1: ax2.plot(Separation, rad_contrast_tot.T[i]/rad_contrast.T[i], color='red', linewidth=2, label = 'Total Intensity/CDI result')
#     else: ax2.plot(Separation, rad_contrast_tot.T[i]/rad_contrast.T[i], linewidth=2)
# #ax2.set_yticks(np.arange(1,6))#6
# #ax2.set_ylim(1,6)
# ax2.set_yticks(np.arange(1,6))#6
# ax2.set_ylim(1,6)
# ax2.set_ylabel('Gain in performance (x )',size=14)
# #ax2.grid(which='major', linewidth=1)
# ax2.legend(fontsize=14)
# #plt.suptitle('SPHERE Contrast (Exp '+str(Series)+')')

# ax2.tick_params(labelsize=16)
# ax2.xaxis.offsetText.set_fontsize(16)

# plt.xlabel('Angular separation (mas)',size=14)
# plt.xlim(100,800)
# plt.tight_layout()
# plt.savefig(directory1+'Contrast_Separation_CDI.png')
# plt.savefig(directory1+'Contrast_Separation_CDI.pdf', format='pdf')
# plt.show()

