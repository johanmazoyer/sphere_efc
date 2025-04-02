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
target_name = 'HR4796A'
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
    remove = [12, 3] #added 3
    

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


processed_directory = ImageDirectory + 'processed_data/vip/'
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
scan_modes = np.arange(1,len(diff1_removed)+1)
filtered_diff = []
filtered_diff_cube_all = []
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
    filtered_diff_cube_all.append(filtered_diff_cube)
    j=j+1

#Saved the three filtered mean cube_diff
filtered_diff = np.array(filtered_diff)
#Save the three filtered cube_diff for various scanned modes
filtered_diff_cube_all = np.array(filtered_diff_cube_all)

fits.writeto(processed_directory+'filtered_diff.fits', filtered_diff, overwrite=True)
fits.writeto(processed_directory+'filtered_diff_cube_all.fits', filtered_diff_cube_all, overwrite=True)

# The CDI part now starts here
filename = probe_type + '_' + zone_to_correct + '_' + str(int(size_probes/37*37)) + 'nm' + '_'
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

#%% Other possibility with PCA on diff
intensity_co_iter = []
for j in scan_modes: # Loop over the iterations
    intensity_co_modes = []
    for i in scan_modes: #Loop over the filtered modes
        diff = filtered_diff_cube_all[:, i-1, j-1]
        resultatestimation = SPHERE.estimateEab(diff, vectoressai)
        intensity_co_modes.append(np.abs(resultatestimation)**2)

    intensity_co_iter.append(intensity_co_modes)
intensity_co_iter = np.array(intensity_co_iter)
fits.writeto(processed_directory+'intensity_co_iter.fits', intensity_co_iter, overwrite=True)


Rotated_stacked_mfiltered_inc = []
Rotated_stacked_hpfiltered_mfiltered_inc=[]
Cube_mfiltered_inc=[]
for i in scan_modes:
    Cube_mfiltered_inc.append(cube_image_coro_removed - intensity_co_iter[:,i-1])
    u,s,vh = perf.get_cube_svd(cube_image_coro_removed - intensity_co_iter[:,i-1])
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


#%% Test extracting signal in the probe images
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

cube_I1plus_removed = cube_I1plus.copy() 
cube_I2plus_removed = cube_I2plus.copy() 
cube_I3plus_removed = cube_I3plus.copy() 
cube_I1moins_removed = cube_I1moins.copy() 
cube_I2moins_removed = cube_I2moins.copy() 
cube_I3moins_removed = cube_I3moins.copy() 

cube_image_coro_removed = cube_image_coro.copy()
newPA = PA.copy()

for i in remove: 
    diff1_removed = np.delete(diff1_removed, i, axis = 0)
    diff2_removed = np.delete(diff2_removed, i, axis = 0)
    diff3_removed = np.delete(diff3_removed, i, axis = 0)

    cube_I1plus_removed = np.delete(cube_I1plus_removed, i, axis = 0)
    cube_I2plus_removed = np.delete(cube_I2plus_removed, i, axis = 0)
    cube_I3plus_removed = np.delete(cube_I3plus_removed, i, axis = 0)
    cube_I1moins_removed = np.delete(cube_I1moins_removed, i, axis = 0)
    cube_I2moins_removed = np.delete(cube_I2moins_removed, i, axis = 0)
    cube_I3moins_removed = np.delete(cube_I3moins_removed, i, axis = 0)

    cube_image_coro_removed = np.delete(cube_image_coro_removed, i, axis = 0)

    newPA = np.delete(newPA, i, axis = 0)

# Compute the PCA and filter modes in 
filtered_mean_diff = []
for cube_diff in [diff1_removed, diff2_removed, diff3_removed]:

    u,s,vh = perf.get_cube_svd(cube_diff)
    nb_images,isz,isz = vh.shape[0], int(np.sqrt(vh.shape[1])), int(np.sqrt(vh.shape[1])) 
    filtered = s.copy()
    filtered[1:]=0
    a = (u * filtered @ vh).reshape(nb_images,isz,isz)
    #Save the mean of filtered diff cube along the various iterations
    filtered_mean_diff.append( np.mean(a, axis=0 )) #mean

#Saved the three filtered mean cube_diff
filtered_mean_diff = np.array(filtered_mean_diff)

fits.writeto(processed_directory+'filtered_mean_diff.fits', filtered_mean_diff, overwrite=True)

# The CDI part now starts here
filename = probe_type + '_' + zone_to_correct + '_' + str(int(size_probes/37*37)) + 'nm' + '_'
filename = MatrixDirectory + lightsource_estim + filename + 'VecteurEstimation' + '.fits'
vectoressai = fits.getdata(filename)

# Get super EF estimate
resultatestimation = SPHERE.estimateEab(filtered_mean_diff, vectoressai)

filename = probe_type + '_' + zone_to_correct + '_' + str(int(size_probes/37*37)) + 'nm' + '_'
filename = MatrixDirectory + lightsource_estim + filename + 'EF_probe_Real' + '.fits'
real_probe = fits.getdata(filename)
filename = probe_type + '_' + zone_to_correct + '_' + str(int(size_probes/37*37)) + 'nm' + '_'
filename = MatrixDirectory + lightsource_estim + filename + 'EF_probe_Imag' + '.fits'
imag_probe = fits.getdata(filename)

EF_probes = real_probe + 1j*imag_probe


#Compute I+ and I-
Iprobes_co_plus = np.abs(resultatestimation + EF_probes)**2
Iprobes_co_moins = np.abs(resultatestimation - EF_probes)**2

#Remove I+ and I- from the cube of probe images
cube_inco_I1plus = cube_I1plus_removed - Iprobes_co_plus[0]
cube_inco_I2plus = cube_I2plus_removed - Iprobes_co_plus[1]
cube_inco_I3plus = cube_I3plus_removed - Iprobes_co_plus[2]
cube_inco_I1moins = cube_I1moins_removed - Iprobes_co_moins[0]
cube_inco_I2moins = cube_I2moins_removed - Iprobes_co_moins[1]
cube_inco_I3moins = cube_I3moins_removed - Iprobes_co_moins[2]
cube_inco_probes= np.array([cube_inco_I1plus, cube_inco_I2plus, cube_inco_I3plus,cube_inco_I1moins,cube_inco_I2moins,cube_inco_I3moins])

fits.writeto(processed_directory+'cube_inco_probes.fits', cube_inco_probes, overwrite=True)


#Get individual incoherent intensity for each probe
Rotated_stacked_mfiltered_inc = []
Rotated_stacked_hpfiltered_mfiltered_inc=[]
for i in range(len(cube_inco_probes)): #Loop over the probes
    u,s,vh = perf.get_cube_svd(cube_inco_probes[i])
    ADI_result = perf.reduction_ADI(u, s, vh, [0], - newPA) #[0] instead of vector
    Rotated_stacked_mfiltered_inc.append(ADI_result)
    #ADI_hide = remove_center_cube(ADI_result, 30)

    # High-pass filtering of ADI[0] (where the cube has simply been rotated and stacked)
    hp_filtered = []
    for high_pass_filter_cut in np.arange(1,11):
        hp_filtered.append(perf.high_pass_filter(ADI_result[0], high_pass_filter_cut))
    hp_filtered = np.array(hp_filtered)
    Rotated_stacked_hpfiltered_mfiltered_inc.append(hp_filtered)

Rotated_stacked_mfiltered_inc = np.array(Rotated_stacked_mfiltered_inc)
Rotated_stacked_hpfiltered_mfiltered_inc = np.array(Rotated_stacked_hpfiltered_mfiltered_inc) * mask


fits.writeto(processed_directory+'Rotated_stacked_mfiltered_inc_probe.fits', Rotated_stacked_mfiltered_inc, overwrite=True)
fits.writeto(processed_directory+'Rotated_stacked_hpfiltered_mfiltered_inc_probe.fits', Rotated_stacked_hpfiltered_mfiltered_inc, overwrite=True)

#Get mean incoherent intensity over the probes and high pass filter
Rotated_stacked_hpfiltered_mfiltered_inc=[]
mean_result = np.mean(Rotated_stacked_mfiltered_inc,axis=0)

hp_filtered = []
for high_pass_filter_cut in np.arange(1,11):
    hp_filtered.append(perf.high_pass_filter(mean_result[0], high_pass_filter_cut))
Rotated_stacked_hpfiltered_mean_inc = np.array(hp_filtered) * mask

fits.writeto(processed_directory+'Rotated_stacked_hpfiltered_mean_inc_probe.fits', Rotated_stacked_hpfiltered_mean_inc, overwrite=True)


#Get the normal CDI and high pass filter
coro_inco = cube_image_coro_removed - np.abs(resultatestimation)**2

u,s,vh = perf.get_cube_svd(coro_inco)
ADI_result = perf.reduction_ADI(u, s, vh, [0], - newPA) #[0] instead of vector
Rotated_stacked_hpfiltered_normalinc = []
hp_filtered = []
for high_pass_filter_cut in np.arange(1,11):
    hp_filtered.append(perf.high_pass_filter(ADI_result[0], high_pass_filter_cut))
hp_filtered = np.array(hp_filtered)
Rotated_stacked_hpfiltered_normalinc.append(hp_filtered)

Rotated_stacked_hpfiltered_normalinc = np.array(Rotated_stacked_hpfiltered_normalinc) * mask
fits.writeto(processed_directory+'Rotated_stacked_hpfiltered_normalinc.fits', Rotated_stacked_hpfiltered_normalinc, overwrite=True)


#Get the mean of probe inco and coro inco and high pass filter
total_intensity_inco = (mean_result[0] + ADI_result[0])/2 #Could you weight for each
Rotated_stacked_hpfiltered_totalinc = []
hp_filtered = []
for high_pass_filter_cut in np.arange(1,11):
    hp_filtered.append(perf.high_pass_filter(total_intensity_inco, high_pass_filter_cut))
hp_filtered = np.array(hp_filtered)
Rotated_stacked_hpfiltered_totalinc.append(hp_filtered)

Rotated_stacked_hpfiltered_totalinc = np.array(Rotated_stacked_hpfiltered_totalinc) * mask
fits.writeto(processed_directory+'Rotated_stacked_hpfiltered_totalinc.fits', Rotated_stacked_hpfiltered_totalinc, overwrite=True)

#%% Test VIP PCA
from hciplot import plot_frames, plot_cubes  # plotting routines
from multiprocessing import cpu_count
from packaging import version

import vip_hci as vip
vvip = vip.__version__
print("VIP version: ", vvip)
if version.parse(vvip) < version.parse("1.0.3"):
    msg = "Please upgrade your version of VIP"
    msg+= "It should be 1.0.3 or above to run this notebook."
    raise ValueError(msg)


mask = matrices.creatingMaskDH(200,'circle',choosepixDH=[-70, 70, 5, 70], circ_rad=[12, 65], circ_side="Full", circ_offset=0, circ_angle=0)
mask_khi = matrices.creatingMaskDH(200,'circle',choosepixDH=[-70, 70, 5, 70], circ_rad=[45, 75], circ_side="Full", circ_offset=0, circ_angle=0)
newPA = PA.copy()

#Remove unwanted data in cube_co and rotate
cube_co_removed = cube_co.copy() * mask
for i in remove: 
    cube_co_removed = np.delete(cube_co_removed, i, axis = 0)
    newPA = np.delete(newPA, i, axis = 0)
cube_co_rotated = perf.rotate_cube(np.array(cube_co_removed), - newPA)    
fits.writeto(processed_directory+'Rotated_coh.fits', cube_co_rotated, overwrite=True)
#%%
from vip_hci.psfsub import pca
from vip_hci.psfsub import median_sub



im = []
for ncomponent in np.arange(10):
    #pca_cdi_fr = median_sub(cube_tot_removed, -newPA, cube_ref=cube_co_removed, collapse_ref='mean')
    hp_filtered = []
    for high_pass_filter_cut in np.arange(1,3):
        pca_cdi_fr = pca(cube_tot_removed, -newPA, ncomp=ncomponent, cube_ref=cube_co_removed,mask_rdi = mask_khi, sigma_hp = high_pass_filter_cut)
        #hp_filtered.append(perf.high_pass_filter(pca_cdi_fr, high_pass_filter_cut))
        hp_filtered.append(pca_cdi_fr)
    im.append(hp_filtered)
im = np.array(im)*mask

#plot_frames(pca_cdi_fr, grid=True)
fits.writeto(processed_directory+'im.fits',im,overwrite=True)

#%% Test VIP contrast curve
from vip_hci.fm import normalize_psf
from vip_hci.metrics import contrast_curve
from vip_hci.config import VLT_SPHERE_IRDIS
pxscale_naco = VLT_SPHERE_IRDIS['plsc']
print(pxscale_naco, "arcsec/px")
PSF,smoothPSF,maxPSF,exppsf = SPHERE.process_PSF(ImageDirectory,lightsource_estim,centerx-21,centery+17,200)
#plt.imshow(PSF)
psfn, flux, fwhm_sphere = normalize_psf(PSF, size=19, debug=True, full_output=True)
print(flux, maxPSF)


cc_nocdi_1 = contrast_curve(cube_tot_removed * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=0, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = 1)
cc_cdi_1 = contrast_curve(cube_tot_removed * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=1, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = 1)
cc_nocdi_5 = contrast_curve(cube_tot_removed * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=0, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = 3)
cc_cdi_5 = contrast_curve(cube_tot_removed * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=1, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = 3)

#%
plt.figure(figsize=(8,5))
plt.plot(cc_nocdi_1['distance']*pxscale_naco, 
         -2.5*np.log10(cc_nocdi_1['sensitivity_student']), 
         'b--', label='5-sigma contrast (no CDI)', alpha=0.5)
plt.plot(cc_cdi_1['distance']*pxscale_naco, 
         -2.5*np.log10(cc_cdi_1['sensitivity_student']), 
         'r--', label='5-sigma contrast (CDI)', alpha=0.5)
plt.plot(cc_nocdi_5['distance']*pxscale_naco, 
         -2.5*np.log10(cc_nocdi_5['sensitivity_student']), 
         'b-', label='5-sigma contrast (no CDI 5)', alpha=0.5)
plt.plot(cc_cdi_5['distance']*pxscale_naco, 
         -2.5*np.log10(cc_cdi_5['sensitivity_student']), 
         'r-', label='5-sigma contrast (CDI 5)', alpha=0.5)


plt.gca().invert_yaxis()
plt.ylabel('Contrast (mag)')
plt.xlabel('Separation (arcsec)')
_ = plt.legend(loc='best')
plt.show()

#%%
plt.figure(figsize=(8,5))

for i in [1, 3, 5]:
    cc = contrast_curve(cube_tot_removed * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=1, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = i, plot = False)
    cc_nocdi = contrast_curve(cube_tot_removed * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=0, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = i, plot = False)
    plt.plot(cc['distance_arcsec'], 
            -2.5*np.log10(cc['sensitivity_gaussian']), 
         'r-',  alpha=(i+2) * 0.1)
    plt.plot(cc_nocdi['distance_arcsec'], 
            -2.5*np.log10(cc_nocdi['sensitivity_gaussian']), 
         'b-',  alpha=(i+2) * 0.1)



plt.gca().invert_yaxis()
plt.ylabel('Contrast (mag)')
plt.xlabel('Separation (arcsec)')
_ = plt.legend(loc='best')
plt.show()

#%% Forward modeling of HR4796
from vip_hci.fm import ScatteredLightDisk
from vip_hci.fm import cube_inject_fakedisk

pixel_scale=0.01225 # pixel scale in arcsec/px
dstar= 72.8 # distance to the star in pc
nx = 200 # number of pixels of your image in X
ny = 200 # number of pixels of your image in Y

itilt = 76.8 # inclination of your disk in degrees
a = 77 # semimajoraxis of the disk in au 
ksi0 = 1. # reference scale height at the semi-major axis of the disk
gamma = 2. # exponant of the vertical exponential decay
alpha_in = 18
alpha_out = -13
beta = 1 #linear flaring
omega = -72
pa = 28
flux_max = 1.8e-4

g1=0.99
g2=-0.14
weight1=0.83
eccentricity = 0.045


fake_disk4 = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=itilt, omega=omega, pxInArcsec=pixel_scale, pa=pa, flux_max=flux_max,
                                density_dico={'name':'2PowerLaws', 'ain':alpha_in, 'aout':alpha_out,
                                              'a':a, 'e':eccentricity, 'ksi0':ksi0, 'gamma':gamma, 'beta':beta},
                                spf_dico={'name':'DoubleHG', 'g':[g1,g2], 'weight':weight1,
                                          'polar':False},
                                )


fake_disk1_map = fake_disk4.compute_scattered_light()
cube_fake_disk3_convolved = cube_inject_fakedisk(fake_disk1_map, -newPA,
                                                 psf=psfn, imlib='vip-fft')

cube_im_minus_disk = cube_tot_removed - cube_fake_disk3_convolved

plot_frames(fake_disk1_map, grid=False, size_factor=6)
fits.writeto(processed_directory+'tot_minus_disk.fits',cube_im_minus_disk,overwrite=True)

ncomponent = 0
high_pass_filter_cut = 3
pca_cdi_fr = pca(cube_im_minus_disk, -newPA, ncomp=ncomponent, cube_ref=cube_co_removed,mask_rdi = mask_khi, sigma_hp = high_pass_filter_cut)
fits.writeto(processed_directory+'im.fits',pca_cdi_fr,overwrite=True)

#%% Contrast curves after forward modeling
from vip_hci.fm import normalize_psf
from vip_hci.metrics import contrast_curve
from vip_hci.config import VLT_SPHERE_IRDIS
pxscale_naco = VLT_SPHERE_IRDIS['plsc']
print(pxscale_naco, "arcsec/px")
PSF,smoothPSF,maxPSF,exppsf = SPHERE.process_PSF(ImageDirectory,lightsource_estim,centerx-21,centery+17,200)
#plt.imshow(PSF)
psfn, flux, fwhm_sphere = normalize_psf(PSF, size=19, debug=True, full_output=True)
print(flux, maxPSF)

high_pass_filter_cut = 3

cc_nocdi_1 = contrast_curve(cube_im_minus_disk * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=0, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = high_pass_filter_cut)
cc_cdi_1 = contrast_curve(cube_im_minus_disk * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=1, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = high_pass_filter_cut)


#%
plt.figure(figsize=(8,5))
plt.plot(cc_nocdi_1['distance']*pxscale_naco, 
         -2.5*np.log10(cc_nocdi_1['sensitivity_student']), 
         'b--', label='5-sigma contrast (no CDI)', alpha=0.5)
plt.plot(cc_cdi_1['distance']*pxscale_naco, 
         -2.5*np.log10(cc_cdi_1['sensitivity_student']), 
         'r--', label='5-sigma contrast (CDI)', alpha=0.5)


plt.gca().invert_yaxis()
plt.ylabel('Contrast (mag)')
plt.xlabel('Separation (arcsec)')
_ = plt.legend(loc='best')
plt.show()
#%% Contrast curves vs iter
cdi=[]
no_cdi = []
for nb_iter in np.arange(len(cube_im_minus_disk)):
    scan_iter = np.arange(1,len(cube_im_minus_disk)-1-nb_iter)[::-1]

    cube_tot_iters = cube_im_minus_disk.copy() #* mask
    cube_co_iters = cube_co_removed.copy() * mask

    newPA_iters = newPA.copy()
    for i in scan_iter:
        cube_tot_iters = np.delete(cube_tot_iters, i, axis = 0)
        cube_co_iters = np.delete(cube_co_iters, i, axis = 0)
        newPA_iters = np.delete(newPA_iters, i, axis = 0)

    
    cdi.append(contrast_curve(cube_tot_iters * maxPSF, -newPA_iters, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=1, cube_ref=cube_co_iters, mask_rdi = mask_khi, sigma_hp = high_pass_filter_cut))
    
    no_cdi.append(contrast_curve(cube_tot_iters * maxPSF, -newPA_iters, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=0, cube_ref=cube_co_iters, mask_rdi = mask_khi, sigma_hp = high_pass_filter_cut))
    


#%%
from matplotlib.lines import Line2D

plt.figure(figsize=(8,5))
#for nb_iter in np.arange(len(cube_im_minus_disk)):
lines= ['--' , ':' ,'-.']
for nb_iter in [0,1,2]:
    plt.plot(cdi[nb_iter]['distance_arcsec'], 
            -2.5*np.log10(cdi[nb_iter]['sensitivity_student']), 
         'r', linestyle=lines[nb_iter], alpha=1)
    plt.plot(no_cdi[nb_iter]['distance_arcsec'], 
            -2.5*np.log10(no_cdi[nb_iter]['sensitivity_student']), 
         'b', linestyle=lines[nb_iter], alpha=1)

line = [Line2D([0], [0], linestyle= '--', label='iter 1', color='k')]
line.append(Line2D([0], [0], linestyle= ':', label='iter 2', color='k'))
line.append(Line2D([0], [0], linestyle= '-.', label='iter 3', color='k'))

plt.gca().invert_yaxis()
plt.ylabel('Contrast (mag)')
plt.xlabel('Separation (arcsec)')
_ = plt.legend(handles= line, loc='best')
plt.show()


#%% Throughput calculation
plt.figure(figsize=(8,5))

for i in [1, 3, 5, 7, None][::-1]:
    cc = contrast_curve(cube_im_minus_disk * maxPSF, -newPA, psfn, fwhm=fwhm_sphere, pxscale=pxscale_naco, starphot=flux, 
                        sigma=5, nbranch=3, algo=pca, ncomp=1, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = i, plot = False)
    if i == None:
        lab = 'No high-pass filter'
        alph =1 
        color = 'black'
        lin = '-'
    else:
        lab = 'sigma = ' + str(i)
        alph = (i) * 1/7 
        color = 'red'
        lin = '-'
    plt.plot(cc['distance_arcsec'], 
            (cc['throughput']), 
         color=color,  alpha = alph, linestyle= lin, label= lab)


plt.ylabel('Throughput')
plt.xlabel('Separation (arcsec)')
_ = plt.legend(loc='lower right')
plt.show()


#%% STIM maps (not working)
ncomponent = 9
high_pass_filter_cut = 3

from vip_hci.metrics import stim_map
pca_img, _,_, pca_res, pca_res_der = pca(cube_tot_removed, -newPA, ncomp=ncomponent, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = high_pass_filter_cut, full_output=True , verbose=False, imlib='skimage', interpolation='biquartic')

stim_map = stim_map(pca_res_der*mask)

from vip_hci.metrics import inverse_stim_map
inv_stim_map = inverse_stim_map(pca_res*mask, -newPA)

if version.parse(vvip) < version.parse("1.6.0"): 
    norm_stim_map = stim_map/np.nanmax(inv_stim_map)
else:
    from vip_hci.metrics import normalized_stim_map
    norm_stim_map = normalized_stim_map(pca_res*mask, -newPA)

plot_frames((stim_map, inv_stim_map, norm_stim_map), grid=True, 
            label=('STIM map', 'inv. STIM map', 'norm. STIM map'))
#plot_frames(( norm_stim_map), grid=True, 
#            label=('norm. STIM map'))

thr_stim_map = norm_stim_map.copy()
thr_stim_map[np.where(thr_stim_map<1)]=0

plot_frames((pca_img, thr_stim_map), grid=True, 
            label=('PCA image (npc=15)', 'thresholded norm. STIM map'))

#%%
ncomponent = 1
high_pass_filter_cut = 3
from vip_hci.metrics import snrmap
pca_img = pca(cube_tot_removed, -newPA, ncomp=ncomponent, cube_ref=cube_co_removed, mask_rdi = mask_khi, sigma_hp = high_pass_filter_cut, verbose=False)#, imlib='skimage', interpolation='biquartic')
#plot_frames(pca_img, grid=True)
snrmap_1 = snrmap(pca_img*mask, fwhm=fwhm_sphere, plot=True, approximated=True)
plot_frames(pca_img*mask)

