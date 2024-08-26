# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:35:50 2021

@author: apotier
"""

import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from matplotlib.colors import LogNorm
import copy
from natsort import natsorted
from astroplan import Observer, FixedTarget
from scipy import ndimage
import astropy.units as u
from scipy.optimize import fmin_powell as fmin_powell
import SPHERE_EFC_Func as SPHERE
import Definitions_for_matrices as matrices


def custom_plot(pup, img, vmin, vmax , norm = None):
    ''' --------------------------------------------------
    Plots two images next to each other.
    -------------------------------------------------- '''
    cmap=copy.copy(matplotlib.cm.jet)
    cmap=copy.copy(matplotlib.cm.get_cmap('Blues_r'))
    cmap.set_bad('black',1.)
    f1 = plt.figure(1, figsize=(12,6))#(12,3)#(12,6)
    f1.clf()
    ax1 = f1.add_subplot(121)
    if norm == 'log':
        ax1.imshow(pup, cmap=cmap, norm=LogNorm(vmin, vmax))
    else:
        ax1.imshow(pup, cmap=cmap, vmin = vmin, vmax = vmax)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = f1.add_subplot(122)
    if norm == 'log':
        im2 = ax2.imshow(img, cmap=cmap, norm=LogNorm(vmin, vmax))
    else:
        im2 = ax2.imshow(img, cmap=cmap, vmin = vmin, vmax = vmax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    #ax2.imshow(img, vmin = np.amin(pup), vmax = np.amax(pup), cmap="hot")
    f1.tight_layout()
    f1.subplots_adjust(bottom=0.2)#0.3#0.2
    cbar_ax = f1.add_axes([0.03, 0.08, 0.945, 0.08])#[0.03, 0.18, 0.945, 0.08]#[0.03, 0.08, 0.945, 0.08]
    cbar = f1.colorbar(im2, cax=cbar_ax, orientation='horizontal')
    cbar_ax.tick_params(labelsize=13)
    cbar_ax.xaxis.offsetText.set_fontsize(13)
    #cbar.set_label('Nanometers', fontsize=14)

    
    
def rms(data):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    nb_pix=data.size
    return np.sqrt(np.sum(data**2)/nb_pix)

def cropimage(img, ctr_x, ctr_y, newsizeimg_x,newsizeimg_y):
    """ --------------------------------------------------
    Crop an image
    
    Parameters:
    ----------
    img: 2D array, image to crop
    ctr_x: int, center of the cropped image in the x direction
    ctr_y: int, center of the cropped image in the y direction
    newsizeimg: int, size of the new image in x and y direction (same dimentsion for both)

    Return:
    ------
    cropped: 2D array, cropped image
    -------------------------------------------------- """
    newimgs2_x = newsizeimg_x / 2
    newimgs2_y = newsizeimg_y / 2
    cropped = img[int(ctr_x-newimgs2_x):int(ctr_x+newimgs2_x),int(ctr_y-newimgs2_y):int(ctr_y+newimgs2_y)]
    return cropped


def contrast_radial(image,scoring_reg,chiffre):
    ''' --------------------------------------------------
    Determine contrast rms in one image, radially in the Dark Hole
    -------------------------------------------------- '''
    #image = np.abs(image)
    isz = len(image)
    xx, yy = np.meshgrid(np.arange(isz)-(isz)/2, np.arange(isz)-(isz)/2)
    rr     = np.hypot(yy, xx)
    nb_meas = int(isz/2/chiffre)
    contrast = np.zeros((nb_meas,3))
    k=0
    for i in chiffre*np.arange(nb_meas):
        whereimage = scoring_reg.copy()
        whereimage[np.where(rr<i-chiffre/2)] = 0
        whereimage[np.where(rr>=i+chiffre/2)] = 0
        contrast_mean = np.nanmean(image[np.where(whereimage)])
        contrast_std = np.nanstd(image[np.where(whereimage)])
        contrast_rms = rms(image[np.where(whereimage)])
        
        contrast[k] = [contrast_mean, contrast_std, contrast_rms]
        k=k+1
    return contrast

def extract_contrast_radial(cubeimage, scoring_region, ring_size, nb_iter):
    """
    

    Parameters
    ----------
    cubeimage : TYPE
        DESCRIPTION.
    scoring_region : TYPE
        DESCRIPTION.
    ring_size : TYPE
        DESCRIPTION.
    nb_iter : TYPE
        DESCRIPTION.

    Returns
    -------
    rad_contrast : TYPE
        DESCRIPTION.

    """
    #nb_iter = len(cubeimage)#-1
    rad_contrast = []
    for i in np.arange(nb_iter):
        rad_contrast.append(contrast_radial(cubeimage[i], scoring_region, ring_size))
    rad_contrast =np.array(rad_contrast).T
    return rad_contrast


def contrast_global(image,scoring_reg):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    scoring_reg : TYPE
        DESCRIPTION.

    Returns
    -------
    contrast_mean : TYPE
        DESCRIPTION.
    contrast_std : TYPE
        DESCRIPTION.
    contrast_rms : TYPE
        DESCRIPTION.

    """
    #image = np.abs(image)
    contrast_mean = np.nanmean(image[np.where(scoring_reg)])
    contrast_std = np.nanstd(image[np.where(scoring_reg)])
    contrast_rms = rms(image[np.where(scoring_reg)])
    return contrast_mean, contrast_std, contrast_rms
    
def extract_contrast_global(cubeimage, scoring_region, nb_iter):
    """
    

    Parameters
    ----------
    cubeimage : TYPE
        DESCRIPTION.
    scoring_region : TYPE
        DESCRIPTION.
    nb_iter : TYPE
        DESCRIPTION.

    Returns
    -------
    contrast : TYPE
        DESCRIPTION.

    """
    #nb_iter = len(cubeimage)#-1
    contrast = []
    for i in np.arange(nb_iter):
        contrast.append(contrast_global(cubeimage[i], scoring_region))
    contrast =np.array(contrast).T
    return contrast

def CubeFits(docs_dir):
    ''' --------------------------------------------------
    Load all the fits image in a directory
    
    Parameters:
    ----------
    doc_dir: Input directory
    
    Return:
    ------
    image_array: numpy array
    -------------------------------------------------- '''
    image_list = []
    for filename in natsorted(glob.glob(docs_dir+'*.fits')):
        print(filename)
        image=fits.getdata(filename)
        image_list.append(image)
        
    image_array = np.array(image_list)
    return image_array


def image_for_gif(fileimage,text_to_add,ld_p,vmin,vmax,title):
    # Data for plotting
    
    isz=len(fileimage)
    #ld_p=24.7
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(np.abs(fileimage),
                   norm=LogNorm(vmin=vmin,vmax=vmax),
                   #vmin=vmin,vmax=vmax,
                   cmap='Blues_r',
                   extent=[-isz/(2*ld_p),isz/(2*ld_p),-isz/(2*ld_p),isz/(2*ld_p)],interpolation='none')
    #ax.set_xlim(-20,20)
    ax.set_ylim(0,20)
    #ax.set_ylim(-20,0)
    ax.text(0,2,text_to_add,size=20, ha='center',weight='heavy')
    ax.set_xlabel('$\lambda/D$', size = 20)
    ax.set_ylabel('$\lambda/D$', size = 20)
    ax.set_title(title, size=20)
    # IMPORTANT ANIMATION CODE HERE
    # Used to keep the limits constant
    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.03,right=0.9)
    #cbar_ax = fig.add_axes([0.1, 0.1, 0.87, 0.05])
    cbar_ax = fig.add_axes([0.15, 0.1, 0.77, 0.05])
    cbar_ax.tick_params(labelsize=15)#20
    cbar = fig.colorbar(im, cax=cbar_ax,aspect=5, orientation='horizontal')
    #cbar.ax.set_xticklabels(["{:.0e}".format(i) for i in cbar.get_ticks()])
    cbar.ax.set_xlabel("Normalized Intensity", size=20)
    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    ax.tick_params(labelsize=20)
    plt.close(fig)
    return image


def PA_on_detector(target_name, time_now, estimated_onsky_PA_of_the_planet, verbose=False):
    """ --------------------------------------------------
    Measure the planet position in SPHERE detector at the given time

    Author: J Mazoyer
    
    Parameters:
    ----------
    target_name: string
        name of the object

    time_now: Time object (astropy)
        Time of observation

    estimated_onsky_PA_of_the_planet: float degrees
        Estimated position angle of the planet in degrees
        
    verbose : bool
        if True print values and intermediate

    Return:
    
    Position angle of the planet

    -------------------------------------------------- """

    vlt = Observer.at_site('Paranal Observatory', timezone="UTC")
    PUPOFFSET = 135.87  # ±0.03 deg
    True_North = -1.75  #+/-0.08 deg (https://arxiv.org/pdf/1609.06681.pdf)

    target = FixedTarget.from_name(target_name)

    PARANGLE_deg = vlt.parallactic_angle(time_now, target).to_value(u.deg)

    # the values actually written in the rot sphere files are
    # -PARANGLE_deg - PUPOFFSET - True_North

    ### zeformula given by Gael
    ### PA_onsky = PA_detector + PARANGLE_deg + True_North + PUPOFFSET

    PA_detector = estimated_onsky_PA_of_the_planet - PARANGLE_deg - True_North - PUPOFFSET

    if verbose:
        print("")
        print("")
        print("target: ", target_name)
        print("planet estimated Postion angle is:", estimated_onsky_PA_of_the_planet)
        print("at observation time: ", time_now)
        # print("parralactic angle given in SPHERE reduc parang files (to check): ", round(-PARANGLE_deg - PUPOFFSET - True_North,4))
        print("PA on detector angle is: ", round(PA_detector, 2))
        print("")
        print("")
    return PA_detector



def high_pass_filter(image, sigma):
    lowpass = ndimage.gaussian_filter(image, sigma)
    image = image - lowpass
    return image



def rescale_CDI(signal_co, signal_tot, maskCDI):
    filtered_co = high_pass_filter(signal_co, 2)
    filtered_tot = high_pass_filter(signal_tot, 2)

    best_params=[]
    best_params3=[]
    for i in np.arange(10):
    #     def cost_function(xy_trans):
    #         unshifted = SPHERE.fancy_xy_trans_slice(filtered_co, xy_trans)
    #         return np.sum(np.abs(filtered_tot[np.where(maskCDI==1)] - unshifted[np.where(maskCDI==1)])**2*1e5)#SPHERE.correl_mismatch(H3[np.where(mask==0)], unshifted[np.where(mask==0)])
            
        
    #     best_params.append(fmin_powell(cost_function, [0, 0], disp=0, callback=None)) #callback=my_callback
    #     #print(best_params[-1])
    #     #print(cost_function([0, 0]),cost_function(best_params[-1]))
    #     filtered_co = SPHERE.fancy_xy_trans_slice(filtered_co, best_params[-1])
    
        
        def cost_function3(factor):
            a = filtered_co * factor
            return np.sum(np.abs(filtered_tot[np.where(maskCDI==1)] - a[np.where(maskCDI==1)])*1e5)
        
        best_params3.append(fmin_powell(cost_function3,1, disp=0, callback=None))
        filtered_co = filtered_co*best_params3[-1]
    print(np.prod(best_params3))
    #weight = np.sum(np.abs(filtered_tot[np.where(maskCDI==1)] - filtered_co[np.where(maskCDI==1)])*1e5)
    #print(weight)
    
    
    for i in np.arange(len(best_params)):
        signal_co = SPHERE.fancy_xy_trans_slice(signal_co, best_params[i])
        signal_co = signal_co*best_params3[i]
        
    return signal_co



def rescale_radial_CDI(signal_co, signal_tot, maskCDI, chiffre):
    filtered_co = high_pass_filter(signal_co, 2)
    filtered_tot = high_pass_filter(signal_tot, 2)

    best_params=[]
    best_params3=[]
    
    isz = len(signal_co)
    xx, yy = np.meshgrid(np.arange(isz)-(isz)/2, np.arange(isz)-(isz)/2)
    rr     = np.hypot(yy, xx)
    nb_meas = int(isz/2/chiffre)
    

    def cost_function(xy_trans):
        unshifted = SPHERE.fancy_xy_trans_slice(filtered_co, xy_trans)
        return np.sum(np.abs(filtered_tot[np.where(maskCDI==1)] - unshifted[np.where(maskCDI==1)])**2*1e5)#SPHERE.correl_mismatch(H3[np.where(mask==0)], unshifted[np.where(mask==0)])
    
    trans = fmin_powell(cost_function, [0, 0], disp=0, callback=None)
    #print(trans)
    if np.sqrt(trans[0]**2+trans[1]**2)>0: trans = [0,0]      
    
    signal_co = SPHERE.fancy_xy_trans_slice(signal_co, trans)
    filtered_co = SPHERE.fancy_xy_trans_slice(filtered_co, trans)

    CDI_result = signal_co.copy()

    for i in chiffre*np.arange(nb_meas):
        whereimage = maskCDI.copy()
        whereimage[np.where(rr<i-chiffre/2)] = 0
        whereimage[np.where(rr>=i+chiffre/2)] = 0
        
    
        def cost_function3(factor):
            a = filtered_co * factor
            return np.sum(np.abs(filtered_tot[np.where(whereimage==1)] - a[np.where(whereimage==1)])**2)
        
        fact = fmin_powell(cost_function3, 1 , disp=0, callback=None)
        if fact<0: fact = [0]
        best_params3.append(fact)
        #print(fact)
        whereimage = np.ones_like(maskCDI)
        whereimage[np.where(rr<i-chiffre/2)] = 0
        whereimage[np.where(rr>=i+chiffre/2)] = 0
        CDI_result[np.where(whereimage==1)] = signal_co[np.where(whereimage==1)] * fact
    
    
    # for i in np.arange(len(best_params3)):
    #     #signal_co = SPHERE.fancy_xy_trans_slice(signal_co, best_params[i])
    #     CDI_result = signal_co*best_params3[i]
        
    return CDI_result
    

def low_pass_filter(image, ld_p):
    puis = matrices.goto_pupil(image)
    isz = len(image)
    pupil1 = SPHERE.roundpupil(isz, isz/2/ld_p)
    enfin = np.real(matrices.goto_focal(puis*pupil1))
    return enfin

def rotate_cube(cube, angles):
    nb_images = len(cube)
    rotated_cube = []#np.zeros_like(cube)
    for i in np.arange(nb_images):
        rotated_cube.append(ndimage.rotate(cube[i], -angles[i], reshape = False, order = 1, prefilter = False))
    rotated_cube = np.array(rotated_cube)
    return rotated_cube

def cube_svd_filtering(u, s, vh, nb_modes_filtered):
    nb_images,isz,isz = vh.shape[0], int(np.sqrt(vh.shape[1])), int(np.sqrt(vh.shape[1])) 
    filtered = s.copy()
    filtered[:nb_modes_filtered]=0
    #filtered[20-nb_modes_filtered:]=0
    new_cube = (u * filtered @ vh).reshape(nb_images,isz,isz)
    return new_cube

def remove_center(image, rad):
    im = image.copy()
    dim = len(image)
    pup = SPHERE.roundpupil(dim, rad)
    im[np.where(pup)] = 0
    return im

def subtract_mean_in_patches(imag,size_patch):
    dim_x, dim_y = imag.shape
    img=imag.copy()
    kernel = np.ones((size_patch,size_patch))/size_patch**2
    mean_convolve = ndimage.convolve(img, kernel)
    img = img - mean_convolve
    return img

def filter_raw_images(cube, angles, stand_thres):
    print('Filtering raw images')
    stand = []
    for i in np.arange(len(cube)):
        stand.append(np.std(cube[i]))
    stand = np.array(stand)
    cube = cube[np.where(stand<stand_thres)]
    angles = angles[np.where(stand<stand_thres)]
    
    return cube, angles

def substract_mean_images(cube, thres):
    print('Subtract mean images')
    nb_images = len(cube)
       
    high_pass_cube = np.zeros_like(cube) 
    for k in np.arange(nb_images):
        high_pass_cube[k] = cube[k] - ndimage.gaussian_filter(cube[k], thres)#10
        #high_pass_cube[k] = subtract_mean_in_patches(cube[k],10)#10
        
    return high_pass_cube





def crop_cube(cube, size):
    print('Cropping images')
    nb_images, dimx, dimy = cube.shape
    ctr_x = int(dimx/2)
    ctr_y = int(dimy/2)
       
    cropped_cube = np.zeros((nb_images, size, size))
    for k in np.arange(nb_images):
        cropped_cube[k] = cropimage(cube[k], ctr_x, ctr_y, size)
    
    return cropped_cube
    
    

def get_cube_svd(cube):
    print('Calculating SVD')
    nb_images = len(cube)
    isz = cube.shape[2]
    #% New basis
    u, s, vh = np.linalg.svd(cube.reshape(nb_images,isz**2),full_matrices=False)
    #u, s, vh = scp_lin.svd(cube.reshape(nb_images,isz**2),full_matrices=False, overwrite_a=True,lapack_driver='gesvd')
    return u, s, vh

def reduction_ADI(u, s, vh, vector_regul, angles):
    print('ADI reduction...')
    isz = int(np.sqrt(vh.shape[1]))
    nb = len(vector_regul)
    
    ADI_result = []
    for i in vector_regul:
        print(i)
        cube_filtered = cube_svd_filtering(u, s, vh, i)
        rotated_cube = rotate_cube(cube_filtered, angles)
        #ADI_result.append(np.quantile(rotated_cube,0.3,axis = 0,method='lower'))#mean?
        ADI_result.append(np.median(rotated_cube,axis = 0))#mean?
    
    ADI_result = np.array(ADI_result)
    
    
        
    return ADI_result

def gaussian_filter_cube(cube, sigma):
    print('Gaussian filtering')
    nb_image = len(cube)
    filtered_cube = np.zeros_like(cube)
    for k in np.arange(nb_image):
        filtered_cube[k] =  ndimage.gaussian_filter(cube[k], sigma)#3 or 5 #low_pass_filter(cube[k], 4)
    
    return filtered_cube


def remove_center_cube(cube, radius_mask):
    print('Removing center')
    filtered_cube = np.zeros_like(cube)
    for k in np.arange(len(cube)):
        filtered_cube[k] = remove_center(cube[k], radius_mask )#30
    return filtered_cube

def extract_normalized_data(file_path):
    data = fits.getdata(file_path)
    ITIME = fits.getval(file_path, 'ITIME')
    COADDS = fits.getval(file_path, 'COADDS')
    return data/ITIME/COADDS

    
def translationFFT(dim_im,a,b):
    """ --------------------------------------------------
    Create a phase ramp of size (dim_im,dim_im) that can be used as follow
    to shift one image by (a,b) pixels : shift_im = real(fft(ifft(im)*exp(i phase ramp)))
    
    Parameters
    ----------
    dim_im : int
        Size of the phase ramp (in pixels)
    a : float
        Shift desired in the x direction (in pixels)
    b : float
        Shift desired in the y direction (in pixels)
    
    Returns
    ------
    masktot : 2D array
        Phase ramp
    -------------------------------------------------- """
    # Verify this function works
    maska = np.linspace(-np.pi * a, np.pi * a, dim_im, endpoint = False)
    maskb = np.linspace(-np.pi * b, np.pi * b, dim_im, endpoint = False)
    xx, yy = np.meshgrid(maska, maskb)
    return np.exp(-1j * xx) * np.exp(-1j * yy)


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def resize_PSF(PSF, new_dim):
    PSF = np.pad(PSF.copy(), int((new_dim - len(PSF))/2), pad_with)
    return PSF

def pol2cart(rho, phi_deg):
    phi_rad = np.pi + phi_deg * np.pi / 180
    x = rho * np.cos(phi_rad)
    y = rho * np.sin(phi_rad)
    return(x, y)
    

def inject_PSF(PSF, contrast, rho, theta):
    translated_PSF = PSF * contrast
    (a, b) = pol2cart(rho, theta)
    translated_PSF = ndimage.shift(translated_PSF, (-b, -a), order=1, mode='constant', cval=0.0, prefilter=False)
    #phase_ramp = translationFFT(len(PSF),a,b)
    #translated_PSF =  np.real(goto_focal( goto_pupil(translated_PSF)*phase_ramp))
    return translated_PSF

def inject_PSF_cube(cube, angles, PSF, contrast, rho, theta):
    print('Injecting PSF in cube')
    injected_cube = []#np.zeros_like(cube)
    
    for k in np.arange(len(cube)):
        #injected_cube[k] = cube[k] + inject_PSF(PSF, contrast, rho, theta - angles[k])
        injected_cube.append(inject_PSF(PSF, contrast, rho, theta - angles[k]))
    
    injected_cube = np.array(injected_cube)
    
    injected_cube = injected_cube + cube
        
        
    return injected_cube


def process_image(ImageDirectory,filename,center,nbiter,ND,target_name,estimated_onsky_PA_of_the_planet, remove_bad_pix = True,high_pass_filter=True):  
    lightsource_estim = 'VLTPupil_'+'FQPM_'
    dimimages = 200
    #centeringateachiter = param['centeringateachiter']
    directory = ImageDirectory + filename
    centerx = center[0]
    centery = center[1]

    #PSF
    PSF,smoothPSF,maxPSF,exppsf = SPHERE.process_PSF(ImageDirectory,lightsource_estim,centerx,centery,dimimages)
    
    #Correction
    
    #Traitement de l'image de référence (première image corono et recentrage subpixelique)
    fileref = SPHERE.last(directory+'iter0_coro_image*.fits')
    imageref = SPHERE.reduceimageSPHERE(fileref,ImageDirectory,maxPSF,int(centerx),int(centery),dimimages,exppsf,ND,remove_bad_pix = remove_bad_pix,high_pass_filter=high_pass_filter)
    imageref = SPHERE.fancy_xy_trans_slice(imageref, [centerx-int(centerx), centery-int(centery)])
    #imageref=cropimage(imageref,int(dimimages/2),int(dimimages/2),int(dimimages/2))
    filecorrection = SPHERE.last(directory+'iter'+str(nbiter-2)+'_coro_image*.fits')
    time_now = fits.getval(filecorrection,'DATE-OBS')
    PA = PA_on_detector(target_name, time_now, estimated_onsky_PA_of_the_planet, verbose=False)
    imagecorrection = SPHERE.reduceimageSPHERE(filecorrection, ImageDirectory, maxPSF,int(centerx),int(centery),dimimages,exppsf,ND,remove_bad_pix = remove_bad_pix,high_pass_filter=high_pass_filter)
    #imagecorrection2 = cropimage(imagecorrection,int(dimimages/2),int(dimimages/2),int(dimimages/2))
    
    def cost_function(xy_trans):
        # Function can use image slices defined in the global scope
        # Calculate X_t - image translated by x_trans
        unshifted = SPHERE.fancy_xy_trans_slice(imagecorrection, xy_trans)
        mask = SPHERE.roundpupil(dimimages,67)
        #Return mismatch measure for the translated image X_t
        return SPHERE.correl_mismatch(imageref[np.where(mask==0)], unshifted[np.where(mask==0)])
   
    best_params = fmin_powell(cost_function, [0, 0], disp=0, callback=None) #callback=my_callback
    
    imagecorrection = SPHERE.fancy_xy_trans_slice(imagecorrection, best_params)
    return imagecorrection, PA
    
