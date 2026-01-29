# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:51:44 2022

@author: apotier
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import scipy.ndimage as snd
from scipy.optimize import fmin_powell as fmin_powell
import scipy.optimize as opt
from scipy import ndimage
import glob
import os

import charis
from spherical.pipeline.pipeline_config import defaultIFSReduction
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm

#Unit conversion and normalisation
#influence matrix normalization = defoc meca en rad @ 632 nm
rad_632_to_nm_opt = 632/2/np.pi

def SHslopes2map(MatrixDirectory, slopes, visu=True):
    """
    Takes in input a 1D vector of 2480 elements, map it to the SH WFS and returns
    2 maps of the slopes in x and y
    """
    if len(slopes) != 2480:
        raise IOError('The input vector must have 2480 elements (currently {0:d})'.format(len(slopes)))
    mapx = np.ndarray((40,40),dtype=float)*np.nan
    mapy = np.ndarray((40,40),dtype=float)*np.nan
    shackgrid = fits.getdata(MatrixDirectory+'shack.grid.fits')
    mapx[shackgrid>0] = slopes[np.arange(1240,dtype=int)*2]
    mapy[shackgrid>0] = slopes[np.arange(1240,dtype=int)*2+1]
    mapx = np.fliplr(np.rot90(mapx,-3))
    mapy = np.fliplr(np.rot90(mapy,-3))
    if visu: 
        fig, ax = plt.subplots(1,2)
        im = ax[0].imshow(mapx, cmap='CMRmap', origin='lower',interpolation='nearest',\
            vmin=np.nanmin(slopes),vmax=np.nanmax(slopes))
        ax[0].set_title('SH slopes X')
        ax[1].imshow(mapy, cmap='CMRmap', origin='lower',interpolation='nearest',\
            vmin=np.nanmin(slopes),vmax=np.nanmax(slopes))
        ax[1].set_title('SH slopes Y')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    return mapx,mapy


def SaveFits(image, head,doc_dir2, name):
    """ --------------------------------------------------
    Save fits file
    
    Parameters:
    ----------
    image: float, file to save
    head: list, header
    doc_dir2: str, directory where to save the file
    name: str, name of the saved file
    -------------------------------------------------- """
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdr = hdul[0].header
    hdr.set(head[0],head[1])
    hdu.writeto(doc_dir2+name+'.fits',overwrite=True)
    
def roundpupil(nbpix, prad1):
    """ --------------------------------------------------
    Create a round pupil (binary mask)
    
    Parameters:
    ----------
    nbpix: int, number of pixels for the 2D array created
    prad1: int, radius of the pupil created in pixels 

    Return:
    ------
    pupilnormal: 2D array, binary mask
    -------------------------------------------------- """
    xx, yy = np.meshgrid(np.arange(nbpix)-(nbpix)/2, np.arange(nbpix)-(nbpix)/2)
    rr     = np.hypot(yy, xx)
    pupilnormal = np.zeros((nbpix,nbpix))
    pupilnormal[rr<=prad1] = 1.
    return pupilnormal

def estimate_efield(Difference, PWP_matrix_per_wvl):
    """ --------------------------------------------------
    Estimate focal plane electric field with PW probing
    
    Parameters:
    ----------
    Difference: Difference of PW images
    PWP_matrix_per_wvl: Matrix PW

    Return:
    ------
    Result: 2D array, complex, focal plane electric field
    -------------------------------------------------- """

    dimimages = Difference.shape[1]
    Difference = Difference.reshape((Difference.shape[0], dimimages ** 2))
    Result_ = np.einsum('ijk,ik->ij', PWP_matrix_per_wvl, Difference.T)
    Result = Result_[:, 0] + 1j * Result_[:, 1]
    Result = Result.reshape(dimimages, dimimages)

    return Result/4
       
   
def solutiontocorrect(mask, ResultatEstimate, invertG, WhichInPupil):
    """ --------------------------------------------------
    Solution in nanometer to dig the DH at next iteration
    
    Parameters:
    ----------
    mask: 2D array, DH region
    ResultatEstimate: 2D array, focal plane elecric field
    invertG: inverted jacobian

    Return:
    ------
    solition: 1D array, floats with nanometers
    -------------------------------------------------- """
    Eab = np.zeros(2*int(np.sum(mask)))
    Resultatbis = (ResultatEstimate[np.where(mask==1)])
    Eab[0:int(np.sum(mask))] = np.real(Resultatbis).flatten()     
    Eab[int(np.sum(mask)):] = np.imag(Resultatbis).flatten()
    cool = np.dot(invertG,Eab)
    
    solution = np.zeros(int(1377))
    solution[WhichInPupil] = cool
    return solution
    
    
def fancy_xy_trans_slice(img_slice, xy_trans):
    """ Return copy of `img_slice` translated by `x_vox_trans` pixels
    
    Parameters
    
    img_slice : array shape (M, N)
    2D image to transform with translation `x_vox_trans`
    x_vox_trans : float
    Number of pixels (pixels) to translate `img_slice`; can be
    positive or negative, and does not need to be integer value.
    """
    if img_slice.ndim == 3:
        trans = []
        for wvl in np.arange(len(img_slice)):
            trans.append(snd.affine_transform(img_slice[wvl], [1, 1], xy_trans, order=1))

        trans_slice = np.array(trans)

    else:
        #Resample image using bilinear interpolation (order=1)
        trans_slice = snd.affine_transform(img_slice, [1, 1], xy_trans, order=1)

    return trans_slice

def my_callback(params):
    print("Trying parameters " + str(params), flush=True)


def correl_mismatch(slice0, slice1):
    """ Negative correlation between the two images, flattened to 1D """
    correl = np.corrcoef(slice0.ravel(), slice1.ravel())[0, 1]
    return -correl
    

def last(files):
    """
    Extract last file with same names in folder

    Parameters
    ----------
    files : File name to extract

    Returns
    -------
    extract : last file in stack

    """
    extract = sorted(glob.glob(files))[-1]
    return extract


def cropimage(img, ctr_x, ctr_y, newsizeimg):
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
    newimgs2 = newsizeimg / 2
    cropped = img[int(ctr_x-newimgs2):int(ctr_x+newimgs2),int(ctr_y-newimgs2):int(ctr_y+newimgs2)]
    return cropped


def get_exptime(file):
    """
    Extract exposure time in an image

    Parameters
    ----------
    file : image file

    Returns
    -------
    exptime : Exposure time in second

    """
    exptime = int(round(fits.getval(file,'EXPTIME')))
    return exptime


# def FindNoisyPix(data,neighborhood_size,threshold):
#     """
#     Find noisy pixels in image an image

#     Parameters
#     ----------
#     data : dark image
#     neighborhood_size : typical distance between bad pixels
#     threshold : defined threshold

#     Returns
#     -------
#     hotpixmap : 2D map filled with 1 at hot pixel location

#     """

#     def std_dev_data(x):
#         return np.nanstd(x)

#     def mean_data(x):
#         return np.nanmean(x)

#     if neighborhood_size % 2 == 0:
#         raise Exception("please use an odd number of neighborhood_size")

#     # this is the real sigma filter that Johan coded. This is actually long and not great
#     # hotpixmap = data*0 
#     # footprint = np.ones((neighborhood_size, neighborhood_size))
#     # footprint[neighborhood_size // 2, neighborhood_size // 2] = 0

#     # data_sigm = ndimage.generic_filter(data, std_dev_data, footprint=footprint)
#     # data_mean = ndimage.generic_filter(data, mean_data, footprint=footprint)
#     # hotpixwh = np.where(np.abs(data - data_mean) > (threshold * data_sigm))

#     # previous method by axel. not good either
#     hotpixmap = data*0 
#     data_med = ndimage.median_filter(data,neighborhood_size)
#     hotpixwh = np.where((np.abs(data_med - data) > (threshold*data_med)))
#     hotpixmap[hotpixwh] = 1
    
#     return hotpixmap

# def noise_filter(data, neighborhood_size, threshold):
#     """
#     Filter noise pixels in image

#     Parameters
#     ----------
#     data : image
#     neighborhood_size : typical distance between bad pixels
#     threshold : defined threshold

#     Commented by JM : You should not do that: bad pix is much more easy to find in dark
#     but you need to filter images

#     Returns
#     -------
#     image : processed image, where hot pixels have been removed

#     """
#     hotpixmap = FindNoisyPix(data,neighborhood_size,threshold)
#     image = mean_window_8pix(data,hotpixmap)
#     return image

def mean_window_8pix(array, hotpix):
    """ --------------------------------------------------
    the hot pixels are averaged to it's eight neighbor.
    I use a sinplified stuff because I don't care about the pixels on the edge.
    
    Parameters:
    ----------
    array: 2D array, raw image 
    hotpix: 2D array, hot pixels both array must have same size

    Return:
    ------
    image: processed coronagraphic image, where hot pixels have been removed
    -------------------------------------------------- """

    wh_dead = np.where(hotpix == 1)
    array[wh_dead] = np.nan
    array_copy = array.copy()

    #At hot pixel locations, the pixel value is equal to the mean of the eight pixels around
    for numdead in range(len(wh_dead[0])):
        i = wh_dead[0][numdead]
        j = wh_dead[1][numdead]

        # pixel on the edge, ignored
        if i == 0 or i == array_copy.shape[0]-1 or j == 0 or j == array_copy.shape[1] -1 :
            continue
        else:
            array[i, j] = np.nanmean([
                array_copy[i - 1, j], array_copy[i + 1, j], array_copy[i, j - 1],
                array_copy[i, j + 1], array_copy[i - 1, j - 1], array_copy[i + 1, j + 1],
                array_copy[i - 1, j + 1], array_copy[i + 1, j - 1]
            ])
    # finally we remove the nans that can happens if you have very large zones of hot pixs
    array[np.isnan(array)] = 0

    return array


def reduceimageSPHERE(param, file,  maxPSF, remove_bad_pix = True, high_pass_filter = False):
    """ --------------------------------------------------
    Processing of SPHERE images before being used and division by the maximum of the PSF
    
    Parameters:
    ----------
    param:
    file: str, path to the file to process
    maxPSF: int, maximum of the raw PSF

    Return:
    ------
    image: processed coronagraphic image, normalized by the max of the PSF
    -------------------------------------------------- """

    directory = param["ImageDirectory"]
    detector = param["detector"]

    # Get image exposure time
    expim = get_exptime(file)

    if detector == "IRDIS":
        # Load dark that correspond to image exposure time
        back = fits.getdata(last(directory+'SPHERE_BKGRD_EFC_'+str(int(expim))+'s_*.fits'))[0] 
        # Load image
        image = np.mean(fits.getdata(file),axis = 0) 

        # Subtract back from image
        image = image - back
        image = reduce_image_IRDIS(param, image, back, remove_bad_pix, high_pass_filter)
        
    elif detector == "IFS":
        image = reduce_image_IFS(param, fits.getdata(file))
    
    #We normalize the image with the max of the PSF
    if len(maxPSF)>1:
        image = (image/expim) / maxPSF[:, None, None]
    else:
        image = (image/expim) / maxPSF
    return image


def reduce_image_IRDIS(param, image, back, remove_bad_pix, high_pass_filter):

    centerx = param["centerx"]
    centery = param["centery"]
    dimimages = param["dimimages"]
    ctr_x = int(centerx)
    ctr_y = int(centery)

    # Crop to keep relevant part of image
    image = cropimage(image,ctr_x,ctr_y,dimimages)
    # Crop to keep relevant part of dark
    back_crop = cropimage(back,ctr_x,ctr_y,dimimages)    
    
    # We remove the hot pixels found in dark
    if remove_bad_pix == True:
        hotpixmap = find_hot_pix_in_dark(back_crop)
        image = mean_window_8pix(image,hotpixmap)

        # finally, we apply a sigma filter to filter out the ~1% remaining bad pix
        image = sigma_filter(image, box_width=3, n_sigma=7, monitor=True)

    # We process the image with a high pass filter    
    if high_pass_filter == True:
        image = high_pass_filter_gauss(image, 2)
    return image

def reduce_image_IFS(param, image):

    return image


def build_calibration_IFS(param):

    root_directory = param["RootDirectory"]
    obs_band = param['obs_band']
    cal_directory = root_directory + 'IFS_cal/'

    #Look for input (raw SPHERE data) and output (charis calibration files) directories
    wavecal_outputdir = cal_directory + 'wavecal_outputdir'
    wavecal_inputdir = cal_directory + 'wavecal_inputdir'

    if not os.path.exists(wavecal_outputdir):
        os.makedirs(wavecal_outputdir)

    config = defaultIFSReduction()

    # Apply resource configuration to all sub-configs
    config.apply_resources()

    # Extract step configuration for cleaner code
    # steps = config.steps

    # Get directory paths from config
    # raw_directory = config.directories.raw_directory
    # reduction_directory = config.directories.reduction_directory

    if obs_band == 'OBS_YJ':
        instrument = charis.instruments.SPHERE('YJ')
        config.extraction.R = 55
    elif obs_band == 'OBS_H':
        instrument = charis.instruments.SPHERE('YH')
        config.extraction.R = 35

    calibration_parameters, extraction_parameters, _ , _ = config.as_plain_dicts()
    extraction_parameters['bgsub'] = True
    extraction_parameters['fitbkgnd'] = False
    extraction_parameters['fitshift'] = True
    extraction_parameters['bg_scaling_without_mask'] = False
    extraction_parameters['maxcpus'] = 1
    extraction_parameters['refine'] = False
    extraction_parameters['flatfield'] = False
    extraction_parameters['resample'] = True
    extraction_parameters['linear_wavelength'] = True
    extraction_parameters['smoothandmask'] = False
    extraction_parameters['method'] = 'optext'

    files_in_calibration_folder = glob.glob(os.path.join(wavecal_outputdir, '*key*.fits'))

    if len(files_in_calibration_folder) == 0: # or overwrite_calibration:
        calibration_wavelength = instrument.calibration_wavelength
        wavecal_file = wavecal_inputdir + '/*.fits'#observation.frames['WAVECAL']['FILE'][0]
        inImage, hdr = charis.buildcalibrations.read_in_file(
            wavecal_file, instrument, calibration_wavelength,
            ncpus=calibration_parameters['ncpus'])
        charis.buildcalibrations.buildcalibrations(
            inImage=inImage, instrument=instrument,
            inLam=calibration_wavelength.value,
            outdir=wavecal_outputdir,
            header=hdr,
            **calibration_parameters)

    return extraction_parameters, wavecal_outputdir, instrument

def process_cube_IFS(filename, instrument, delta_wave = 5):
    
    #Stack, rotate and save results from charis

    #Pick resampled IFS cube file
    root, _ = os.path.splitext(filename)
    filename = root + '_cube_resampled_DIT_000.fits'

    cube, header = fits.getdata(filename, ext=1, header=True)
    raw_nb_wave = len(cube) #39

    min_wave = instrument.wavelength_range[0].value
    max_wave = instrument.wavelength_range[1].value
    wavelengths = np.linspace(min_wave, max_wave, raw_nb_wave)
    wavelength_0 = (min_wave + max_wave)/2
    d_wavelength = (max_wave - min_wave)

    #Stacking data to get ~5% images
    nb_images_per_stack = int(delta_wave / ( d_wavelength / wavelength_0 / raw_nb_wave * 100 ))
    remainder = raw_nb_wave%nb_images_per_stack
    if remainder<3: #To ensure it suppresses the lowest and highest wavelength
        remainder = remainder + nb_images_per_stack
    if remainder % 2 == 0:
        wavelength_cropped = wavelengths[int(remainder/2):-int(remainder/2)]
    else:
        wavelength_cropped = wavelengths[int(remainder/2)+1:-int(remainder/2)]

    #Centraled wavelength stacked
    wavelength_stacked = [np.mean(wavelength_cropped[a:a+nb_images_per_stack]) for a in nb_images_per_stack * np.arange(len(wavelength_cropped)//nb_images_per_stack)]
    wavelength_stacked = np.array(wavelength_stacked,dtype=int)

    #Rotate
    angle_in_degree = -102 + 1.75 #Maire SPIE 2016 Still a slight angle remains wrt IRDIS 1.75
    if remainder%2 == 0:
        cube_cropped = cube[int(remainder/2):-int(remainder/2)]
    else:
        cube_cropped = cube[int(remainder/2)+1:-int(remainder/2)]

    #Stack wavelength
    cube_stacked = [ndimage.rotate(np.mean(cube_cropped[a:a+nb_images_per_stack], axis=0), -angle_in_degree) for a in nb_images_per_stack*np.arange(len(cube_cropped)//nb_images_per_stack)]
    cube_stacked = np.array(cube_stacked)

    #Rename result to fit IRDIS-like code (remove IFS in filenames)
    before, _, after = root.partition("IFS_")
    new_namefile = before + after + '.fits'  # 'after' is everything after "IFS_"
    fits.writeto(new_namefile, cube_stacked, header, overwrite = True)

    os.remove(filename)
    os.remove(root + '_cube_DIT_000.fits')


def extract_cube_IFS(file, wavecal_outputdir, cube_outputdir, extraction_parameters):

    hdr = fits.getheader(file)
    ndit: int = int(hdr['HIERARCH ESO DET NDIT'])
    for dit_index in tqdm(
                    range(ndit),
                    desc=f"Extract datacube (1 CPU)",
                    unit="DIT",
                    leave=False,
                ):
        charis.extractcube.getcube(
            filename=file,
            dit=dit_index,
            bgpath=None,#wavecal_outputdir + '/SPHERE_IRDIFS_IFS_CAL_DARK259_0005.fits',
            calibdir=wavecal_outputdir + '/',
            outdir=cube_outputdir,
            **extraction_parameters,
        )

def find_hot_pix_in_dark(dark):
    """
    Find noisy pixels in image an dark. I checked with Philippe Delorme, this
    is what they are doing on sphere. This is super fast and not that bad
        JM
    Parameters
    ----------
    dark : dark image
    
    Returns
    -------
    hotpixmap : 2D map filled with 1 at hot pixel location

    """

    # We do a first pass just to remove very brigh pix or negative bright in darks 
    threshold_sup_bad_pix = 100
    threshold_inf_bad_pix = -100

    above_threshold_pix = np.zeros(dark.shape)
    above_threshold_pix[np.where(dark > threshold_sup_bad_pix)] = 1

    under_threshold_pix = np.zeros(dark.shape)
    under_threshold_pix[np.where(dark < threshold_inf_bad_pix)] = 1

    copy_dark_nan_pix = np.copy(dark)
    copy_dark_nan_pix[np.where(dark > threshold_sup_bad_pix)] = np.nan
    copy_dark_nan_pix[np.where(dark < threshold_inf_bad_pix)] = np.nan

    # We do a second pass based on a 3 sigma filter globally
    remaining_noisy_pix = np.zeros(dark.shape)
    remaining_noisy_pix[np.where(copy_dark_nan_pix - np.nanmean(copy_dark_nan_pix)> 30* np.nanstd(copy_dark_nan_pix))] = 1 # WAS 3!!

    hotpixmap = np.clip(above_threshold_pix + under_threshold_pix + remaining_noisy_pix, 0,1 )
    return hotpixmap


def high_pass_filter_gauss(image, sigma):
    """
    

    Parameters
    ----------
    image : Image to filter
    sigma : gaussian parameter in pixel

    Returns
    -------
    image : filtered image

    """
    lowpass = ndimage.gaussian_filter(image, sigma)
    image = image - lowpass
    return image

def rescale_coherent_component(signal_co, signal_tot, maskDH, nb_loop):
    """
    

    Parameters
    ----------
    signal_co : Image resulted from PW, removed from bad pixels, with halo
    signal_tot : Coro image, with halo
    maskDH : region where scaling is calculated
    nb_loop: number of iterations for calculation

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Copy coherent signal to preserve
    signal_co_copy = signal_co.copy()

    # High pass filter coherent and total signals
    filtered_co = high_pass_filter_gauss(signal_co_copy, 2)
    filtered_tot = high_pass_filter_gauss(signal_tot, 2)

    # Trying to match filtered coherent signal and filtered total intensity in DH
    best_params = []
    best_params2 = []
    best_params3 = []
    for i in np.arange(nb_loop):
        # Translation in x,y
        def cost_function(xy_trans):
            # Function can use image slices defined in the global scope
            # Calculate X_t - image translated by x_trans
            unshifted = fancy_xy_trans_slice(filtered_co, xy_trans)
            #Return mismatch measure for the translated image X_t
            return np.sum(np.abs(filtered_tot[np.where(maskDH)] - unshifted[np.where(maskDH)])**2*1e5)
            
        # Compute solution
        best_params.append( fmin_powell(cost_function, [0, 0], disp=0, callback=None) )
        # Apply solution
        filtered_co = fancy_xy_trans_slice(filtered_co, best_params[-1])
        
        # Translation in z
        def cost_function2(factor):
            a = filtered_co + factor
            return np.sum(np.abs(filtered_tot[np.where(maskDH)] - a[np.where(maskDH)])*1e5)
        
        # Compute solution
        best_params2.append( fmin_powell(cost_function2, 1e-5, disp=0, callback=None) )
        # Apply solution
        filtered_co = filtered_co + best_params2[-1]
        
        # Scaling in z
        def cost_function3(factor):
            a = filtered_co * factor
            return np.sum(np.abs(filtered_tot[np.where(maskDH)] - a[np.where(maskDH)])*1e5)
        
        # Compute solution
        best_params3.append( fmin_powell(cost_function3, 0.9, disp=0, callback=None) )
        # Apply solution
        filtered_co = filtered_co * best_params3[-1]
    
    # Global scaling
    scaling = np.prod( best_params3 )
    
    # Compute solution to unfiltered data
    for i in np.arange(len(best_params)):
        signal_co_copy = fancy_xy_trans_slice( signal_co_copy, best_params[i] )
        signal_co_copy = signal_co_copy + best_params2[i]
        signal_co_copy = signal_co_copy * best_params3[i]
    
    return signal_co_copy, scaling


def process_PSF(param):
    """
    Process non coronagraphic point spread function and return relevant data

    Parameters
    ----------
    directory : PSF path
    lightsource_estim : on-sky or internal source?
    centerx : unprecise position of the PSF in the image
    centery : unprecise position of the PSF in the image
    dimimages : Final PSF image dimension

    Returns
    -------
    PSF : Processed PSF image
    smoothPSF : Low-pass filtered PSF image
    maxPSF : maximum value of PSF
    exppsf : PSF exposure time

    """
    lightsource_estim = param['lightsource_estim']
    ImageDirectory = param["ImageDirectory"]

    #TODO: MIND ND FILTER PER WVLGTH!! 
    if param["which_nd"] == 'ND_3.5':
        ND = 1/0.00105
    elif param["which_nd"] == 'ND_2.0':
        ND = 1/0.0179
    else:
        ND = 1.


    file_PSF = last(ImageDirectory+lightsource_estim+'OffAxisPSF*.fits')
    exppsf = get_exptime(file_PSF)
    PSF = reduceimageSPHERE(param, file_PSF, np.array([1]), remove_bad_pix = False, high_pass_filter=False)
    PSF = PSF * ND
    smoothPSF = []
    maxPSF = []
    for wvl in np.arange(len(PSF)):
        smoothPSF.append(snd.median_filter(PSF[wvl], size=3))
        max_coord = np.unravel_index(np.argmax(smoothPSF[-1], axis=None), smoothPSF[-1].shape)
        maxPSF.append( PSF[ wvl , max_coord[0] , max_coord[1] ] ) 

    smoothPSF = np.array(smoothPSF)
    maxPSF = np.array(maxPSF)
    return PSF, smoothPSF, maxPSF, exppsf



def createdifference(param):
    """ --------------------------------------------------
    Create difference cube of PW images
    
    Parameters:
    ----------
    directory: str, directory of the images location
    filenameroot: str, name of the files
    posprobes: array, index of poked actuators used for PW
    nbiter: Iteration number to retrieve the good data
    ctr_x: int, center of processed image in the x direction
    ctr_y: int, center of processed image in the y direction

    Return:
    ------
    Difference: 3D array, cube of images
    -------------------------------------------------- """
    
    lightsource_estim = param['lightsource_estim']
    dimimages = param['dimimages']
    centeringateachiter = param['centeringateachiter']
    ImageDirectory = param["ImageDirectory"]
    directory = ImageDirectory + param["exp_name"]
    centerx = param['centerx']
    centery = param['centery']
    nbiter = param['nbiter']
    posprobes = param['posprobes']
    estim_algorithm = param['estim_algorithm']
    MatrixDirectory = param["MatrixDirectory"]
    size_probes = param["size_probes"]
    probe_type = param["probe_type"]
    zone_to_correct = param["zone_to_correct"]
    detector = param["detector"]

    #PSF
    PSF, _, maxPSF, _ = process_PSF(param)
    #print('!!!! ACTION: MAXIMUM PSF HAS TO BE VERIFIED ON IMAGE: ', maxPSF, flush=True)
    
    #Correction
    filecorrection = last(directory + 'iter' + str(nbiter-2) + '_coro_image*.fits')
    imagecorrection = reduceimageSPHERE(param, filecorrection, maxPSF)
        
    #Traitement de l'image de référence (première image corono et recentrage subpixelique)
    if centeringateachiter == 1 and detector == "IRDIS":
        fileref = last(directory + 'iter0_coro_image*.fits')
        imageref = reduceimageSPHERE(param, fileref, maxPSF)
        imageref = fancy_xy_trans_slice(imageref, [centerx-int(centerx), centery-int(centery)])
        
        def cost_function(xy_trans):
            # Function can use image slices defined in the global scope
            # Calculate X_t - image translated by x_trans
            unshifted = fancy_xy_trans_slice(imagecorrection, xy_trans)
            mask = roundpupil(dimimages, 67)
            #Return mismatch measure for the translated image X_t
            return correl_mismatch(imageref[np.where(mask == 0)], unshifted[np.where(mask == 0)])
        
        #Calcul de la translation du centre par rapport à la réference: best param.
        #Les images probes sont ensuite translatées de best param
        best_params = fmin_powell(cost_function, [0, 0], disp=0, callback=None) #callback=my_callback
        print('   Shifting recorded images by: ',best_params,' pixels' , flush=True)
    else:
        print('No recentering', flush=True)
        best_params = [centerx-int(centerx), centery-int(centery)]
    imagecorrection = fancy_xy_trans_slice(imagecorrection, best_params)
    
    
    #Probes
    numprobes = len(posprobes)
    Difference = np.zeros((numprobes,dimimages,dimimages))  
    Images_to_display=[]
    k = 0
    j = 1
    for i in posprobes:
        image_name = last(directory+'iter'+str(nbiter-1)+'_Probe_'+'%04d' % j+'*.fits')
        #print('Loading the probe image {0:s}'.format(image_name), flush=True)
        Ikplus = reduceimageSPHERE(param, image_name, maxPSF)
        Ikplus = fancy_xy_trans_slice(Ikplus, best_params)
        Images_to_display.append(extract_image(Ikplus-imagecorrection))
        j = j + 1
        
        if estim_algorithm == 'PWP':
            image_name = last(directory+'iter'+str(nbiter-1)+'_Probe_'+'%04d' % j+'*.fits')
            #print('Loading the probe image {0:s}'.format(image_name), flush=True)
            Ikmoins = reduceimageSPHERE(param, image_name, maxPSF)
            Ikmoins = fancy_xy_trans_slice(Ikmoins, best_params)
            Images_to_display.append(extract_image(Ikmoins-imagecorrection))
            j = j + 1
            
        elif estim_algorithm == 'BTW':
            filename = probe_type + '_' + zone_to_correct + '_' + str(size_probes) + 'nm' + '_'
            Probe_intens = fits.getdata(MatrixDirectory + lightsource_estim + filename + 'Intensity_probe.fits')[k]
            Ikplus = 2*Ikplus
            Ikmoins = 2*(imagecorrection + Probe_intens) #Missing model component
            Images_to_display.append(np.zeros((170-30,170-30)))
        
        else: 
            print('ERROR: Unvalid ESTIM_ALGORITHM value: should either be PWP or BTW', flush=True)
            break
            
        Difference[k] = (Ikplus-Ikmoins)
        k = k + 1

    Images_to_display.append(extract_image(PSF, final_size = len(PSF[0])))
    return Difference, imagecorrection, Images_to_display

def extract_image(image_to_extract, final_size = 140, index = 0):
    if image_to_extract.ndim == 3:
        image = image_to_extract[index]
    else:
        image = image_to_extract

    ctr = int(len(image) /2)
    return cropimage(image, ctr, ctr, final_size)



def display(image, axe, title, vmin, vmax , norm = None):
    """ --------------------------------------------------
    Create a binary mask.
    
    Parameters:
    ----------
    image: 2D array, image to display
    axe: name of the axe to display
    title: str, name of the image
    vmin: float, min scale of the display
    vmax: float, max scale of the display
    norm: Can be LogNorm

    -------------------------------------------------- """
    cmap = 'inferno'
    if norm == 'log':
        axe.imshow(image, norm=LogNorm(vmin, vmax), cmap = cmap)
    else:
        axe.imshow(image, vmin = vmin, vmax = vmax, cmap = cmap)
    axe.set_xticks([])
    axe.set_yticks([])
    axe.set_title(title,size=7)
    plt.draw()
    plt.pause(0.1)
    return 0


def contrast_global(image,scoring_reg):
    """
    Calculate contrast in the scoring region

    Parameters
    ----------
    image : Image
    scoring_reg : Binary mask to define scoring region

    Returns
    -------
    contrast_mean : mean contrast
    contrast_std : contrast rms

    """
    contrast_mean = np.nanmean(image[np.where(scoring_reg)])
    contrast_std = np.nanstd(image[np.where(scoring_reg)])
    return contrast_mean, contrast_std
    
def extract_contrast_global(cubeimage, scoring_region, index = 0):
    """
    Calculate contrast in image cube

    Parameters
    ----------
    cubeimage : image cube
    scoring_region : Binary mask to define scoring region

    Returns
    -------
    contrast :array of mean contrast and contrast rms

    """
    if cubeimage.ndim == 3:
        image = cubeimage[index]
    else:
        image = cubeimage
    
    return contrast_global(image, scoring_region)


def resultEFC(param):
    """ --------------------------------------------------
    Give EFC solution in WFS slope
    
    Parameters:
    ----------
    directory:
    filenameroot:
    posprobes:
    nbiter:
    centerx:
    centery:

    Return:
    ------
    resultatestimation:
    slopes: 
    -------------------------------------------------- """
    amplitudeEFCMatrix = param['amplitudeEFCMatrix']
    MatrixDirectory = param["MatrixDirectory"]
    lightsource_corr = param["lightsource_corr"]
    lightsource_estim = param["lightsource_estim"]
    zone_to_correct = param["zone_to_correct"]
    size_probes = param["size_probes"]
    dhsize = param["dhsize"]
    corr_mode = param['corr_mode']
    gain = param['gain']
    rescaling = param['rescaling']
    probe_type = param['probe_type']
    
    filename = probe_type + '_' + zone_to_correct + '_' + str(size_probes) + 'nm' + '_'
    PWP_matrix = fits.getdata(MatrixDirectory + lightsource_estim + filename + 'PWP_matrix.fits')
    maskDH = fits.getdata(MatrixDirectory+'mask_DH'+str(dhsize)+'.fits')
    
    intensity_co = []
    intensity_inco =[]
    #imagecorrection = []
    resultatestimation = []

    print('- Creating difference of images...', flush=True)
    Difference, imagecorrection, Images_to_display = createdifference(param)

    for wvl in np.arange(len(PWP_matrix)):
        print('- Estimating the focal plane electric field...', flush=True)
        Difference_per_wvl = Difference[:, wvl]
        imagecorrection_per_wvl = imagecorrection[wvl]
        PWP_matrix_per_wvl = PWP_matrix[wvl]
        resultatestimation_per_wvl = estimate_efield(Difference_per_wvl, PWP_matrix_per_wvl)
        intensity_co_per_wvl = np.abs(resultatestimation)**2

        if rescaling == 1:
            print('- Rescaling solution and computing incoherent component...', flush=True)
            intensity_co_per_wvl, intensity_inco_per_wvl, scaling = rescale_coherent_component(intensity_co_per_wvl, imagecorrection_per_wvl, maskDH, 5)
            print('- Applied factor = ' + str(scaling), flush=True)
            resultatestimation_per_wvl = resultatestimation_per_wvl * scaling
        
        intensity_inco_per_wvl = imagecorrection_per_wvl - intensity_co_per_wvl
        
        intensity_co.append(intensity_co_per_wvl)
        intensity_inco.append(intensity_inco_per_wvl)
        #imagecorrection.append(imagecorrection_per_wvl)
        resultatestimation.append(resultatestimation_per_wvl)
    
    intensity_co = np.array(intensity_co)
    intensity_inco = np.array(intensity_inco)
    #imagecorrection = np.array(imagecorrection)
    resultatestimation = np.array(resultatestimation)
    
    if gain!=0:
        print('- Calculating slopes to generate the Dark Hole with EFC...', flush=True)
        WhichInPupil = fits.getdata(MatrixDirectory+lightsource_estim+'WhichInPupil0_5.fits')
        invertGDH = fits.getdata(MatrixDirectory+lightsource_corr+'Interactionmatrix_DH'+str(dhsize)+'_SVD'+str(corr_mode)+'.fits')

        
        solution1 = solutiontocorrect(maskDH, resultatestimation, invertGDH, WhichInPupil)
        solution1 = solution1*amplitudeEFCMatrix/rad_632_to_nm_opt
        solution1 = -gain*solution1
        slopes = VoltToSlope(MatrixDirectory, solution1)
    
    else:
        slopes = 0
        
    return intensity_co, intensity_inco, imagecorrection, Images_to_display, slopes
        



def recordslopes(slopes, dir, refslope, namerecord):
    """ --------------------------------------------------
    Save a slope.
    
    Parameters:
    ----------
    slopes:
    dir:
    refslope:
    namerecord:
    
    -------------------------------------------------- """
    ref = dir + refslope + '.fits'
    #hdul = fits.open(ref) #Charge la forme des pentes actuelles
    data, header1 = fits.getdata(ref, header=True)
    fits.writeto(dir+namerecord+'.fits', np.asarray(data+slopes,dtype=np.float32), header1, overwrite=True)
    return 0
    
   
   
    
def recordnewprobes(param, dir, refslope, nbiter):
    """ --------------------------------------------------
    Save new slopes to create PW probes on the DM
    
    Parameters:
    ----------
    param:
    dir:
    refslope:
    nbiter:

    -------------------------------------------------- """
    
    MatrixDirectory = param["MatrixDirectory"]
    nbiter = param['nbiter']
    lightsource_estim = param["lightsource_estim"]
    size_probes = param["size_probes"]
    estim_algorithm = param['estim_algorithm']
    probe_type = param['probe_type']
    zone_to_correct = param["zone_to_correct"]
    
    filename = probe_type + '_' + zone_to_correct + '_' + str(size_probes) + 'nm' + '_'
    tensionDM = fits.getdata(MatrixDirectory + lightsource_estim + filename + 'Voltage_probe.fits')
    tensionDM = tensionDM/rad_632_to_nm_opt
    k = 1
    for j in np.arange(len(tensionDM)):
        slopetopush = VoltToSlope(MatrixDirectory, tensionDM[j])
        recordslopes(slopetopush, dir, refslope, 'iter'+str(nbiter)+'probe'+str(k))
        k = k + 1
        
        if estim_algorithm == 'PWP':
            slopetopush = VoltToSlope(MatrixDirectory, -tensionDM[j])
            recordslopes(slopetopush,dir,refslope,'iter'+str(nbiter)+'probe'+str(k))
            k = k + 1
    return 0
        
def FullIterEFC(param):
    """ --------------------------------------------------
    PW + EFC Iteration wrapper
    
    Parameters:
    ----------
    dir:
    posprobes:
    nbiter:
    filenameroot:
    record:

    -------------------------------------------------- """
    
    dir = param["ImageDirectory"]
    MatrixDirectory = param["MatrixDirectory"]
    posprobes = param["posprobes"]
    nbiter = param['nbiter']
    filenameroot = param["exp_name"]
    size_probes = param["size_probes"]
    dimimages = param["dimimages"]
    onsky = param["onsky"]
    slope_ini = param["slope_ini"]
    detector = param['detector']
    gain = param['gain']
    #Check if the directory dir exists
    if os.path.isdir(dir) is False:
        #Create the directory
        os.mkdir(dir)
    dir2 = dir + filenameroot
        
    dhsize = param["dhsize"]
    maskDH = fits.getdata(MatrixDirectory+'mask_DH'+str(dhsize)+'.fits')

    if nbiter == 1:
        print('Creating slopes for Cosine, PSFOffAxis and new probes...', flush=True)
        #Copy the reference slope with the right name for iteration 0 of ExperimentXXXX
        recordslopes(np.zeros(2480), dir, slope_ini, filenameroot+'iter0correction')
        #Create the cosine of 10nm peak-to-valley amplitude for centering
        recordCoswithvolt(param, 10, slope_ini)
    else:

        if detector == "IFS":
            # Calibrate the IFS if undone before
            extraction_parameters, wavecal_outputdir, instrument = build_calibration_IFS(param)

            # Extract all the IFS raw images in the experiment/iter and create cubes
            filenames = []
            for name in os.listdir(dir):
                if "IFS_" in name:
                    # Split at the first occurrence of "IFS_"
                    before, _, after = name.partition("IFS_")
                    x = before + after  # 'after' is everything after "IFS_"

                    if x and not os.path.exists(os.path.join(dir, x)):
                        filenames.append(dir + name)
            #print(filenames)
            do_extract_cube_IFS = partial(extract_cube_IFS, wavecal_outputdir=wavecal_outputdir, cube_outputdir=dir, extraction_parameters=extraction_parameters)  # freeze b and c
            Parallel(n_jobs=-1)(delayed(do_extract_cube_IFS)(filename) for filename in filenames)

            do_process_cube_IFS = partial(process_cube_IFS, instrument=instrument, delta_wave=5)  # freeze b and c
            Parallel(n_jobs=-1)(delayed(do_process_cube_IFS)(filename) for filename in filenames)


        if nbiter == 2:
            #Calculate the center of the first coronagraphic image using the waffle
            print('Calculating center of the first coronagraphic image:', flush=True)
            data,centerx,centery = find_center_with_cosine(param)
            SaveFits([centerx,centery], ['',0], dir2, 'centerxy')
        
        centerx, centery = fits.getdata(dir2 + 'centerxy.fits')
        param['centerx'] = centerx
        param['centery'] = centery
        #Estimation of the electric field using the pair-wise probing (return the electric field and the slopes)
        print('Estimating the electric field using the pair-wise probing:', flush=True)
        coherent_signal, incoherent_signal, imagecorrection, Images_to_display, pentespourcorrection  = resultEFC(param)
        refslope = 'iter' + str(nbiter-2) + 'correction'
        #Record the slopes to apply for correction at the next iteration (only if not CDI i.e. gain!=0)
        if gain!=0:
            recordslopes(pentespourcorrection, dir2, refslope, 'iter'+str(nbiter-1)+'correction')
            
        
        #Save data 
        fits.writeto(dir2+'iter'+str(nbiter-2)+'CoherentSignal.fits', coherent_signal, overwrite = True)
        fits.writeto(dir2+'iter'+str(nbiter-2)+'IncoherentSignal.fits', incoherent_signal, overwrite = True)
        fits.writeto(dir2+'iter'+str(nbiter-2)+'TotalIntensity.fits', imagecorrection, overwrite = True)
        
        Contrast_tot = str(format(extract_contrast_global(imagecorrection,maskDH)[0],'.2e'))
        Contrast_cor = str(format(extract_contrast_global(coherent_signal,maskDH)[0],'.2e'))
        Contrast_inc = str(format(extract_contrast_global(incoherent_signal,maskDH)[0],'.2e'))
        
        print('Contrast in DH region at iter '+str(nbiter-2)+ ' = ' , Contrast_tot, flush=True)
        
        to_write = [Contrast_tot, Contrast_cor, Contrast_inc]
        if os.path.exists(dir2 + 'Contrast_vs_iter.txt'):
            with open(dir2 + 'Contrast_vs_iter.txt', 'r') as f:
                text_in_file = f.readlines()
        else:
            with open(dir2 + 'Contrast_vs_iter.txt', 'a+') as f:
                text_in_file = f.readlines()
        
        if len(text_in_file)==nbiter-2:
            with open(dir2 + 'Contrast_vs_iter.txt', 'a+') as f:
                f.writelines( " ".join(to_write) + "\n" )
        else:
            text_in_file[nbiter-2] = " ".join(to_write) + "\n"
            with open(dir2 + 'Contrast_vs_iter.txt', 'w') as f:
                f.writelines( text_in_file )
                
        # Display data
        imagecorrection_to_display = high_pass_filter_gauss(extract_image(imagecorrection), 2)
        coherent_signal_to_display = high_pass_filter_gauss(extract_image(coherent_signal), 2)
        incoherent_signal_to_display = high_pass_filter_gauss(extract_image(incoherent_signal), 2)
        
        
        if onsky == 1:
            vmin = -6e-6 #1e-7
            vmax = 6e-6 #1e-3
            norm = None #'log'
        
        else:
            vmin = -6e-6 #1e-7
            vmax = 6e-6 #1e-3
            norm = None #'log'
        
        plt.close()
        fig = plt.figure(constrained_layout=True,figsize=(12,7.5))
        ( fig1 , fig2 ) , ( fig3 , fig4 ) = fig.subfigures(2, 2)
        fig1.suptitle('Coro image iter'+str(nbiter-2), size=10)
        fig2.suptitle('Probe images iter'+str(nbiter-2), size=10)
        
        ax1 = fig1.subplots(1, 1, sharex=True, sharey=True)      
        display(imagecorrection_to_display, ax1, '', vmin = vmin, vmax = vmax, norm = norm)
        ax1.text(1, 12, 'Contrast = ' + Contrast_tot, size=15, color ='red', weight='bold')
        PSF_to_display = Images_to_display[-1]
        ax1bis = fig1.add_axes([0.65, 0.70, 0.25, 0.25])
        display(PSF_to_display, ax1bis, 'PSF' , vmin = 1, vmax = np.amax(PSF_to_display), norm='log')

        ax2 = fig2.subplots(len(posprobes), 2, sharex=True, sharey=True)
        k=0
        for j in np.arange(len(posprobes)):
            display(Images_to_display[k], ax2.flat[k] , title='+ '+str(posprobes[j]), vmin=-1e-4, vmax=1e-4)
            k=k+1
            display(Images_to_display[k], ax2.flat[k] , title='- '+str(posprobes[j]), vmin=-1e-4, vmax=1e-4)
            k=k+1
        
        ax3 = fig3.subplots(2,2)
        display(coherent_signal_to_display, ax3.flat[0] , title='Coherent iter' + str(nbiter-2), vmin = vmin, vmax = vmax, norm = norm )
        display(incoherent_signal_to_display, ax3.flat[2] , title='Incoherent iter' + str(nbiter-2), vmin = vmin, vmax = vmax, norm = norm)
        
        if os.path.isfile(dir2 + refslope + '.fits'):
            #If EFC
            slopes_to_display = pentespourcorrection + fits.getdata(dir2 + refslope + '.fits')[0]
        else:
            #If CDI only
            slopes_to_display = pentespourcorrection + fits.getdata(dir2 + 'iter0correction' + '.fits')[0]
        display(SHslopes2map(param['MatrixDirectory'], slopes_to_display, visu=False)[0], ax3.flat[1], title = 'Slopes SH in X to apply for iter'+str(nbiter-1), vmin =np.amin(slopes_to_display), vmax = np.amax(slopes_to_display) )
        display(SHslopes2map(param['MatrixDirectory'], slopes_to_display, visu=False)[1], ax3.flat[3], title = 'Slopes SH in Y to apply for iter'+str(nbiter-1), vmin =np.amin(slopes_to_display), vmax = np.amax(slopes_to_display) )
        
        ax4 = fig4.subplots(1,1)
                
            

        
        if nbiter == 2:
            print('!!!! ACTION: CHECK CENTERX AND CENTERY CORRESPOND TO THE CENTER OF THE CORO IMAGE AND CLOSE THE WINDOW', flush=True)
            print('If false, change the guess in the MainEFCBash.sh file and run the iter again', flush=True)
            
            #I do not use crop of data here so the center index can be checked on the full image.
            display(data, ax4, 'Cosine for centering', vmin=-5e2, vmax=5e2)
        else:
            pastContrast_tot = []
            pastContrast_co = []
            pastContrast_inco = []
            with open(dir2 + 'Contrast_vs_iter.txt') as f:
                for line in f:
                    pastContrast = np.float64(line.split())
                    pastContrast_tot.append(pastContrast[0])
                    pastContrast_co.append(pastContrast[1])
                    pastContrast_inco.append(pastContrast[2]) 
                
            ax4.plot(pastContrast_tot, marker = 'P', ms=10,  linewidth = 2, label = 'tot')
            ax4.plot(pastContrast_co ,marker = 'P', ms=10,  linewidth = 2, label = 'coherent')
            ax4.plot(pastContrast_inco, marker = 'P', ms=10,  linewidth = 2, label = 'incoherent')
            
            ax4.set_yscale('log')
            ax4.set_ylim(1e-7,1e-4)
            ax4.tick_params(axis='both', which='both', labelsize=8)
            ax4.set_title('Mean contrast in DH vs iteration', size=10)
            ax4.legend()
        print('Close each image to proceed', flush=True)
        plt.draw()
        if nbiter == 2:
            plt.show()
        else:
            plt.pause(1)
            plt.close()
    
    #Record new slope at first CDI iteration only (nbiter == 1) or at all EFC iteration (gain!=0)
    if gain!=0 or nbiter == 1:    
        #Record the slopes to apply for probing at the next iteration
        refslope = 'iter' + str(nbiter-1) + 'correction'
        recordnewprobes(param, dir2, refslope, nbiter)
        print('Done with recording new slopes!', flush=True)
        
    return 0


    
def VoltToSlope(MatrixDirectory,Volt):
    """ --------------------------------------------------
    Conversion volt on the DM to slope on the WFS
    
    Parameters:
    ----------
    Volt:

    Return:
    ------
    Slopetopush: 
    -------------------------------------------------- """
    # daily calibrations !! Ask for HO_IM matrix before starting and save in directory!! ####################
    V2S = fits.getdata(MatrixDirectory+ 'CLMatrixOptimiser.HO_IM.fits')
    # renormalization for weighted center of gravity: 0.4 sensitivity
    V2S = V2S / 0.4
    Slopetopush = V2S @ Volt
    return Slopetopush
    

    
def recordCoswithvolt(param, amptopushinnm, refslope):
    """ --------------------------------------------------
    Creation of the cosine (in slope) to apply on the DM    
    
    Parameters:
    ----------
    amptopushinnm:
    dir:
    refslope:

    -------------------------------------------------- """
    MatrixDirectory = param["MatrixDirectory"]
    dir = param["ImageDirectory"]
    # Read SAXO calibrations
    # static calibrations
    IMF_inv = fits.getdata(MatrixDirectory + 'SAXO_DM_IFM_INV.fits', ignore_missing_end=True)
    
    nam = ['cos_00deg','cos_30deg','cos_90deg']
    nbper = 10.
    dim = 240
    angle = 30
    xx, yy = np.meshgrid(np.arange(240)-(240)/2, np.arange(240)-(240)/2)
    cc = np.cos(2*np.pi*(xx*np.cos(0*np.pi/180.))*nbper/dim) #45 instead of 0?
    cc = ndimage.rotate(cc, angle, reshape = False, mode = 'grid-mirror')
    coe = (cc.flatten())@IMF_inv
    coe = coe*amptopushinnm/rad_632_to_nm_opt
    slopetopush = VoltToSlope(MatrixDirectory, coe)
    recordslopes(slopetopush, dir, refslope, nam[0] + '_' + str(amptopushinnm) + 'nm')
    return 0


def record_slope_from_file(param, file_path, amptopushinnm, refslope, name):
    """ --------------------------------------------------
    Creation of the phase shape (in slope) to apply on the DM    
    
    Parameters:
    ----------
    amptopushinnm:
    dir:
    refslope:

    -------------------------------------------------- """
    MatrixDirectory = param["MatrixDirectory"]
    dir = param["ImageDirectory"]
    # Read SAXO calibrations
    # static calibrations
    IMF_inv = fits.getdata(MatrixDirectory + 'SAXO_DM_IFM_INV.fits', ignore_missing_end = True)
    
    cc = fits.getdata(file_path)
    coe = (cc.flatten())@IMF_inv
    coe = coe * amptopushinnm / rad_632_to_nm_opt
    slopetopush = VoltToSlope(MatrixDirectory, coe)
    recordslopes(slopetopush, dir, refslope, name)
    return 0    
    
    
    
    
def find_center_with_cosine(param):    
    """ --------------------------------------------------
    Find center of the coronagraphic image using previously applied cosined.
    
    Parameters:
    ----------
    dir: location of the cosine

    Return:
    ------
    centerx, centery: floats, center of the coronagraphic image
    -------------------------------------------------- """
    x0_up = param["x0_up"]
    y0_up = param["y0_up"]
    x1_up = param["x1_up"]
    y1_up = param["y1_up"]
    dir = param["ImageDirectory"] + param["exp_name"]
    
    def twoD_Gaussian(xy, amplitude, sigma_x, sigma_y, xo, yo,h):
        xo = float(xo)
        yo = float(yo) 
        theta=0   
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
        g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                + c*((y-yo)**2)))+h
        
        return (g).flatten()
    
    #LOOK THE FITS FILE AND CHANGE QUIKLY X0,Y0,X1,Y1
    cosinepluscoro = last(dir+'CosineForCentering*.fits')
    coro = last(dir+'iter0_coro_image*.fits')
    
    #Fit gaussian functions
    #fits.getdata(cosinepluscoro)
    data = fits.getdata(cosinepluscoro)[-1]-fits.getdata(coro)[-1]
    data1 = cropimage(data,x0_up,y0_up,30)
    data2 = cropimage(data,x1_up,y1_up,30)
    data1[np.where(data1<0)] = 0
    data2[np.where(data2<0)] = 0
   
    w,h = data1.shape
    x, y = np.mgrid[0:w, 0:h]
    xy = (x,y)
                
    #Fit 2D Gaussian with fixed parameters for the top PSF
    initial_guess = (np.amax(data1), 1 , 1 , np.unravel_index(np.argmax(data1, axis=None), data1.shape)[0] , np.unravel_index(np.argmax(data1, axis=None), data1.shape)[1] ,np.mean(data1))
    
    try:
        popt1, _ = opt.curve_fit(twoD_Gaussian, xy, (data1).flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed top PSF", flush=True)

    #Fit 2D Gaussian with fixed parameters for the bottom PSF
    initial_guess = (np.amax(data2), 1 , 1 , np.unravel_index(np.argmax(data2, axis=None), data2.shape)[0] , np.unravel_index(np.argmax(data2, axis=None), data2.shape)[1] ,np.mean(data2))
    
    try:
        popt2, _ = opt.curve_fit(twoD_Gaussian, xy, (data2).flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed bottom PSF", flush=True)
        
    centerx = ((popt1[3]+x0_up-15)+(popt2[3]+x1_up-15))/2
    centery = ((popt1[4]+y0_up-15)+(popt2[4]+y1_up-15))/2
    
    print('- centerx = ',centery, flush=True)
    print('- centery = ',centerx, flush=True)
    return data, centerx, centery


def sigma_filter(image, box_width, n_sigma=3, ignore_edges=False, monitor=False):

    # NAME:
    #	SIGMA_FILTER
    # PURPOSE:
    #	Replace pixels more than a specified pixels deviant from its neighbors
    # EXPLANATION:
    #	Computes the mean and standard deviation of pixels in a box centered at
    #	each pixel of the image, but excluding the center pixel. If the center
    #	pixel value exceeds some # of standard deviations from the mean, it is
    #	replaced by the mean in box. Note option to process pixels on the edges.
    # CALLING SEQUENCE:
    #	Result = sigma_filter( image, box_width, n_sigma=(#), /ALL,/MON )
    # INPUTS:
    #	image = 2-D image (matrix)
    #	box_width = width of square filter box, in # pixels (default = 3)
    #	n_sigma = # standard deviations to define outliers, floating point,
    #			recommend > 2, default = 3. For gaussian statistics:
    #			n_sigma = 1 smooths 35% of pixels, 2 = 5%, 3 = 1%.
    #   ignore_edges: if False, we also apply the sigma filter to the edges. 
    #               If true, they're left untouched.
    #   monitor: prints information about % pixels replaced.
    #   
    # CALLS:
    #	function filter_image( )
    # PROCEDURE:
    #	Compute mean over moving box-cars using smooth, subtract center values,
    #	compute variance using smooth on deviations from mean,
    #	check where pixel deviation from mean is within variance of box,
    #	replace those pixels in smoothed image (mean) with orignal values,
    #	return the resulting partial mean image.
    # MODIFICATION HISTORY:
    #	Written, 1991, Frank Varosi and Dan Gezari NASA/GSFC
    #	F.V.1992, added optional keywords /ITER,/MON,VAR=,DEV=,N_CHANGE=.
    #	Converted to IDL V5.0   W. Landsman   September 1997
    #   Translated to python with chat GPT by Johan
    #-

    if box_width <3:
        raise ValueError("box_width must be an odd integer > 2")
    
    if box_width % 2 == 0:
         raise ValueError("box_width must be an odd integer > 2")
    
    bw2 = box_width**2

    smooth = np.ones((box_width, box_width))
    smooth[1:-1, 1:-1] = 0

    if ignore_edges:
        mean = (snd.generic_filter(image, np.mean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - image) / (bw2 - 1)
        wh_nan = np.isnan(mean)
        mean[wh_nan] = 0
    else:
        mean = (snd.generic_filter(image, np.mean, footprint=smooth, mode='mirror') * bw2 - image) / (bw2 -1)
        # mean = (generic_filter(image, np.nanmean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - image) / (bw2 -1)

    imdev = (image - mean)**2
    fact = float(n_sigma**2) / (bw2 - 2)

    if ignore_edges:
        imvar = fact * (snd.generic_filter(imdev, np.mean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - imdev)
        imdev[np.isnan(imvar)] = 0
        imvar[np.isnan(imvar)] = 0
    else:
        imvar = fact * (snd.generic_filter(imdev, np.nanmean, footprint=smooth, mode='mirror') * bw2 - imdev)
        # imvar = fact * (generic_filter(imdev, np.nanmean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - imdev)

    # chek which pixels are ok
    wok = np.where(imdev <= imvar)
    nok = wok[0].size

    npix = image.size
    nchange = npix - nok

    if monitor:
        if ignore_edges:
            print(f"{(nchange)*100./npix:.2f}% of pixels replaced (edges ignored), n_sigma={n_sigma:.1f}")
        else:
            print(f"{nchange*100./npix:.2f}% of pixels replaced, n_sigma={n_sigma:.1f}")

    if nok == npix:
        return image
    if nok > 0:
        mean[wok] = image[wok]

    if ignore_edges:
        mean[wh_nan] = image[wh_nan]

    return mean