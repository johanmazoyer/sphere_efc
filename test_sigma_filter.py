import numpy as np
import time
from astropy.io import fits
from scipy import ndimage

import numpy as np
from scipy.ndimage.filters import generic_filter


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

    if box_width < 3:
        raise ValueError("box_width must be an odd integer > 2")

    if box_width % 2 == 0:
        raise ValueError("box_width must be an odd integer > 2")

    bw2 = box_width**2

    smooth = np.ones((box_width, box_width))
    smooth[1:-1, 1:-1] = 0

    if ignore_edges:
        mean = (generic_filter(image, np.mean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - image) / (bw2 - 1)
        wh_nan = np.isnan(mean)
        mean[wh_nan] = 0
    else:
        mean = (generic_filter(image, np.mean, footprint=smooth, mode='mirror') * bw2 - image) / (bw2 - 1)
        # mean = (generic_filter(image, np.nanmean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - image) / (bw2 -1)

    imdev = (image - mean)**2
    fact = float(n_sigma**2) / (bw2 - 2)

    if ignore_edges:
        imvar = fact * (generic_filter(imdev, np.mean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - imdev)
        imdev[np.isnan(imvar)] = 0
        imvar[np.isnan(imvar)] = 0
    else:
        imvar = fact * (generic_filter(imdev, np.nanmean, footprint=smooth, mode='mirror') * bw2 - imdev)
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


def std_dev_data(x):
    return np.nanstd(x)


def mean_data(x):
    return np.nanmean(x)


def FindNoisyPix(data, neighborhood_size, threshold):
    """
    Find noisy pixels in image an image

    Parameters
    ----------
    data : dark image
    neighborhood_size : typical distance between bad pixels
    threshold : defined threshold

    Returns
    -------
    hotpixmap : 2D map filled with 1 at hot pixel location

    """
    hotpixmap = data * 0

    if neighborhood_size % 2 == 0:
        raise Exception("please use an odd number of neighborhood_size")

    footprint = np.ones((neighborhood_size, neighborhood_size))
    footprint[neighborhood_size // 2, neighborhood_size // 2] = 0

    start_time = time.time()
    data_sigm = ndimage.generic_filter(data, std_dev_data, footprint=footprint)
    print("measure sigmas:", time.time() - start_time)

    start_time = time.time()
    data_mean = ndimage.generic_filter(data, mean_data, footprint=footprint)
    print("measure means:", time.time() - start_time)

    hotpixwh = np.where(np.abs(data - data_mean) > (threshold * data_sigm))

    # data_med = ndimage.median_filter(data,neighborhood_size)
    # hotpixwh = np.where((np.abs(data_med - data) > (threshold*data_med)))

    hotpixmap[hotpixwh] = 1

    return hotpixmap


def noise_filter(data, neighborhood_size, threshold):
    """
    Filter noise pixels in image

    Parameters
    ----------
    data : image
    neighborhood_size : typical distance between bad pixels
    threshold : defined threshold

    Returns
    -------
    image : processed image, where hot pixels have been removed

    """
    hotpixmap = FindNoisyPix(data, neighborhood_size, threshold)
    image = mean_window_8pix(data, hotpixmap)
    return image


def mean_window_8pix(array, hotpix):
    """ --------------------------------------------------
    the hot pixels are averaged to it's eigh neighbor.
    I use a sinplified stuff  because I don't care about the pixels on the edge
    
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
        if i == 0 or i == array_copy.shape[0] - 1 or j == 0 or j == array_copy.shape[1] - 1:
            continue
        else:
            array[i, j] = np.nanmean([
                array_copy[i - 1, j], array_copy[i + 1, j], array_copy[i, j - 1], array_copy[i, j + 1],
                array_copy[i - 1, j - 1], array_copy[i + 1, j + 1], array_copy[i - 1, j + 1], array_copy[i + 1, j - 1]
            ])
    # finally we remove the nans that can happens if you have very large zones of hot pixs
    array[np.isnan(array)] = 0

    return array


orig_data = fits.getdata(
    "/Users/jmazoyer/Dropbox/ExchangeFolder/efc_sphere_may23/20230328/Experiment0002_iter0_coro_image_087_0001.fits")[
        0, 74:250, 1420:1600]

dark1 = fits.getdata(
    "/Users/jmazoyer/Dropbox/ExchangeFolder/efc_sphere_may23/20230328/SPHERE_BKGRD_EFC_1s_087_0001.fits")[0, 74:250,
                                                                                                          1420:1600]

dark2 = fits.getdata(
    "/Users/jmazoyer/Dropbox/ExchangeFolder/efc_sphere_may23/20230328/SPHERE_BKGRD_EFC_1s_087_0002.fits")[0, 74:250,
                                                                                                          1420:1600]

dark = (dark1 + dark2) / 2

init_dark = np.copy(dark)

outputdir = "/Users/jmazoyer/Desktop/test_badpix_new/"

# start_time = time.time()
# Noisy_pix = FindNoisyPix(dark, 3,2)
# fits.writeto(outputdir + "Noisy_pix.fits", Noisy_pix, overwrite=True)

threshold_sup_bad_pix = 200
threshold_inf_bad_pix = -50

above_threshold_pix = orig_data * 0
above_threshold_pix[np.where(dark > threshold_sup_bad_pix)] = 1
fits.writeto(outputdir + "above_threshold_pix.fits", above_threshold_pix, overwrite=True)

under_threshold_pix = orig_data * 0
under_threshold_pix[np.where(dark < threshold_inf_bad_pix)] = 1
fits.writeto(outputdir + "under_threshold_pix.fits", under_threshold_pix, overwrite=True)

dark[np.where(dark > threshold_sup_bad_pix)] = np.nan
dark[np.where(dark < threshold_inf_bad_pix)] = np.nan

remaining_noisy_pix_dark = orig_data * 0
remaining_noisy_pix_dark[np.where(dark - np.nanmean(dark) > 3 * np.nanstd(dark))] = 1
fits.writeto(outputdir + "remaining_noisy_pix.fits", remaining_noisy_pix_dark, overwrite=True)

Noisy_pix = np.clip(above_threshold_pix + under_threshold_pix + remaining_noisy_pix_dark, 0, 1)

fits.writeto(outputdir + "all_noisy_pix.fits", Noisy_pix, overwrite=True)

# Noisy_pix_bis = orig_data*0
# Noisy_pix_bis[np.where(dark - np.nanmean(dark)> 3* np.nanstd(dark))] = 1

# fits.writeto(outputdir + "Noisy_pix_bis.fits", Noisy_pix_bis, overwrite=True)

# asd

filter_data = mean_window_8pix(orig_data - dark, Noisy_pix)
# print(time.time() - start_time)

fits.writeto(outputdir + "orig.fits", orig_data, overwrite=True)
fits.writeto(outputdir + "orig_minus_dark.fits", orig_data - init_dark, overwrite=True)

# fits.writeto(outputdir + "dark1.fits", dark1, overwrite=True)
# fits.writeto(outputdir + "dark2.fits", dark2, overwrite=True)
fits.writeto(outputdir + "dark.fits", dark, overwrite=True)

fits.writeto(outputdir + "filter.fits", filter_data, overwrite=True)

# sigma_filt_ignore_edges = sigma_filter(filter_data, box_width=3, n_sigma=7, monitor=True, ignore_edges=True)
sigma_filt_with_edges = sigma_filter(filter_data, box_width=3, n_sigma=7, monitor=True, ignore_edges=False)

# fits.writeto(outputdir + "sigma_filt_ignore_edges.fits", sigma_filt_ignore_edges, overwrite=True)
fits.writeto(outputdir + "sigma_filt_with_edges.fits", sigma_filt_with_edges, overwrite=True)

just_sigma_filt = sigma_filter(orig_data - init_dark, box_width=3, n_sigma=7, monitor=True, ignore_edges=False)
fits.writeto(outputdir + "just_sigma_filt.fits", just_sigma_filt, overwrite=True)
