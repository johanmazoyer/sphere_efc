import numpy as np
import time
from astropy.io import fits
from scipy import ndimage


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
    the hot pixels are averaged to it's eigh neighbor. I do a slightly complicated stuff
    when I nan pad the array in case the hot pix is on the edge of the array. 
    That can probably be simplified if you don't care about the pixels on the edge
    
    Parameters:
    ----------
    array: 2D array, raw image 
    hotpix: 2D array, hot pixels both array must have same size

    Return:
    ------
    image: processed coronagraphic image, where hot pixels have been removed
    -------------------------------------------------- """
    # The image is expanded
    array_expand = np.zeros((np.shape(array)[0] + 2, np.shape(array)[1] + 2))
    array_expand[:] = np.nan
    array_expand[1:-1, 1:-1] = array

    # The hotpix map is expanded
    hotpix_expand = np.zeros((np.shape(array)[0] + 2, np.shape(array)[1] + 2))
    hotpix_expand[1:-1, 1:-1] = hotpix

    wh_dead = np.where(hotpix_expand == 1)
    # we first nan the hot pix in case there are several close to each others
    array_expand[wh_dead] = np.nan
    # The expanded array is copied
    array_expand_copy = array_expand.copy()

    #At hot pixel locations, the pixel value is equal to the mean of the eight pixels around
    for numdead in range(len(wh_dead[0])):
        i = wh_dead[0][numdead]
        j = wh_dead[1][numdead]
        array_expand[i, j] = np.nanmean([
            array_expand_copy[i - 1, j], array_expand_copy[i + 1, j], array_expand_copy[i, j - 1],
            array_expand_copy[i, j + 1], array_expand_copy[i - 1, j - 1], array_expand_copy[i + 1, j + 1],
            array_expand_copy[i - 1, j + 1], array_expand_copy[i + 1, j - 1]
        ])

    # finally we remove the nans that can happens if you have very large zones of hot pixs
    array_expand[np.isnan(array_expand)] = 0

    return array_expand[1:-1, 1:-1]


orig_data = fits.getdata(
    "/Users/jmazoyer/Downloads/efc-jan24-2022/Experiment0019_iter0_coro_image_047_0001.fits")[0]#, :100, 1024:1124]

dark = fits.getdata(
    "/Users/jmazoyer/Downloads/efc-jan24-2022/SPHERE_BKGRD_EFC_1s_043_0002.fits")[0]#, :100, 1024:1124]

# start_time = time.time()
# Noisy_pix = FindNoisyPix(dark, 3,2)
# fits.writeto("/Users/jmazoyer/Desktop/Noisy_pix.fits", Noisy_pix, overwrite=True)

threshold_bad_pix = 1000

above_threshold_pix =  orig_data*0
above_threshold_pix[np.where(dark > threshold_bad_pix)] = 1
dark[np.where(dark > threshold_bad_pix)] = np.nan
fits.writeto("/Users/jmazoyer/Desktop/above_threshold_pix.fits", above_threshold_pix, overwrite=True)


remaining_noisy_pix = orig_data*0
remaining_noisy_pix[np.where(dark - np.nanmean(dark)> 3* np.nanstd(dark))] = 1
fits.writeto("/Users/jmazoyer/Desktop/remaining_noisy_pix.fits", remaining_noisy_pix, overwrite=True)

Noisy_pix = np.clip(above_threshold_pix +remaining_noisy_pix, 0,1 )

fits.writeto("/Users/jmazoyer/Desktop/all_noisy_pix.fits", Noisy_pix, overwrite=True)

# Noisy_pix_bis = orig_data*0
# Noisy_pix_bis[np.where(dark - np.nanmean(dark)> 3* np.nanstd(dark))] = 1

# fits.writeto("/Users/jmazoyer/Desktop/Noisy_pix_bis.fits", Noisy_pix_bis, overwrite=True)

# asd

filter_data = mean_window_8pix(orig_data, Noisy_pix)
# print(time.time() - start_time)

fits.writeto("/Users/jmazoyer/Desktop/orig.fits", orig_data, overwrite=True)
fits.writeto("/Users/jmazoyer/Desktop/dark.fits", dark, overwrite=True)

fits.writeto("/Users/jmazoyer/Desktop/filter.fits", filter_data, overwrite=True)
