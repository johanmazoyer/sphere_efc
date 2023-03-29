import numpy as np
import time
from astropy.io import fits
from scipy import ndimage

import glob

import numpy as np
from pathlib import Path

from SPHERE_EFC_Func import mean_window_8pix, find_hot_pix_in_dark, sigma_filter


folder_data = "/Users/jmazoyer/Dropbox/ExchangeFolder/efc_sphere_may23/20230328/"

outputdir = "/Users/jmazoyer/Desktop/test_badpix_new/"


dark1 = fits.getdata(folder_data + "SPHERE_BKGRD_EFC_1s_087_0001.fits")[0, 74:250, 1420:1600]
dark2 = fits.getdata(folder_data + "SPHERE_BKGRD_EFC_1s_087_0002.fits")[0, 74:250,1420:1600]

dark = (dark1 + dark2) / 2

# fits.writeto(outputdir + "dark1.fits", dark1, overwrite=True)
# fits.writeto(outputdir + "dark2.fits", dark2, overwrite=True)
# fits.writeto(outputdir + "dark.fits", dark, overwrite=True)

copy_dark_nan_pix = np.copy(dark)



# start_time = time.time()
# Noisy_pix = FindNoisyPix(dark, 3,2)
# fits.writeto(outputdir + "Noisy_pix.fits", Noisy_pix, overwrite=True)

threshold_sup_bad_pix = 100
threshold_inf_bad_pix = -100

above_threshold_pix = np.zeros(dark.shape)
above_threshold_pix[np.where(dark > threshold_sup_bad_pix)] = 1
# fits.writeto(outputdir + "above_threshold_pix.fits", above_threshold_pix, overwrite=True)

under_threshold_pix = np.zeros(dark.shape)
under_threshold_pix[np.where(dark < threshold_inf_bad_pix)] = 1
# fits.writeto(outputdir + "under_threshold_pix.fits", under_threshold_pix, overwrite=True)

copy_dark_nan_pix[np.where(dark > threshold_sup_bad_pix)] = np.nan
copy_dark_nan_pix[np.where(dark < threshold_inf_bad_pix)] = np.nan

remaining_noisy_pix_dark = np.zeros(dark.shape)
remaining_noisy_pix_dark[np.where(copy_dark_nan_pix - np.nanmean(copy_dark_nan_pix) > 3 * np.nanstd(copy_dark_nan_pix))] = 1
# fits.writeto(outputdir + "remaining_noisy_pix.fits", remaining_noisy_pix_dark, overwrite=True)

Noisy_pix = np.clip(above_threshold_pix + under_threshold_pix + remaining_noisy_pix_dark, 0, 1)

# fits.writeto(outputdir + "all_noisy_pix.fits", Noisy_pix, overwrite=True)


fullpath =  folder_data + "Experiment0002_iter0_coro_image_087_0001.fits"
list_file = glob.glob(folder_data + "*_coro_image_087_0001.fits")

for fullpath in list_file:

    orig_data = fits.getdata(fullpath)[0, 74:250, 1420:1600]
    filename = Path(fullpath).stem

    fits.writeto(outputdir + filename+"_darksub.fits", orig_data - dark, overwrite=True)


    # filter_data = mean_window_8pix(orig_data - dark, Noisy_pix)
    # fits.writeto(outputdir + filename+ "_nobadpix.fits", filter_data, overwrite=True)



    filter_data_real_func = mean_window_8pix(orig_data - dark, find_hot_pix_in_dark(dark))
    fits.writeto(outputdir + filename+ "_nobadpix_func.fits", filter_data_real_func, overwrite=True)



    # sigma_filt_ignore_edges = sigma_filter(filter_data, box_width=3, n_sigma=7, monitor=True, ignore_edges=True)
    sigma_filt_with_edges = sigma_filter(filter_data_real_func, box_width=3, n_sigma=7, monitor=True, ignore_edges=False)

    # fits.writeto(outputdir + "sigma_filt_ignore_edges.fits", sigma_filt_ignore_edges, overwrite=True)
    fits.writeto(outputdir + filename+ "_nobadpix_sigma_filt.fits" , sigma_filt_with_edges, overwrite=True)

