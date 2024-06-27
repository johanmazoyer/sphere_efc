import numpy as np

from SPHERE_EFC_Func import roundpupil

# pip install pyklip
from pyklip.klip import klip_math

from astropy.io import fits


def simple_pca(science_im, cube_ref, numbasis, zone=None):
    """
    Simple function for PCA that does the linear algebra based on pyklip package

    Args:
        sci: array of length (a , b)  containing the science data
        ref_psfs: (N, a , b) array of the N reference PSFs that characterizes the PSF of the p pixels
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length K)
        zone: binary array of length a x b containing the zoen on which we do the PCA.
                    If None, all image is used


    Returns:
        PCA_proj : array of shape (K , a , b) that is the sci image projected onto the PCA basis
                                    for each of the K PCA basis. Eq 8 in Soummer et al. 2012
        sci_img_sub: array of shape (K , a , b) that is the PSF subtracted data for each of the K PCA basis
                               cutoffs. Eq 9 in Soummer et al. 2012
        PCA_basis: array of shape (max(numbasis) , a , b). Eq 5 in Soummer et al 2012

    """
    PCA_proj_2d = np.zeros((len(numbasis), cube_ref.shape[1], cube_ref.shape[2]))
    im_subctracted2d = np.zeros((len(numbasis), cube_ref.shape[1], cube_ref.shape[2]))
    pca_basis2d = np.zeros((np.max(numbasis), cube_ref.shape[1], cube_ref.shape[2]))

    if zone is None:
        zone = np.ones((cube_ref.shape[1], cube_ref.shape[2]))

    im_sci_flat = science_im[zone == 1].flatten()
    my_data_fl = np.zeros((cube_ref.shape[0], np.sum(zone).astype(int)))
    for i in range(cube_ref.shape[0]):
        my_data_fl[i, :] = cube_ref[i, zone == 1].flatten()

    im_subctracted_flatten, pca_basis_flatten = klip_math(im_sci_flat,
                                                          my_data_fl,
                                                          np.array(numbasis),
                                                          return_basis=True,
                                                          return_basis_and_eig=False)

    for i in range(len(numbasis)):
        im2d = np.zeros((cube_ref.shape[1], cube_ref.shape[2]))
        im2d[zone == 1] = im_subctracted_flatten[:, i]
        im_subctracted2d[i] = im2d
        PCA_proj_2d[i] = science_im - im_subctracted2d[i]

    for i in range(np.max(numbasis)):
        vec2d = np.zeros((cube_ref.shape[1], cube_ref.shape[2]))
        vec2d[zone == 1] = pca_basis_flatten[i, :]
        pca_basis2d[i] = vec2d

    return PCA_proj_2d, im_subctracted2d, pca_basis2d


dir_data = "/Users/jmazoyer/Desktop/"
cube_data = fits.getdata(dir_data + "bad_atm_data_HD298936_DB_H23_2016-01-19_cropped.fits")

# cut data for the test
cube_data = cube_data[0:8]

radmin = 15
radmax = 70
zone_eval = roundpupil(cube_data.shape[1], radmax) * (1 - roundpupil(cube_data.shape[1], radmin))

# fits.writeto("/Users/jmazoyer/Desktop/dark_hole.fits", dark_hole, overwrite=True)
# print(toto.shape)

# we do a simple PCA of first images compared to all the rest
pca_projs_images, images_sub, pca_basis = simple_pca(cube_data[0], cube_data[1:], [2, 3, 5], zone=zone_eval)

fits.writeto(dir_data + "pca_projs_images.fits", pca_projs_images, overwrite=True)
fits.writeto(dir_data + "images_sub.fits", images_sub, overwrite=True)
fits.writeto(dir_data + "pca_basis.fits", pca_basis, overwrite=True)
