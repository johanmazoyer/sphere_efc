import numpy as np

from SPHERE_EFC_Func import roundpupil

# pip install pyklip
from pyklip.klip import klip_math, rotate

from astropy.io import fits


def simple_pca(science_im, cube_ref, numbasis=None, zone=None):
    """
    Simple function for PCA that does the linear algebra based on pyklip package

    Args:
        sci: array of length (a , b)  containing the science data
        ref_psfs: (N, a , b) array of the N reference PSFs that characterizes the PSF of the p pixels
        numbasis: number of PCA basis vectors to measure (can be an int or an array of ints of length K).
                        if None, all the basis vectors are computed.
        zone: binary array of length a x b containing the zoen on which we do the PCA.
                    If None, all image is used


    Returns:
        PCA_proj : array of shape (K , a , b) that is the sci image projected onto the PCA basis
                                    for each of the K PCA basis. Eq 8 in Soummer et al. 2012
        sci_img_sub: array of shape (K , a , b) that is the PSF subtracted data for each of the K PCA basis
                               cutoffs. Eq 9 in Soummer et al. 2012
        PCA_basis: array of shape (max(numbasis) , a , b). Eq 5 in Soummer et al 2012

    """

    if numbasis is None:
        numbasis = np.arange(cube_ref.shape[0])

    if np.max(numbasis) > cube_ref.shape[0]:
        print("a number in numbasis is bigger than the number of reference PSFs")
        raise (SystemExit)

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
        im2d[zone == 0] = np.nan
        im_subctracted2d[i] = im2d
        PCA_proj_2d[i] = science_im - im_subctracted2d[i]

    for i in range(np.max(numbasis)):
        vec2d = np.zeros((cube_ref.shape[1], cube_ref.shape[2]))
        vec2d[zone == 1] = pca_basis_flatten[i, :]
        vec2d[zone == 0] = np.nan
        pca_basis2d[i] = vec2d

    return PCA_proj_2d, im_subctracted2d, pca_basis2d


# you can DL this beta pic data to test
# https://www.dropbox.com/scl/fi/jwxxleuaoy1pxdt7ar09l/datacube_SPHERE_binned_centered.fits?rlkey=l8p39jqje7zinswqpkdwj1zdn&dl=0

# you can DL parangs
# https://www.dropbox.com/scl/fi/ach6l0vh9xiuh4aghe4qz/parangs_binned.fits?rlkey=0behz4wvyahz3is2o19rlqkly&dl=0

dir_data = "/Users/jmazoyer/Desktop/test_pca_fold/"
cube_data = fits.getdata(dir_data + "datacube_SPHERE_binned_centered.fits")

# cut data for the test
# cube_data = cube_data[0:30]

radmin = 15
radmax = 70
zone_eval = roundpupil(cube_data.shape[1], radmax) * (1 - roundpupil(cube_data.shape[1], radmin))

# fits.writeto("/Users/jmazoyer/Desktop/dark_hole.fits", dark_hole, overwrite=True)
# print(toto.shape)

pca_modes = [1, 5, 15, 20]

# we do a simple PCA of first images compared to all the rest
pca_projs_images, images_sub, pca_basis = simple_pca(cube_data[0], cube_data[1:], pca_modes, zone=zone_eval)

fits.writeto(dir_data + "pca_1st_images_proj.fits", pca_projs_images, overwrite=True)
fits.writeto(dir_data + "1st_images_sub.fits", images_sub, overwrite=True)
fits.writeto(dir_data + "pca_basis_all_but_1st.fits", pca_basis, overwrite=True)

# if you want just the PCA basis of all the elements
_, _, pca_basis = simple_pca(np.zeros((cube_data.shape[1], cube_data.shape[2])),
                             cube_data, [cube_data.shape[0]],
                             zone=zone_eval)
fits.writeto(dir_data + "pca_basis_all.fits", pca_basis, overwrite=True)


# to do a full pca of a data set (indiv pca on each image + derotation + mean) for each
# number of PCA modes

# number of modes
number_pca_modes = len(pca_modes)
parangs = fits.getdata(dir_data + "parangs_binned.fits")
images_subs = np.zeros((cube_data.shape[0], number_pca_modes, cube_data.shape[1], cube_data.shape[2]))
images_subs_derot = np.zeros((cube_data.shape[0], number_pca_modes, cube_data.shape[1], cube_data.shape[2]))

# loop on the images of the cube
for i in range(cube_data.shape[0]):

    # to avoid self subctrastion I remove the images from which I do the subtraction from the ref cube
    index_except_i = np.delete(np.arange(cube_data.shape[0]), i)
    _, images_subs[i], _ = simple_pca(cube_data[i], cube_data[index_except_i], pca_modes, zone=zone_eval)

    # loop on the KL numbers
    for j in range(number_pca_modes):
        images_subs_derot[i, j] = rotate(images_subs[i, j], parangs[i], center=[100, 100])

fits.writeto(dir_data + "images_subs.fits", images_subs, overwrite=True)
fits.writeto(dir_data + "final_pca.fits", np.mean(images_subs_derot, axis=0), overwrite=True)
