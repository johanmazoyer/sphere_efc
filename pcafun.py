import numpy as np

from SPHERE_EFC_Func import roundpupil

# pip install pyklip
from pyklip.klip import rotate

import scipy.linalg as la

from astropy.io import fits



# Fonction de pyklip ecrite par J Wang. Je la colle ici pour que vous pusiseiz vraiment bouger des
# petits paramètres si vous pensez que c'est necessaire. C'est mieux que pas mal de truc qu'on peut trouver car
# - on peut faire de manière très rapide un pca subtraction abec plein de modes PCA différents
# - on peut avoir des nans dans l'image si on veut cahcher des pixesl mort
# - il traite aussi les valeurs eigen de la matrice de covar negatives ce qui peut arriver
# Normalement vous n'avaiez pas à utiliser celle cit je vous ai mis un "wrapper" avec une fonciton qui "flatten" votre zone
# et qui vous sort les valeus qui vous interesent

def klip_math(sci, ref_psfs, numbasis, covar_psfs=None, return_basis=False, return_basis_and_eig=False):
    """
    Helper function for KLIP that does the linear algebra

    Args:
        sci: array of length p containing the science data
        ref_psfs: N x p array of the N reference PSFs that
                  characterizes the PSF of the p pixels
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length b)
        covar_psfs: covariance matrix of reference psfs passed in so you don't have to calculate it here
        return_basis: If true, return KL basis vectors (used when onesegment==True)
        return_basis_and_eig: If true, return KL basis vectors as well as the eigenvalues and eigenvectors of the
                                covariance matrix. Used for KLIP Forward Modelling of Laurent Pueyo.

    Returns:
        sub_img_rows_selected: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                               cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p
        KL_basis: array of shape (max(numbasis),p). Only if return_basis or return_basis_and_eig is True.
        evals: Eigenvalues of the covariance matrix. The covariance matrix is assumed NOT to be normalized by (p-1).
                Only if return_basis_and_eig is True.
        evecs: Eigenvectors of the covariance matrix. The covariance matrix is assumed NOT to be normalized by (p-1).
                Only if return_basis_and_eig is True.
    """
    # for the science image, subtract the mean and mask bad pixels
    sci_mean_sub = sci - np.nanmean(sci)
    # sci_nanpix = np.where(np.isnan(sci_mean_sub))
    # sci_mean_sub[sci_nanpix] = 0

    # do the same for the reference PSFs
    # playing some tricks to vectorize the subtraction
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    # calculate the covariance matrix for the reference PSFs
    # note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    # we have to correct for that a few lines down when consturcting the KL
    # vectors since that's not part of the equation in the KLIP paper
    if covar_psfs is None:
        covar_psfs = np.cov(ref_psfs_mean_sub)

    # maximum number of KL modes
    tot_basis = covar_psfs.shape[0]

    # only pick numbasis requested that are valid. We can't compute more KL basis than there are reference PSFs
    # do numbasis - 1 for ease of indexing since index 0 is using 1 KL basis vector
    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate

    # calculate eigenvalues and eigenvectors of covariance matrix, but only the ones we need (up to max basis)
    evals, evecs = la.eigh(covar_psfs, subset_by_index=(tot_basis-max_basis, tot_basis-1))

    # check if there are negative eignevalues as they will cause NaNs later that we have to remove
    # the eigenvalues are ordered smallest to largest
    #check_nans = evals[-1] < 0 # currently this checks that *all* the evals are neg, but we want just one.
    # also, include 0 because that is a bad value too
    check_nans = np.any(evals <= 0) # alternatively, check_nans = evals[0] <= 0

    # scipy.linalg.eigh spits out the eigenvalues/vectors smallest first so we need to reverse
    # we're going to recopy them to hopefully improve caching when doing matrix multiplication
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication

    # keep an index of the negative eignevalues for future reference if there are any
    if check_nans:
        neg_evals = (np.where(evals <= 0))[0]

    # calculate the KL basis vectors
    kl_basis = np.dot(ref_psfs_mean_sub.T, evecs)
    # JB question: Why is there this [None, :]? (It adds an empty first dimension)
    kl_basis = kl_basis * (1. / np.sqrt(evals * (np.size(sci) - 1)))[None, :]  #multiply a value for each row

    # sort to KL basis in descending order (largest first)
    # kl_basis = kl_basis[:,eig_args_all]

    # duplicate science image by the max_basis to do simultaneous calculation for different k_KLIP
    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis, 1))
    sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis), 1)) # this is the output image which has less rows

    # bad pixel mask
    # do it first for the image we're just doing computations on but don't care about the output
    sci_nanpix = np.where(np.isnan(sci_mean_sub_rows))
    sci_mean_sub_rows[sci_nanpix] = 0
    # now do it for the output image
    sci_nanpix = np.where(np.isnan(sci_rows_selected))
    sci_rows_selected[sci_nanpix] = 0

    # do the KLIP equation, but now all the different k_KLIP simultaneously
    # calculate the inner product of science image with each of the different kl_basis vectors
    # TODO: can we optimize this so it doesn't have to multiply all the rows because in the next lines we only select some of them
    inner_products = np.dot(sci_mean_sub_rows, np.require(kl_basis, requirements=['F']))
    # select the KLIP modes we want for each level of KLIP by multiplying by lower diagonal matrix
    lower_tri = np.tril(np.ones([max_basis, max_basis]))
    inner_products = inner_products * lower_tri
    # if there are NaNs due to negative eigenvalues, make sure they don't mess up the matrix multiplicatoin
    # by setting the appropriate values to zero
    if check_nans:
        needs_to_be_zeroed = np.where(lower_tri == 0)
        inner_products[needs_to_be_zeroed] = 0
        # make a KLIP PSF for each amount of klip basis, but only for the amounts of klip basis we actually output
        kl_basis[:, neg_evals] = 0
        klip_psf = np.dot(inner_products[numbasis,:], kl_basis.T)
        # for KLIP PSFs that use so many KL modes that they become nans, we have to put nan's back in those
        badbasis = np.where(numbasis >= np.min(neg_evals)) #use basis with negative eignevalues
        klip_psf[badbasis[0], :] = np.nan
    else:
        # make a KLIP PSF for each amount of klip basis, but only for the amounts of klip basis we actually output
        klip_psf = np.dot(inner_products[numbasis,:], kl_basis.T)

    # make subtracted image for each number of klip basis
    sub_img_rows_selected = sci_rows_selected - klip_psf

    # restore NaNs
    sub_img_rows_selected[sci_nanpix] = np.nan


    if return_basis is True:
        return sub_img_rows_selected.transpose(), kl_basis.transpose()
    elif return_basis_and_eig is True:
        return sub_img_rows_selected.transpose(), kl_basis.transpose(),evals*(np.size(sci)-1), evecs
    else:
        return sub_img_rows_selected.transpose()


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
