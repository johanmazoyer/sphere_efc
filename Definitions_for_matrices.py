# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:57:10 2021

@author: apotier
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from poppy import matrixDFT

fft = np.fft.fft2
ifft = np.fft.ifft2
shift = np.fft.fftshift
ishift=np.fft.ifftshift


def Upload_CoroConfig(ModelDirectory, coro, wavelength):
    """
    

    Parameters
    ----------
    ModelDirectory : path to model directory
    coro : coro type
    wavelength : 

    Returns
    -------
    mask384 : 2D apodizer
    Pup384 : 2D pupil
    ALC : 2D focal plane mask
    Lyot384: 2D Lyot stop

    """
    if coro == 'APLC':
        mask384 = fits.getdata(ModelDirectory+'apod-4.0lovD_384-192.fits')
        Pup384 = fits.getdata(ModelDirectory+'generated_VLT_pup_384-192.fits')
        ALC = 'ALC2'
        Lyot384 = fits.getdata(ModelDirectory+'sphere_stop_ST_ALC2.fits')
        
    elif coro == 'FQPM':
        #used to define the sampling of the focal images
        mask384 = roundpupil(384,int(384/2))#fits.getdata(ModelDirectory+'apod-4.0lovD_384-192.fits')
        # VLT pupil
        Pup384 = fits.getdata(ModelDirectory+'generated_VLT_pup_384-192.fits')
        ALC=''
        # Lyot stop (AXEL: NEED TO BE UPDATED WITH THE FQPM LYOT FUNCTION)
        Lyot384 = fits.getdata(ModelDirectory+'sphere_stop_ST_ALC2.fits')
    return mask384, Pup384, ALC, Lyot384


def roundpupil(nbpix,prad1):
    """
    Generate unobstructed pupil

    Parameters
    ----------
    nbpix : size image
    prad1 : pupil radius in pixels

    Returns
    -------
    pupilnormal : 2D pupil binary

    """
    xx, yy = np.meshgrid(np.arange(nbpix)-(nbpix)/2, np.arange(nbpix)-(nbpix)/2)
    rr     = np.hypot(yy, xx)
    pupilnormal = np.zeros((nbpix,nbpix))
    pupilnormal[rr<=prad1] = 1.
    return pupilnormal


def SaveFits(image,head,doc_dir2,name,replace=False):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    head : TYPE
        DESCRIPTION.
    doc_dir2 : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    replace : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdr = hdul[0].header
    hdr.set(head[0],head[1])
    hdu.writeto(doc_dir2+name+'.fits', overwrite=replace)



#RGa
def definition_isz(pupsizetmp,wave):
    """
    Extract relevant data 

    Parameters
    ----------
    pupsizetmp : Entrance pupil size in pix (should be 384)
    wave : Wavelength

    Returns
    -------
    isz: lyot plane image size
    ld_mas: l/D in milliarcsec

    """
    pupsizeinmeter = 8 #Pupsizeinmeter

    #Raccourcis conversions angles
    d2rad    = np.pi / 180.0 # degree to radian conversion factor
    d2arcsec = 3600
    arcsec2rad = d2rad/d2arcsec  # radian to milliarcsecond conversion factor
    
    #SPHERE detector resol
    resolinarcsec_pix = 12.25e-3  #arcsec/pix
    resolinrad_pix = resolinarcsec_pix*arcsec2rad  #rad/pix
    resolinpix_rad = 1 / resolinrad_pix     #pix/rad
    
    ld_rad = wave / pupsizeinmeter #lambda/D en radian
    ld_p = ld_rad * resolinpix_rad  #lambda/D en pixel
    ld_mas = ld_rad / arcsec2rad *1e3 #lambda/D en milliarcsec
    isz=int(pupsizetmp*ld_p)#-1 # Nombre de pixels dans le plan pupille pour atteindre la résolution ld_p voulue
    return [isz,ld_mas]

def pupiltodetector(input_wavefront, wave, lyot_mask, Name_ALC, isz_foc, coro, pupparf=False):
    """
    Propagate E-field from entrance pupil plane to detector

    Parameters
    ----------
    input_wavefront : E-field in entrance pupil
    wave : wavelength
    lyot_mask : 2D Lyot mask
    Name_ALC : Lyot focal mask
    isz_foc : Processed image size on detector
    coro : coronagraph type
    pupparf : boolean representing perfect pupil

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    detector_img: image on detector

    """
    
    pup_shape = input_wavefront.shape
    # Size of pupil in pixel (384)
    pupsize = pup_shape[0]
    # Nombre de pixels dans le plan pupille pour atteindre la résolution ld_p voulue
    [isz_pup,ld_mas] = definition_isz(pupsize, wave)

    if coro == 'APLC':

        if Name_ALC == 'ALC2':
            # Taille masque corono en lambda/D ALC2
            radFPMinld=92.5/ld_mas 
        elif Name_ALC == 'ALC3':
            # Taille masque corono en lambda/D ALC3
            radFPMinld=120/ld_mas  
        else:
           raise ValueError("ALC name unsupported")

        # [pixels/(l/D)] sampling factor of the computed focal plane field with the MFT
        mft_sampling = 100
        # focal plane field of view for computing the field. REQUIRED??
        occulter_fov = 5
        # radius of the occulting mask in pixels.
        occ_rad_pix = radFPMinld * mft_sampling
        # force even dimension
        npix         = (int(np.round(occulter_fov * mft_sampling)) // 2 ) *2
        # focal plane field of view for computing the field.
        occulter_fov = float(npix) / float(mft_sampling)
        
        # Computing focal plane image
        focal_plane = matrixDFT.matrix_dft(input_wavefront, occulter_fov, npix, centering='FFTSTYLE')
    
        # Computing occulted area in pixel
        occulter_area = roundpupil(npix, occ_rad_pix)
        # Computing rejected lyot plane
        lyot_plane_rejected = matrixDFT.matrix_dft(focal_plane*occulter_area, occulter_fov, pup_shape, inverse=True, centering='FFTSTYLE')
    
        # E-field Lyot stop
        before_lyot_stop = (input_wavefront - lyot_plane_rejected)
        after_lyot_stop = before_lyot_stop * lyot_mask
        
        # Resample Lyot stop
        after_lyot_stop2 = zeropad(after_lyot_stop, isz_pup)  
        
        
    elif coro == 'FQPM':
        # Proper sampling of input entrance pupil E-field and Lyot
        input_wavefront = zeropad(input_wavefront, isz_pup)
        lyot_mask = zeropad(lyot_mask, isz_pup)
        # Centering between 4 pixels
        PSFcentering = translationFFT(isz_pup,.5,.5)
        # Create FQPM
        fqpm_mask = create_fqpm(isz_pup)
        
        # Compute Lyot stop E-field
        before_lyot_stop = goto_pupil(goto_focal(input_wavefront*PSFcentering)*fqpm_mask)
     
        # Return perfect pupil for FQPM (i.e. pupil that null the E-field in the focal plane with no aberration)
        # Axel: I think it is unnecessary
        if pupparf == True:
            pupperf = (shift(ifft(fft(shift(before_lyot_stop*(1-lyot_mask)))*fqpm_mask))*np.conjugate(PSFcentering))#[int(isz_pup/2-pupsize/2):int(isz_pup/2+pupsize/2),int(isz_pup/2-pupsize/2):int(isz_pup/2+pupsize/2)]
            return cropimage(pupperf, isz_pup/2, isz_pup/2, pupsize)
 
        # Filtering Lyot stop E field with Lyot stop pupil
        after_lyot_stop2 = before_lyot_stop * lyot_mask * np.conjugate(PSFcentering)
 
        
    else:
        print('coro should be APLC or FQPM')

    # Propagate LS to detector
    detector_img = goto_focal(after_lyot_stop2)#[int(isz_pup/2-isz_foc/2):int(isz_pup/2+isz_foc/2),int(isz_pup/2-isz_foc/2):int(isz_pup/2+isz_foc/2)]
    
    # Crop image to relevant size
    detector_img = cropimage(detector_img, isz_pup/2, isz_pup/2, isz_foc)
    
    return detector_img
    
def goto_focal(pupil_plane):
    """
    Propagate E-field from pupil plane to focal plane

    Parameters
    ----------
    pupil_plane : TYPE
        DESCRIPTION.

    Returns
    -------
    focal_plane : TYPE
        DESCRIPTION.

    """
    pupil_plane = shift(pupil_plane)
    focal_plane = fft(pupil_plane)
    focal_plane = shift(focal_plane)
    return focal_plane

def goto_pupil(focal_plane):
    """
    Propagate E-field from focal plane to pupil plane

    Parameters
    ----------
    focal_plane : TYPE
        DESCRIPTION.

    Returns
    -------
    pupil_plane : TYPE
        DESCRIPTION.

    """
    focal_plane = ishift(focal_plane)
    pupil_plane = ifft(focal_plane)
    pupil_plane = ishift(pupil_plane)
    return pupil_plane
    
    
    
def invertDSCC(interact, cut ,goal='e', regul="truncation", visu=False):
    """
    Invert with SVD + regularization

    Parameters
    ----------
    interact : TYPE
        DESCRIPTION.
    cut : TYPE
        DESCRIPTION.
    goal : TYPE, optional
        DESCRIPTION. The default is 'e'.
    regul : TYPE, optional
        DESCRIPTION. The default is "truncation".
    visu : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    list
        DESCRIPTION.

    """
    U, s, V = np.linalg.svd(interact, full_matrices=False)
    S = np.diag(s)
    InvS=np.linalg.inv(S)
    if(visu==True):
        plt.plot(np.diag(InvS),'r.')
        plt.yscale('log')

        
    if goal == 'e':
        InvS[np.where(InvS>cut)]=0
    
    
    if goal == "c":
        if regul == "truncation":
            InvS[cut:] = 0
        if regul == "tikhonov":
            InvS = np.diag(s / (s**2 + s[cut]**2))
            if visu == True:
                plt.plot(np.diag(InvS), "b.")
                plt.yscale("log")
                # plt.show()

    plt.show()                           
    pseudoinverse = np.dot(np.dot(np.transpose(V),InvS),np.transpose(U))
    return [np.diag(InvS),pseudoinverse]


def createvectorprobes(input_wavefront, wave, lyot_mask , Name_ALC , isz_foc, pushact, posprobes , cutsvd, coro):
    """
    Create PW matrix

    Parameters
    ----------
    input_wavefront : TYPE
        DESCRIPTION.
    wave : TYPE
        DESCRIPTION.
    lyot_mask : TYPE
        DESCRIPTION.
    Name_ALC : TYPE
        DESCRIPTION.
    isz_foc : TYPE
        DESCRIPTION.
    pushact : TYPE
        DESCRIPTION.
    posprobes : TYPE
        DESCRIPTION.
    cutsvd : TYPE
        DESCRIPTION.
    coro : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    # Initialize parameters
    pup_shape = input_wavefront.shape
    # Size of pupil in pixel (384)
    pupsize = pup_shape[0]
    numprobe = len(posprobes)
    deltapsik = np.zeros((numprobe , isz_foc , isz_foc),dtype=complex)
    probephase = np.zeros((numprobe , pupsize , pupsize))
    matrix = np.zeros((numprobe,2))
    Vecteurenvoi = np.zeros((isz_foc**2,2,numprobe))
    SVD = np.zeros((2,isz_foc,isz_foc))
    
    
    [isz_pup,ld_mas] = definition_isz(pupsize, wave)
    # Create off-axis PSF
    maskoffaxis = cropimage(translationFFT(isz_pup,30,30), int(isz_pup/2), int(isz_pup)/2, pupsize)
    OffAxisPSF = pupiltodetector(maskoffaxis*input_wavefront, wave, lyot_mask , Name_ALC , isz_foc,coro)
    squaremaxPSF = np.amax(np.abs(OffAxisPSF))

    # Regularization
    cutsvd = 0.3*squaremaxPSF*8/(400/37)
    
    # Get constant E-field in the detector with corono
    pupilnoabb = pupiltodetector(input_wavefront , wave , lyot_mask , Name_ALC , isz_foc,coro)

    k=0
    for i in posprobes:
        print(i)
        # Poke one actuator and compute complex entrance pupil
        probephase[k] = pushact[i]
        probephase[k] = 2*np.pi*(probephase[k])*1e-9/wave
        #entrance pupil plane field (can be real, or complex with amplitude and phase)
        input_wavefront_k = input_wavefront*(1+1j*probephase[k])
        # Propagate input through the coronagraph, remove constant E-field and normalize with PSF
        deltapsikbis = pupiltodetector(input_wavefront_k, wave,lyot_mask,Name_ALC,isz_foc,coro)
        deltapsik[k] = (deltapsikbis-pupilnoabb)/squaremaxPSF
        k=k+1

    # invert matrix
    l=0
    for i in np.arange(isz_foc):
        for j in np.arange(isz_foc):
            matrix[:,0] = np.real(deltapsik[:,i,j])
            matrix[:,1] = np.imag(deltapsik[:,i,j])
            try:
                SVD[:,i,j] = invertDSCC(matrix,cutsvd,visu=False)[0]
                Vecteurenvoi[l] = invertDSCC(matrix,cutsvd,visu=False)[1]
            except:
                print('Careful: Error! for l='+str(l))
                SVD[:,i,j] = np.zeros(2)
                Vecteurenvoi[l] = np.zeros((2,numprobe))
            l = l+1  
    return [Vecteurenvoi,SVD]
    

def creatingWhichinPupil(pupil, pushact, cutinpupil):
    """
    Compute an array with actuator indices that are located inside the pupil

    Parameters
    ----------
    pupil : TYPE
        DESCRIPTION.
    pushact : TYPE
        DESCRIPTION.
    cutinpupil : TYPE
        DESCRIPTION.

    Returns
    -------
    WhichInPupil : TYPE
        DESCRIPTION.

    """
    WhichInPupil = []
    for i in np.arange(len(pushact)):
        Psivector = - pushact[i]
        cut = cutinpupil * np.sum(-pushact[182])
        if(np.sum(Psivector * pupil) > cut):
            WhichInPupil.append(i)
    
    WhichInPupil = np.array(WhichInPupil)
    return WhichInPupil
    
    
    
def creatingMaskDH(dimimages,
                   shape,
                   choosepixDH=[8, 35, -35, 35],
                   circ_rad=[8, 10],
                   circ_side="Full",
                   circ_offset=8,
                   circ_angle=0):
    """ --------------------------------------------------
    Create a binary mask.
    
    Parameters:
    ----------
    dimimages: int, size of the output squared mask
    shape: string, can be 'square' or 'circle' , define the shape of the binary mask.
    choosepixDH: 1D array, if shape is 'square', define the edges of the binary mask in pixels.
    circ_rad: 1D array, if shape is 'circle', define the inner and outer edge of the binary mask
    circ_side: string, if shape is 'circle', can define to keep only one side of the circle
    circ_offset : float, remove pixels that are closer than circ_offset if circ_side is set
    circ_angle : float, if circ_side is set, remove pixels within a cone of angle circ_angle

    Return:
    ------
    maskDH: 2D array, binary mask
    -------------------------------------------------- """
    xx, yy = np.meshgrid(
        np.arange(dimimages) - (dimimages) / 2,
        np.arange(dimimages) - (dimimages) / 2)
    rr = np.hypot(yy, xx)
    if shape == "square":
        maskDH = np.ones((dimimages, dimimages))
        maskDH[xx < choosepixDH[0]] = 0
        maskDH[xx > choosepixDH[1]] = 0
        maskDH[yy < choosepixDH[2]] = 0
        maskDH[yy > choosepixDH[3]] = 0
    if shape == "circle":
        maskDH = np.ones((dimimages, dimimages))
        maskDH[rr >= circ_rad[1]] = 0
        maskDH[rr < circ_rad[0]] = 0
        if circ_side == "Right":
            maskDH[xx < np.abs(circ_offset)] = 0
            if circ_angle != 0:
                maskDH[yy - xx / np.tan(circ_angle * np.pi / 180) > 0] = 0
                maskDH[yy + xx / np.tan(circ_angle * np.pi / 180) < 0] = 0
        if circ_side == "Left":
            maskDH[xx > -np.abs(circ_offset)] = 0
            if circ_angle != 0:
                maskDH[yy - xx / np.tan(circ_angle * np.pi / 180) < 0] = 0
                maskDH[yy + xx / np.tan(circ_angle * np.pi / 180) > 0] = 0
        if circ_side == "Bottom":
            maskDH[yy < np.abs(circ_offset)] = 0
            if circ_angle != 0:
                maskDH[yy - xx * np.tan(circ_angle * np.pi / 180) < 0] = 0
                maskDH[yy + xx * np.tan(circ_angle * np.pi / 180) < 0] = 0
        if circ_side == "Top":
            maskDH[yy > -np.abs(circ_offset)] = 0
            if circ_angle != 0:
                maskDH[yy - xx * np.tan(circ_angle * np.pi / 180) > 0] = 0
                maskDH[yy + xx * np.tan(circ_angle * np.pi / 180) > 0] = 0
    return maskDH
    
    
    
def creatingCorrectionmatrix(input_wavefront, wave, lyot_mask , Name_ALC , isz_foc, pushact, Whichact, coro):
    """
    Create full jacobian

    Parameters
    ----------
    input_wavefront : TYPE
        DESCRIPTION.
    wave : TYPE
        DESCRIPTION.
    lyot_mask : TYPE
        DESCRIPTION.
    Name_ALC : TYPE
        DESCRIPTION.
    isz_foc : TYPE
        DESCRIPTION.
    pushact : TYPE
        DESCRIPTION.
    Whichact : TYPE
        DESCRIPTION.
    coro : TYPE
        DESCRIPTION.

    Returns
    -------
    Gmatrixbis : TYPE
        DESCRIPTION.

    """
    pup_shape = input_wavefront.shape
    # Size of pupil in pixel (384)
    pupsize = pup_shape[0]
    [isz_pup,ld_mas] = definition_isz(pupsize, wave)
    # Create off-axis PSF
    maskoffaxis = cropimage(translationFFT(isz_pup,30,30), int(isz_pup/2), int(isz_pup)/2, pupsize)
    OffAxisPSF = pupiltodetector(maskoffaxis*input_wavefront, wave, lyot_mask , Name_ALC , isz_foc, coro)
    squaremaxPSF = np.amax(np.abs(OffAxisPSF))
    
    # Get constant E-field in the detector with corono
    pupilnoabb = pupiltodetector(input_wavefront , wave , lyot_mask , Name_ALC , isz_foc, coro)
    
    # Jacobian calculation cube (Real-Imag, pixels, nb modes)
    Gmatrixbis=np.zeros((2,int(isz_foc*isz_foc),len(Whichact)))

    k=0
    for i in Whichact:
        print(i)
        # Poke one actuator and compute complex entrance pupil
        Psivector = pushact[i]
        Psivector = 2*np.pi*(Psivector)*1e-9/wave
        # Entrance pupil plane field (can be real, or complex with amplitude and phase)
        input_wavefront_k = input_wavefront*(1+1j*Psivector)
        # Propagate input through the coronagraph, remove constant E-field and normalize with PSF
        Gvectorbisbis = (pupiltodetector(input_wavefront_k , wave , lyot_mask , Name_ALC , isz_foc, coro)- pupilnoabb)/squaremaxPSF
    
        # Fill jacobian with real data
        Gmatrixbis[0,:,k] = np.real(Gvectorbisbis).flatten()
        
        # Fill jacobian with imag data
        Gmatrixbis[1,:,k] = np.imag(Gvectorbisbis).flatten()
        k=k+1
    return Gmatrixbis


def get_masked_jacobian(complex_jacobian, mask):
    """
    Extract DH pixels in jacobian calculated with creatingCorrectionmatrix

    Parameters
    ----------
    complex_jacobian : (Real-Imag, pixels, nb modes)
    mask : TYPE
        DESCRIPTION.

    Returns
    -------
    masked_jacobian : TYPE
        DESCRIPTION.

    """
    mask_flattened = mask.flatten()
    # Create jacobian in good dimensions (real values + imag values, nb modes)
    masked_jacobian = np.zeros((2*int(np.sum(mask)),complex_jacobian.shape[2]))
    
    # Fill in jacobian
    masked_jacobian[0:int(np.sum(mask))] = complex_jacobian[0,np.where(mask_flattened)]
    masked_jacobian[int(np.sum(mask)):] = complex_jacobian[1,np.where(mask_flattened)]
    
    return masked_jacobian
        
    


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


def zeropad(tab,dim):
    """
    Zero padding around image

    Parameters
    ----------
    tab : image
    dim : new image dimension

    Returns
    -------
    newtab : image in new dimension, padded with zeroes

    """
    newtab = np.zeros((dim,dim),dtype=complex)
    left = int(dim/2-tab.shape[0]/2)
    right = int(dim/2+tab.shape[0]/2)
    bottom = int(dim/2-tab.shape[1]/2)
    top = int(dim/2+tab.shape[1]/2)
    newtab[left:right,bottom:top] = tab
    return newtab

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

def create_fqpm(isz_pup): 
    """
    Create Four Quadrant Phase Mask of size isz_pup

    Parameters
    ----------
    isz_pup : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """       
    fqpm_mask = np.ones((isz_pup,isz_pup))
    fqpm_mask[0:int(isz_pup/2),0:int(isz_pup/2)] = -1
    fqpm_mask[int(isz_pup/2):isz_pup,int(isz_pup/2):isz_pup] = -1
    return fqpm_mask
    