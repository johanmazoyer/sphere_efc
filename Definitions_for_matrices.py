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


def roundpupil(nbpix,prad1):
    xx, yy = np.meshgrid(np.arange(nbpix)-(nbpix)/2, np.arange(nbpix)-(nbpix)/2)
    rr     = np.hypot(yy, xx)
    pupilnormal=np.zeros((nbpix,nbpix))
    pupilnormal[rr<=prad1]=1.
    return pupilnormal


def SaveFits(image,head,doc_dir2,name,replace=False):
    hdu=fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdr=hdul[0].header
    hdr.set(head[0],head[1])
    hdu.writeto(doc_dir2+name+'.fits', overwrite=replace)

    
def increasepixelinimage(image,pixafter,cut):
    pixbefore=len(image)
    fftimagerescale=np.zeros((pixafter,pixafter),dtype=complex)
    fftimage=shift(fft(shift(image)))
    fftimagerescale[int(pixafter/2-pixbefore/2):int(pixafter/2+pixbefore/2),int(pixafter/2-pixbefore/2):int(pixafter/2+pixbefore/2)]=fftimage
    imagerescale=abs(shift(ifft(shift(fftimagerescale))))*(pixafter/pixbefore)**2
    imagerescale[np.where(imagerescale<cut)]=0
    return imagerescale

#RGa
def definition_isz(pupsizetmp,wave):
    pupsizeinmeter=8 #Pupsizeinmeter

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
    #print('pixel per resolution element:', ld_p)
    #ld_p=3.5
    isz=int(pupsizetmp*ld_p)#-1 # Nombre de pixels dans le plan pupille pour atteindre la résolution ld_p voulue
    return [isz,ld_mas]

def pupiltodetector(input_wavefront , wave, lyot_mask , Name_ALC , isz_foc,coro,PSFcentering,pupparf=False):
    
    pup_shape = input_wavefront.shape
    pupsize = pup_shape[0] #Size of pupil in pixel (384)

    if coro == 'APLC':
    
# Nombre de pixels dans le plan pupille pour atteindre la résolution ld_p voulue
        [isz_pup,ld_mas] = definition_isz(pupsize,wave)

        if Name_ALC == 'ALC2':
            radFPMinld=92.5/ld_mas #Taille masque corono en lambda/D ALC2
        elif Name_ALC == 'ALC3':
            radFPMinld=120/ld_mas  #Taille masque corono en lambda/D ALC3
        else:
           raise ValueError("ALC name unsupported")

    #print('Size Lyot mask in l/d:', radFPMinld)


        mft_sampling = 100 #[pixels/(l/D)] sampling factor of the computed focal plane field with the MFT
        occulter_fov = 5  #focal plane field of view for computing the field.
        occ_rad_pix = radFPMinld * mft_sampling    #radius of the occulting mask in pixels.
        npix         = (int(np.round(occulter_fov * mft_sampling)) // 2 ) *2 # force even dimension
        occulter_fov = float(npix) / float(mft_sampling)
    
        focal_plane = matrixDFT.matrix_dft(input_wavefront, occulter_fov, npix, centering='FFTSTYLE')
    
    
        occulter_area = roundpupil(npix, occ_rad_pix)
        lyot_plane_rejected = matrixDFT.matrix_dft(focal_plane*occulter_area,occulter_fov, pup_shape, inverse=True,centering='FFTSTYLE')
        focal_plane = focal_plane*(1.-.8*occulter_area)
    
        before_lyot_stop = (input_wavefront - lyot_plane_rejected)
        after_lyot_stop = before_lyot_stop * lyot_mask
    
    
        after_lyot_stop2 = np.zeros((isz_pup,isz_pup),dtype=complex)
        after_lyot_stop2[int(isz_pup/2-pupsize/2):int(isz_pup/2+pupsize/2),int(isz_pup/2-pupsize/2):int(isz_pup/2+pupsize/2)]=after_lyot_stop
        
        detector_img = shift(fft(shift(after_lyot_stop2)))[int(isz_pup/2-isz_foc/2):int(isz_pup/2+isz_foc/2),int(isz_pup/2-isz_foc/2):int(isz_pup/2+isz_foc/2)]
    elif coro == 'FQPM':
        fqpm_mask=np.ones((pupsize,pupsize))
        fqpm_mask[0:int(pupsize/2),0:int(pupsize/2)]=-1
        fqpm_mask[int(pupsize/2):pupsize,int(pupsize/2):pupsize]=-1
#        import os
#        MatrixDirectory=os.getcwd()+'/MatricesAndModel/'
#        SaveFits(fqpm_mask,['',0],MatrixDirectory,'fqpm',replace=True)
#        SaveFits(np.abs(input_wavefront),['',0],MatrixDirectory,'input_wavefront',replace=True)
#        SaveFits(np.abs(lyot_mask),['',0],MatrixDirectory,'lyot',replace=True)
        before_lyot_stop = shift(ifft(fft(shift(input_wavefront*PSFcentering))*fqpm_mask))
#        SaveFits(np.abs(before_lyot_stop),['',0],MatrixDirectory,'before_lyot',replace=True)

        if pupparf == True:
            return shift(ifft(fft(shift(before_lyot_stop*(1-lyot_mask)))*fqpm_mask))*np.conjugate(PSFcentering)
 
        after_lyot_stop=before_lyot_stop*lyot_mask
#        SaveFits(np.abs(after_lyot_stop),['',0],MatrixDirectory,'afterlyot',replace=True)
 
        detector_img=shift(fft(shift(after_lyot_stop*np.conjugate(PSFcentering))))[int(pupsize/2-isz_foc/2):int(pupsize/2+isz_foc/2),int(pupsize/2-isz_foc/2):int(pupsize/2+isz_foc/2)]
 #       SaveFits(np.abs(detector_img)**2,['',0],MatrixDirectory,'im',replace=True)
    else:
        print('coro should be APLC or FQPM')


    return detector_img
    
    
    
def invertDSCC(interact, cut ,goal='e', regul="truncation", visu=False):
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


def createvectorprobes(input_wavefront, wave, lyot_mask , Name_ALC , isz_foc, pushact, posprobes , cutsvd, coro, PSFcentering):
    pup_shape = input_wavefront.shape
    pupsize = pup_shape[0] #Size of pupil in pixel (384)
    numprobe = len(posprobes)
    deltapsik = np.zeros((numprobe , isz_foc , isz_foc),dtype=complex)
    probephase = np.zeros((numprobe , pupsize , pupsize))
    matrix = np.zeros((numprobe,2))
    Vecteurenvoi = np.zeros((isz_foc**2,2,numprobe))
    SVD = np.zeros((2,isz_foc,isz_foc))
    
    maskoffaxis = translationFFT(pupsize,30,30)
    OffAxisPSF = pupiltodetector(maskoffaxis*input_wavefront, wave, lyot_mask , Name_ALC , isz_foc,coro,PSFcentering)
    squaremaxPSF = np.amax(np.abs(OffAxisPSF))

    cutsvd = 0.3*squaremaxPSF*8/(400/37)
    
    pupilnoabb = pupiltodetector(input_wavefront , wave , lyot_mask , Name_ALC , isz_foc,coro,PSFcentering)

    k=0
    for i in posprobes:
        print(i)
        if coro == 'APLC':
            probephase[k] = pushact[i]
        elif coro == 'FQPM':
            probephase[k] = zeropad(pushact[i],pupsize)
        probephase[k] = 2*np.pi*(probephase[k])*1e-9/wave
        input_wavefront_k = input_wavefront*(1+1j*probephase[k]) #entrance pupil plane field (can be real, or complex with amplitude and phase)
        deltapsikbis = pupiltodetector(input_wavefront_k, wave,lyot_mask,Name_ALC,isz_foc,coro,PSFcentering)
        deltapsik[k] = (deltapsikbis-pupilnoabb)/squaremaxPSF
        k=k+1

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
    
    
    
def creatingCorrectionmatrix(input_wavefront, wave, lyot_mask , Name_ALC , isz_foc, pushact, mask, Whichact,coro,PSFcentering):
    pup_shape = input_wavefront.shape
    pupsize = pup_shape[0] #Size of pupil in pixel (384)
    maskoffaxis = translationFFT(pupsize,30,30)
    OffAxisPSF = pupiltodetector(maskoffaxis*input_wavefront, wave, lyot_mask , Name_ALC , isz_foc,coro,PSFcentering)
    squaremaxPSF = np.amax(np.abs(OffAxisPSF))
    
    pupilnoabb = pupiltodetector(input_wavefront , wave , lyot_mask , Name_ALC , isz_foc,coro,PSFcentering)
    
    Gmatrixbis=np.zeros((2*int(np.sum(mask)),len(Whichact)))

    k=0
    for i in Whichact:
        print(i)
        if coro == 'APLC':
            Psivector = pushact[i]
        elif coro == 'FQPM':
            Psivector = zeropad(pushact[i],pupsize)
        Psivector=2*np.pi*(Psivector)*1e-9/wave
        input_wavefront_k=input_wavefront*(1+1j*Psivector) #entrance pupil plane field (can be real, or complex with amplitude and phase)
        
        
        Gvectorbisbis=(pupiltodetector(input_wavefront_k , wave , lyot_mask , Name_ALC , isz_foc,coro,PSFcentering)- pupilnoabb)/squaremaxPSF
    
        
        Gmatrixbis[0:int(np.sum(mask)),k]=(np.real(Gvectorbisbis)[np.where(mask==1)]).flatten()
        Gmatrixbis[int(np.sum(mask)):,k]=(np.imag(Gvectorbisbis)[np.where(mask==1)]).flatten()
        k=k+1
    return Gmatrixbis



def translationFFT(pupsize,a,b):
    maskx=np.zeros((pupsize,pupsize))
    masky=np.zeros((pupsize,pupsize))
    for i in np.arange(pupsize):
        for j in np.arange(pupsize):
            maskx[i,j]=j*np.pi*2*a/pupsize-a*np.pi
            masky[i,j]=i*np.pi*2*b/pupsize-b*np.pi
    masktot=np.exp(-1j*maskx)*np.exp(-1j*masky)
    return masktot


def zeropad(tab,dim):
    newtab=np.zeros((dim,dim),dtype=tab.dtype)
    newtab[int(dim/2-tab.shape[0]/2):int(dim/2+tab.shape[0]/2),int(dim/2-tab.shape[1]/2):int(dim/2+tab.shape[1]/2)]=tab
    return newtab
    