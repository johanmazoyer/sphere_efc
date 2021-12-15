#Tests with Zahed, with EssaiBash.sh
#Use with MainSphereEFC
#LyotStop was modified!
#Version 6/12/2019
## Parameters and function


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from astropy.io import fits
from poppy import matrixDFT




def roundpupil(nbpix,prad1):
    xx, yy = np.meshgrid(np.arange(nbpix)-(nbpix)/2, np.arange(nbpix)-(nbpix)/2)
    rr     = np.hypot(yy, xx)
    pupilnormal=np.zeros((nbpix,nbpix))
    pupilnormal[rr<=prad1]=1.
    return pupilnormal



def LoadImageFits(docs_dir):
    openfits=fits.open(docs_dir)
    image=openfits[0].data
    return image

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


def pupiltodetector(input_wavefront,lyot_mask,occ_rad_ld):
    mft_sampling=100 #[pixels/(l/D)] sampling factor of the computed focal plane field with the MFT
    
    pup_shape = input_wavefront.shape
    lbd = 1.  #wavelength of the simulated image
    lbd_ref = lbd #wavelength used as reference to define the sampling.
    
    lbd_coeff = lbd/lbd_ref
    occulter_fov = 5  #focal plane field of view for computing the field.
    occ_rad_pix = occ_rad_ld*mft_sampling    #radius of the occulting mask in pixels.
    
    npix         = (int(np.round(occulter_fov * mft_sampling)) // 2 ) *2 # force even dimension
    
    occulter_fov = float(npix) / float(mft_sampling)
    
    focal_plane = matrixDFT.matrix_dft(input_wavefront, occulter_fov, npix, centering='FFTSTYLE')
    
    
    occulter_area = roundpupil(npix, occ_rad_pix)
    lyot_plane_rejected = matrixDFT.matrix_dft(focal_plane*occulter_area,occulter_fov, pup_shape, inverse=True,centering='FFTSTYLE')
    focal_plane = focal_plane*(1.-.8*occulter_area)
    
    before_lyot_stop = (input_wavefront - lyot_plane_rejected)
    
    after_lyot_stop = before_lyot_stop * lyot_mask
    
    
    after_lyot_stop2=np.zeros((isz,isz),dtype=complex)
    after_lyot_stop2[int(isz/2-pupsize/2):int(isz/2+pupsize/2),int(isz/2-pupsize/2):int(isz/2+pupsize/2)]=after_lyot_stop
        
    detector_img=shift(fft(shift(after_lyot_stop2)))[int(isz/2-dimimages/2):int(isz/2+dimimages/2),int(isz/2-dimimages/2):int(isz/2+dimimages/2)]
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
    pseudoinverse=np.dot(np.dot(np.transpose(V),InvS),np.transpose(U))
    return [np.diag(InvS),pseudoinverse]


def createvectorprobes(posprobes,cutsvd):
    numprobe=len(posprobes)
    deltapsik=np.zeros((numprobe,dimimages,dimimages),dtype=complex)
    probephase=np.zeros((numprobe,pupsize,pupsize))
    matrix=np.zeros((numprobe,2))
    Vecteurenvoi=np.zeros((dimimages**2,2,numprobe))
    SVD=np.zeros((2,dimimages,dimimages))
    lyot_mask=Lyot384 #Lyot mask
    occ_rad_ld=radFPMinld
    if onsky==0:
        pupilnoabb=pupiltodetector(mask384,lyot_mask,occ_rad_ld)
    else:
        pupilnoabb=pupiltodetector(mask384*Pup384,lyot_mask,occ_rad_ld)
    k=0
    for i in posprobes:
        probephase[k]=pushact[i]
        probephase[k]=2*np.pi*(probephase[k])*1e-9/wave
        if onsky==0:
            input_wavefront=mask384*(1+1j*probephase[k]) #entrance pupil plane field (can be real, or complex with amplitude and phase)
        else:
            input_wavefront=mask384*Pup384*(1+1j*probephase[k]) #entrance pupil plane field (can be real, or complex with amplitude and phase)
        
        deltapsikbis=pupiltodetector(input_wavefront,lyot_mask,occ_rad_ld)
        
        deltapsik[k]=(deltapsikbis-pupilnoabb)/squaremaxPSF
        k=k+1

    l=0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            matrix[:,0]=real(deltapsik[:,i,j])
            matrix[:,1]=im(deltapsik[:,i,j])
            try:
                SVD[:,i,j]=invertDSCC(matrix,cutsvd,visu=False)[0]
                Vecteurenvoi[l]=invertDSCC(matrix,cutsvd,visu=False)[1]
            except:
                print('Careful: Error! for l='+str(l))
                SVD[:,i,j]=np.zeros(2)
                Vecteurenvoi[l]=np.zeros((2,numprobe))
            l=l+1  
    return [Vecteurenvoi,SVD]
    

def creatingWhichinPupil(cutinpupil):
    pupille=Lyot384
    WhichInPupil = []
    for i in np.arange(int(1377)):
        Psivector=-pushact[i]
        cut=cutinpupil*np.sum(-pushact[182])
        if(np.sum(Psivector*pupille)>cut):
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
    
    
    
def creatingCorrectionmatrix(mask,Whichact):
    Gmatrixbis=np.zeros((2*int(np.sum(mask)),len(Whichact)))
    lyot_mask=Lyot384 #Lyot mask
    occ_rad_ld=radFPMinld
    if onsky==0:
        pupilnoabb=pupiltodetector(mask384,lyot_mask,occ_rad_ld)
    else:
        pupilnoabb=pupiltodetector(mask384*Pup384,lyot_mask,occ_rad_ld)
    k=0
    for i in Whichact:
        print(i)
        Psivector=pushact[i]
        Psivector=2*np.pi*(Psivector)*1e-9/wave
        if onsky==0:
            input_wavefront=mask384*(1+1j*Psivector) #entrance pupil plane field (can be real, or complex with amplitude and phase)
        else:
            input_wavefront=mask384*Pup384*(1+1j*Psivector) #entrance pupil plane field (can be real, or complex with amplitude and phase)
        
        
        Gvectorbisbis=(pupiltodetector(input_wavefront,lyot_mask,occ_rad_ld)-pupilnoabb)/squaremaxPSF
    
        
        Gmatrixbis[0:int(np.sum(mask)),k]=(real(Gvectorbisbis)[np.where(mask==1)]).flatten()
        Gmatrixbis[int(np.sum(mask)):,k]=(im(Gvectorbisbis)[np.where(mask==1)]).flatten()
        k=k+1
    return Gmatrixbis



def translationFFT(a,b):
    maskx=np.zeros((pupsize,pupsize))
    masky=np.zeros((pupsize,pupsize))
    for i in np.arange(pupsize):
        for j in np.arange(pupsize):
            maskx[i,j]=j*np.pi*2*a/pupsize-a*np.pi
            masky[i,j]=i*np.pi*2*b/pupsize-b*np.pi
    masktot=np.exp(-1j*maskx)*np.exp(-1j*masky)
    return masktot
    
    

#Raccourcis FFT
fft = np.fft.fft2
ifft = np.fft.ifft2
shift = np.fft.fftshift
ishift=np.fft.ifftshift

abs=np.abs
im=np.imag
real=np.real
mean=np.mean
dot=np.dot
amax=np.amax

#Raccourcis conversions angles
dtor    = np.pi / 180.0 # degree to radian conversion factor
rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor
pupsize=384  #Pupsize in pixel
pupsizeinmeter=8 #Pupsizeinmeter


wave=1.667e-6
resolinarcsec_pix=12.25e-3  #arcsec/pix
resolinrad_pix=resolinarcsec_pix/3600*np.pi/180  #rad/pix
resolinpix_rad=1/resolinrad_pix     #pix/rad

ld_rad=wave/pupsizeinmeter #lambda/D en radian

ld_p=resolinpix_rad*ld_rad  #lambda/D en pixel
print('pixel per resolution element:', ld_p)
#ld_p=3.5


isz=int(pupsize*ld_p)#-1 # Nombre de pixels dans le plan pupille pour atteindre la résolution ld_p voulue
#print(isz)

ld_mas=ld_rad*180/np.pi*60*60*1e3  #lambda/D en milliarcsec
#radFPMinld=120/ld_mas  #Taille masque corono en lambda/D ALC3
radFPMinld=92.5/ld_mas #Taille masque corono en lambda/D ALC2
print('Size Lyot mask in l/d:', radFPMinld)

radFPMinpix=int(radFPMinld*ld_p)   #Taille masque corono en pixel

# directory where are all the different matrices (CLMatrixOptimiser.HO_IM.fits , etc..)
MatrixDirectory=os.getcwd()+'/MatricesAndModel/'
# directory where are all the different model planes (Apod, Lyot, etc..)
ModelDirectory=os.getcwd()+'/Model/'

mask384=LoadImageFits(ModelDirectory+'apod-4.0lovD_384-192.fits')
Pup384=LoadImageFits(ModelDirectory+'generated_VLT_pup_384-192.fits')
Lyot384=LoadImageFits(ModelDirectory+'sphere_stop_ST_ALC2.fits')



maskoffaxis=translationFFT(30,30)
dimimages=400



onsky=1 #1 if on sky correction
createPW=False
createmask=False
createwhich=False
createjacobian=False
createEFCmatrix=True


if onsky==0:
    OffAxisPSF=pupiltodetector(maskoffaxis*mask384,Lyot384,radFPMinld)
    lightsource='InternalPupil_'
else:
    OffAxisPSF=pupiltodetector(maskoffaxis*mask384*Pup384,Lyot384,radFPMinld)
    lightsource='VLTPupil_'

#Amplitude in x nm/37 for the PW pokes such that pushact amplitude is equal to x nm
amplitudePW=400/37
#Amplitude in x nm/37 for the pokes to create the jacobian matrix such that pushact amplitude is equal to x nm (usually 296nm here)
amplitudeEFCMatrix=8

squaremaxPSF=np.amax(np.abs(OffAxisPSF))





#### Pour estimation

if createPW==True:
    print('...Creating VectorProbes...')
    pushact=amplitudePW*LoadImageFits(ModelDirectory+'PushActInPup384SecondWay.fits')
    # Choose probes positions
    posprobes = [678,679,680,681]#0.04cutestimation
    posprobes=[678,679,680] #0.1cutestimation
    posprobes=[678,679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
    #Choose the truncation above where the pixels won't be taken into account for estimation
    cutestimation = 0.3*squaremaxPSF*8/amplitudePW

    vectoressai,SVD = createvectorprobes(posprobes,cutestimation)
    ##
    choosepixvisu = [-55,55,-55,55]
    maskDH = creatingMaskDH(dimimages, 'square', choosepixDH = choosepixvisu)

    plt.imshow(SVD[1]*maskDH)
    plt.show()
    ##
    SaveFits(SVD[1],['',0],MatrixDirectory,lightsource+'CorrectedZone',replace=True)
    ##
    SaveFits(vectoressai,['',0],MatrixDirectory,lightsource+'VecteurEstimation_'+str(len(posprobes))+'probes'+str(int(amplitudePW*37))+'nm',replace=True)

#### Pour correction 

if createwhich==True:
    print('...Creating DH and Gmatrix...')
    WhichInPupil = creatingWhichinPupil(0.5)
    print(len(WhichInPupil))
    SaveFits(WhichInPupil,['',0],MatrixDirectory,'WhichInPupil0_5',replace=True)

#Choose the four corners of your dark hole (in pixels)
if createmask==True:
    choosepix = [-55,55,10,55] #DH3
    choosepix = [-55,55,-55,-10] #DH1
    maskDH = creatingMaskDH(dimimages, 'square', choosepixDH = choosepix)
    namemask='1'
    SaveFits(maskDH,['',0],MatrixDirectory,'mask_DH'+namemask,replace=True)
    plt.imshow((maskDH)) #Afficher où le DH apparaît sur l'image au final
    plt.pause(0.1)
##
if createjacobian==True:
    namemask='1'
    pushact=amplitudeEFCMatrix*LoadImageFits(ModelDirectory+'PushActInPup384SecondWay.fits')
    maskDH=LoadImageFits(MatrixDirectory+'mask_DH'+namemask+'.fits')
    WhichInPupil=LoadImageFits(MatrixDirectory+'WhichInPupil0_5.fits')
    #Creating Matrix
    Gmatrix = creatingCorrectionmatrix(maskDH,WhichInPupil)
    #Saving matrix
    SaveFits(Gmatrix,['',0],ModelDirectory,lightsource+'Gmatrix_DH'+namemask,replace=True)

#### Uncomment below to create and save the interaction matrix
if createEFCmatrix==True:
    namemask='1'
    Gmatrix = LoadImageFits(ModelDirectory+lightsource+'Gmatrix_DH'+namemask+'.fits')
    #Set how many modes you want to use to correct
    nbmodes = 600
    invertGDH = invertDSCC(Gmatrix,nbmodes,goal='c',visu=True)[1]
    corr_mode='1'
    SaveFits(invertGDH,['',0],MatrixDirectory,lightsource+'Interactionmatrix_DH'+namemask+'_SVD'+corr_mode,replace=True)
