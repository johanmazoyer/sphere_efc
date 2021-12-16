#Tests with Zahed, with EssaiBash.sh
#Use with MainSphereEFC
#LyotStop was modified!
#Version 6/12/2019
## Parameters and function


import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import Definitions_for_matrices as def_mat


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


# directory where are all the different matrices (CLMatrixOptimiser.HO_IM.fits , etc..)
MatrixDirectory=os.getcwd()+'/MatricesAndModel/'
# directory where are all the different model planes (Apod, Lyot, etc..)
ModelDirectory=os.getcwd()+'/Model/'


wave=1.667e-6
mask384=fits.getdata(ModelDirectory+'apod-4.0lovD_384-192.fits')
Pup384=fits.getdata(ModelDirectory+'generated_VLT_pup_384-192.fits')
ALC='ALC2'
Lyot384=fits.getdata(ModelDirectory+'sphere_stop_ST_ALC2.fits')
raw_pushact = fits.getdata(ModelDirectory+'PushActInPup384SecondWay.fits')


#maskoffaxis=def_mat.translationFFT(30,30)
dimimages=400



onsky=1 #1 if on sky correction
createPW=False
createmask=False
createwhich=False
createjacobian=True
createEFCmatrix=True


if onsky==0:
    input_wavefront = mask384
    lightsource='InternalPupil_'
else:
    input_wavefront = mask384*Pup384
    lightsource='VLTPupil_'

#OffAxisPSF=def_mat.pupiltodetector(maskoffaxis*mask384*Pup384, wave,Lyot384,ALC,dimimages)

#Amplitude in x nm/37 for the PW pokes such that pushact amplitude is equal to x nm
amplitudePW=400/37
#Amplitude in x nm/37 for the pokes to create the jacobian matrix such that pushact amplitude is equal to x nm (usually 296nm here)
amplitudeEFCMatrix=8

#squaremaxPSF=np.amax(np.abs(OffAxisPSF))


#### Pour estimation

if createPW==True:
    print('...Creating VectorProbes...')
    pushact=amplitudePW*raw_pushact
    # Choose probes positions
    posprobes = [678,679,680,681]#0.04cutestimation
    posprobes=[678,679,680] #0.1cutestimation
    posprobes=[678,679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
    #Choose the truncation above where the pixels won't be taken into account for estimation
    cutestimation = 0#0.3*squaremaxPSF*8/amplitudePW

    vectoressai,SVD = def_mat.createvectorprobes(input_wavefront,
                                                 wave,
                                                 Lyot384 ,
                                                 ALC ,
                                                 dimimages ,
                                                 pushact ,
                                                 posprobes ,
                                                 cutestimation )
    ##
    choosepixvisu = [-55,55,-55,55]
    maskDH = def_mat.creatingMaskDH(dimimages, 'square', choosepixDH = choosepixvisu)

    plt.imshow(SVD[1]*maskDH)
    plt.show()
    ##
    def_mat.SaveFits(SVD[1],['',0],MatrixDirectory,lightsource+'CorrectedZone',replace=True)
    ##
    def_mat.SaveFits(vectoressai,['',0],MatrixDirectory,lightsource+'VecteurEstimation_'+str(len(posprobes))+'probes'+str(int(amplitudePW*37))+'nm',replace=True)

#### Pour correction 

if createwhich==True:
    print('...Creating DH and Gmatrix...')
    WhichInPupil = def_mat.creatingWhichinPupil(Lyot384, raw_pushact, 0.5)
    print(len(WhichInPupil))
    def_mat.SaveFits(WhichInPupil,['',0],MatrixDirectory,'WhichInPupil0_5',replace=True)

#Choose the four corners of your dark hole (in pixels)
if createmask==True:
    print('...Creating mask DH...')
    choosepix = [-55,55,10,55] #DH3
    choosepix = [-55,55,-55,-10] #DH1
    maskDH = def_mat.creatingMaskDH(dimimages, 'square', choosepixDH = choosepix)
    namemask='1'
    def_mat.SaveFits(maskDH,['',0],MatrixDirectory,'mask_DH'+namemask,replace=True)
    plt.imshow((maskDH)) #Afficher où le DH apparaît sur l'image au final
    plt.pause(0.1)
##
if createjacobian==True:
    print('...Creating Jacobian...')
    namemask='1'
    pushact=amplitudeEFCMatrix*fits.getdata(ModelDirectory+'PushActInPup384SecondWay.fits')
    maskDH=fits.getdata(MatrixDirectory+'mask_DH'+namemask+'.fits')
    WhichInPupil=fits.getdata(MatrixDirectory+'WhichInPupil0_5.fits')
    #Creating Matrix
    Gmatrix = def_mat.creatingCorrectionmatrix(input_wavefront,
                                                 wave,
                                                 Lyot384 ,
                                                 ALC ,
                                                 dimimages ,
                                                 pushact ,
                                                 maskDH,
                                                 WhichInPupil)
    #Saving matrix
    def_mat.SaveFits(Gmatrix,['',0],ModelDirectory,lightsource+'Gmatrix_DH'+namemask,replace=True)

#### Uncomment below to create and save the interaction matrix
if createEFCmatrix==True:
    print('...Creating EFC matrix...')
    namemask='1'
    Gmatrix = fits.getdata(ModelDirectory+lightsource+'Gmatrix_DH'+namemask+'.fits')
    #Set how many modes you want to use to correct
    nbmodes = 600
    invertGDH = def_mat.invertDSCC(Gmatrix,nbmodes,goal='c',visu=True)[1]
    corr_mode='1'
    def_mat.SaveFits(invertGDH,['',0],MatrixDirectory,lightsource+'Interactionmatrix_DH'+namemask+'_SVD'+corr_mode,replace=True)
