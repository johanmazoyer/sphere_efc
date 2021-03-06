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

#coro = 'APLC'
coro = 'FQPM'
dimimages=200
wave=1.667e-6
onsky=0 #1 if on sky correction

zone_to_correct = 'horizontal' #vertical
createPW=False

createwhich=False
createjacobian=False

#name of the mask that can be saved with createmask and then used in createEFCmatrix
namemask='2'
maskDH = def_mat.creatingMaskDH(dimimages, 'circle', circ_rad=[10,55], circ_side='Top', circ_offset=8)
createmask=False

nbmodes = 600
corr_mode='1'
createEFCmatrix=True




mask384, Pup384, ALC, Lyot384 = def_mat.Upload_CoroConfig(ModelDirectory, coro, wave)

#Perfect pupil for FQPM (remove numeric noise)
if coro == 'FQPM':
    if onsky == 0:
        mask384=def_mat.pupiltodetector(mask384,wave,Lyot384,'',dimimages,coro,pupparf=True)
    
raw_pushact = fits.getdata(ModelDirectory+'PushActInPup384SecondWay.fits')

if onsky==0:
    input_wavefront = mask384
    lightsource='InternalPupil_'
else:
    input_wavefront = mask384*Pup384
    lightsource='VLTPupil_'

lightsource = lightsource+coro+'_'

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
    if zone_to_correct == 'vertical':
        posprobes=[678,679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
    elif zone_to_correct == 'horizontal':
        posprobes=[893,934]
    #Choose the truncation above where the pixels won't be taken into account for estimation
    cutestimation = 0#0.3*squaremaxPSF*8/amplitudePW

    vectoressai,SVD = def_mat.createvectorprobes(input_wavefront,
                                                 wave,
                                                 Lyot384 ,
                                                 ALC ,
                                                 dimimages ,
                                                 pushact ,
                                                 posprobes ,
                                                 cutestimation,
                                                 coro)
    ##
    choosepixvisu = [-55,55,-55,55]
    maskvisu = def_mat.creatingMaskDH(dimimages, 'square', choosepixDH = choosepixvisu)

    plt.imshow(SVD[1]*maskvisu)
    plt.show()
    ##
    def_mat.SaveFits(SVD[1],['',0],MatrixDirectory,lightsource+zone_to_correct+'CorrectedZone',replace=True)
    ##
    def_mat.SaveFits(vectoressai,['',0],MatrixDirectory,lightsource+'VecteurEstimation_'+zone_to_correct+str(int(amplitudePW*37))+'nm',replace=True)

#### Pour correction 

if createwhich==True:
    print('...Creating DH and Gmatrix...')
    WhichInPupil = def_mat.creatingWhichinPupil(Lyot384, raw_pushact, 0.5)
    print('Number of actuators in visible through the Lyot: ',len(WhichInPupil))
    def_mat.SaveFits(WhichInPupil,['',0],MatrixDirectory,lightsource+'WhichInPupil0_5',replace=True)


##
if createjacobian==True:
    print('...Creating Jacobian...')
    pushact=amplitudeEFCMatrix*fits.getdata(ModelDirectory+'PushActInPup384SecondWay.fits')
    WhichInPupil=fits.getdata(MatrixDirectory+lightsource+'WhichInPupil0_5.fits')
    #Creating Matrix
    Gmatrix = def_mat.creatingCorrectionmatrix(input_wavefront,
                                                 wave,
                                                 Lyot384 ,
                                                 ALC ,
                                                 dimimages ,
                                                 pushact ,
                                                 WhichInPupil,
                                                 coro)

    #Saving matrix
    def_mat.SaveFits(Gmatrix,['',0],ModelDirectory,lightsource+'Jacobian',replace=True)


#Choose the four corners of your dark hole (in pixels)
if createmask==True:
    print('...Creating mask DH...')
    #choosepix = [-55,55,10,55] #DH3
    #choosepix = [-55,55,-55,-10] #DH1
    #maskDH = def_mat.creatingMaskDH(dimimages, 'square', choosepixDH = choosepix)
    def_mat.SaveFits(maskDH,['',0],MatrixDirectory,'mask_DH'+namemask,replace=True)
    plt.imshow((maskDH)) #Afficher o?? le DH appara??t sur l'image au final
    plt.pause(0.1)


#### Uncomment below to create and save the interaction matrix
if createEFCmatrix==True:
    print('...Creating EFC matrix...')
    maskDH=fits.getdata(MatrixDirectory+'mask_DH'+namemask+'.fits')
    Gmatrix = fits.getdata(ModelDirectory+lightsource+'Jacobian.fits')
    masked_Gmatrix = def_mat.get_masked_jacobian(Gmatrix, maskDH)
    #Set how many modes you want to use to correct
    invertGDH = def_mat.invertDSCC(masked_Gmatrix,nbmodes,goal='c',regul='tikhonov',visu=True)[1]
    def_mat.SaveFits(invertGDH,['',0],MatrixDirectory,lightsource+'Interactionmatrix_DH'+namemask+'_SVD'+corr_mode,replace=True)
























