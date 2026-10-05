#Tests with Zahed, with EssaiBash.sh
#Use with MainSphereEFC
#LyotStop was modified!
#Version 2026/06/29 17h30 UT
## Parameters and function


import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import Definitions_for_matrices as def_mat

# directory where are all the different matrices (CLMatrixOptimiser.HO_IM.fits , etc..)
MatrixDirectory = os.getcwd()+'/MatricesAndModel/'
# directory where are all the different model planes (Apod, Lyot, etc..)
ModelDirectory = os.getcwd()+'/Model/'

coro = 'APLC'
#coro = 'FQPM'
dimimages = 200
detector = 'IFS_OBS_H' #'IFS_OBS_H' or 'IFS_OBS_YJ' or 'IRDIS'

MatrixDirectory = MatrixDirectory + detector + '/'
if os.path.isdir(MatrixDirectory) is False:
        #Create the directory
        os.mkdir(MatrixDirectory)

if detector == 'IFS_OBS_YJ' or detector == 'IFS_OBS_H':
    #waves = fits.getdata(MatrixDirectory + detector + '_wavelength_full.fits')
    waves = fits.getdata(MatrixDirectory + detector + '_wavelength.fits')
    waves = waves * 1e-9
elif detector == 'IRDIS_H3':
    waves = [1.667e-6]
onsky = 0 #1 if on sky correction

zone_to_correct = 'vertical' #vertical #horizontal #'FDH'
createPW = False
probe_type = 'individual_act' #'sinc' #'individual_act'

createwhich = False
createjacobian = False

#name of the mask that can be saved with createmask and then used in createEFCmatrix
namemask ='51'
createmask = True

mask_shape = 'circle' #circle or square
pix_limit_in_ld = [8, 35, -35, 35] # Only used if mask_shape = 'square'
circ_rad_in_ld = [3, 16] # Only used if mask_shape = 'circle', in lambda/D
circ_side = 'Bottom' # Only used if mask_shape = 'circle'. Can be Full, Top, Bottom, Left or Right
circ_offset_in_ld = 2 # Only used if mask_shape = 'circle', in lambda/D
circ_angle = 0 # Only used if mask_shape = 'circle', in degree


nbmodes = 500
correction_channel="equal_weight" #Either "longer_weight", "equal_weight", or an integer
createEFCmatrix = True



#Wrapper: extract objects in the different pupil and focal planes, depending on the coronagraph and wavelength
mask384, Pup384, ALC, Lyot384 = def_mat.Upload_CoroConfig(ModelDirectory, coro)

#Perfect pupil for FQPM (remove numeric noise)
#AXEL : I don't think this is required:
# if coro == 'FQPM':
#     if onsky == 0:
#         mask384 = def_mat.pupiltodetector(mask384, wave, Lyot384, '', dimimages, coro, pupparf=True)
        
#Cube of actuator positions in pupil    
raw_pushact = fits.getdata(ModelDirectory+'PushActInPup384SecondWay.fits')

if onsky==0:
    input_wavefront = mask384
    lightsource = 'InternalPupil_'
else:
    input_wavefront = mask384*Pup384
    lightsource = 'VLTPupil_'

lightsource = lightsource + coro + '_'

#Amplitude in x nm/37 for the PW pokes such that pushact amplitude is equal to x nm
amplitudePW = 296/37
#Amplitude in x nm/37 for the pokes to create the jacobian matrix such that pushact amplitude is equal to x nm (usually 296nm here)
amplitudeEFCMatrix = 8

if detector == 'IFS_OBS_H' or detector == 'IFS_OBS_YJ':
    resolinarcsec_pix = 7.4e-3
elif detector == 'IRDIS_H3' or detector == 'IRDIS_H2':
    resolinarcsec_pix = 12.25e-3
else:
    print('Error resolinarcsec_pix undefined !!!')

#### Pour estimation

if createPW == True:
    print('...Creating VectorProbes...')
    print('Probe type: ' + probe_type)
    # Choose probes positions
    if coro == 'APLC':
        if zone_to_correct == 'vertical':
            posprobes = [678 , 679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
        elif zone_to_correct == 'horizontal':
            posprobes = [893 , 934]
        elif zone_to_correct == 'FDH':
            posprobes = [678 , 679, 720]
    
        
    elif coro == 'FQPM':
        if zone_to_correct == 'vertical':
            posprobes = [678 , 679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
        elif zone_to_correct == 'horizontal':
            posprobes = [1089 , 1125] #FQPM
        elif zone_to_correct == 'FDH':
            raise ValueError('This setting is not available for FQPM yet')
        
    #Choose the truncation above where the pixels won't be taken into account for estimation (not used currently here)
    cutestimation = 1e20#0.3*squaremaxPSF*8/amplitudePW

    PWP_matrix = []

    for wave in waves :
        print('wavelength: ', format(wave, '.2e'))
        PWP_one_wvl,SVD,int_probes,probevoltage = def_mat.createvectorprobes(input_wavefront,
                                                                            wave,
                                                                            Lyot384 ,
                                                                            ALC ,
                                                                            dimimages ,
                                                                            raw_pushact ,
                                                                            amplitudePW,
                                                                            posprobes ,
                                                                            cutestimation,
                                                                            coro,
                                                                            probe_type,
                                                                            resolinarcsec_pix = resolinarcsec_pix)
        
        PWP_matrix.append(PWP_one_wvl)

    PWP_matrix = np.array(PWP_matrix)

    filename = probe_type + '_' + zone_to_correct + '_' + str(int(amplitudePW*37)) + 'nm' + '_'
    ##
    fits.writeto(MatrixDirectory + lightsource + filename + 'CorrectedZone.fits', SVD[1], overwrite = True)
    ##
    fits.writeto(MatrixDirectory + lightsource + filename + 'PWP_matrix.fits', PWP_matrix, overwrite = True)
    ##
    fits.writeto(MatrixDirectory + lightsource + filename + 'Intensity_probe.fits', int_probes, overwrite = True)
    ##
    fits.writeto(MatrixDirectory + lightsource + filename + 'Voltage_probe.fits', probevoltage, overwrite = True)


#### Pour correction 

if createwhich==True:
    print('...Creating DH and Gmatrix...')
    WhichInPupil = def_mat.creatingWhichinPupil(Lyot384, raw_pushact, 0.5)
    print('Number of actuators in visible through the Lyot: ', len(WhichInPupil))
    fits.writeto(MatrixDirectory + lightsource + 'WhichInPupil0_5.fits', WhichInPupil, overwrite = True)


##
if detector == 'IFS_OBS_H':
    waves = waves[::2]

if createjacobian==True:
    print('...Creating Jacobian...')
    pushact = amplitudeEFCMatrix * raw_pushact
    WhichInPupil = fits.getdata(MatrixDirectory + lightsource + 'WhichInPupil0_5.fits')
    #Creating Matrix
    Gmatrix = []
    for wave in waves :
        print('wavelength: ', format(wave, '.2e'))
        Gmatrix_one_wvl = def_mat.creatingCorrectionmatrix(input_wavefront,
                                                 wave,
                                                 Lyot384 ,
                                                 ALC ,
                                                 dimimages ,
                                                 pushact ,
                                                 WhichInPupil,
                                                 coro,
                                                 resolinarcsec_pix = resolinarcsec_pix)
        Gmatrix.append(Gmatrix_one_wvl)

    Gmatrix = np.array(Gmatrix)
    #Saving matrix
    fits.writeto(MatrixDirectory + lightsource + 'Jacobian.fits', Gmatrix, overwrite = True)


#Choose the four corners of your dark hole (in pixels)
if createmask == True:
    print('...Creating mask DH...')
    maskDH = def_mat.create_mask_in_ld(dimimages, resolinarcsec_pix, np.array(waves), mask_shape, pix_limit_in_ld, circ_rad_in_ld, circ_side, circ_offset_in_ld, circ_angle)
    fits.writeto(MatrixDirectory + '../mask_DH' + namemask + '.fits', maskDH, overwrite = True)
    plt.imshow(maskDH[0]) #Afficher où le DH apparaît sur l'image au final


#### Uncomment below to create and save the interaction matrix
if createEFCmatrix == True:
    print('...Creating EFC matrix...')
    def_mat.create_interaction_matrix(MatrixDirectory, lightsource, len(waves), namemask, nbmodes, correction_channel)




















