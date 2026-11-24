import os
import numpy as np
import matplotlib.pyplot as plt
import glob

#import Performance_function as perf
import sphere_efc.SPHERE_EFC_Func as SPHERE
from astropy.io import fits
from scipy.optimize import fmin_powell as fmin_powell
import glob
from natsort import natsorted

import importlib

importlib.reload(SPHERE)

import matplotlib as mpl

mpl.rcParams['image.origin'] = 'lower'


def reprocess_experiment_IFS(param, output_dir, delta_wave = 0, refine = True):

    SPHERE.IFS_from_raw_to_cube(param, delta_wave = delta_wave, refine = refine, output_dir = output_dir)

    print('[WARNING] Pay attention to where the matrices are. Go to CreateMatrixfromModelEFConSPHERE.py first and run PWP with waves = full. Same, the dark hole mask should have the same number of spectral channels than the PWP matrices.')

    param["MatrixDirectory"] = param["MatrixDirectory"] + '/' + param["detector"] + '/'
    param["ImageDirectory"] = output_dir
    dir2 = output_dir + param["exp_name"]
    
    pattern = output_dir + param["exp_name"] + "*Probe_0001*.fits"

    print(pattern,'\n')

    iter_count = len(glob.glob(pattern))
    
    L = [print(f) for f in glob.glob(pattern)]


    print('iter_count =', iter_count)
    for nbiter in range(2, iter_count + 2):
        print('\nProcessing iteration ', nbiter)
        param['nbiter'] = nbiter
        coherent_signal, incoherent_signal, imagecorrection, _, _  = SPHERE.resultEFC(param)
        head_image = SPHERE.create_header_from_dict(param)    
            
        #Save data 
        fits.writeto(dir2 + 'iter' + str(nbiter-2) + 'CoherentSignal.fits', coherent_signal, header = head_image, overwrite = True)
        fits.writeto(dir2 + 'iter' + str(nbiter-2) + 'IncoherentSignal.fits', incoherent_signal, header = head_image, overwrite = True)
        fits.writeto(dir2 + 'iter' + str(nbiter-2) + 'TotalIntensity.fits', imagecorrection, header = head_image, overwrite = True)




if __name__ == "__main__":

    WORK_PATH1 = os.path.dirname(__file__) +'/../'

    
    whoiam = 'celia' # useful for the data path

    if whoiam == 'axel':
        #Put matrix directory (MatricesAndModel/) path below
        MatrixDirectory= '/Users/apoitier/Documents/Research/Softwares/sphere_efc/sphere_efc/MatricesAndModel/'
        #Directory where all the IFS raw images will be extracted. Think also about the backgrounds
        ImageDirectory= '/Users/apoitier/Documents/Research/Instruments/SPHERE/Data/Aout2026/2026-08-15/Experiments/Exp0010/'
        #Directory where to save all those images
        processed_directory = '/Users/apoitier/Documents/Research/Instruments/SPHERE/Data/Test/'

    elif whoiam == 'celia':
        #Put matrix directory (MatricesAndModel/) path below
        MatrixDirectory= '/Users/cdesgran/Documents/work/code/sphere_efc/sphere_efc/MatricesAndModel/'
        #Directory where all the IFS raw images will be extracted. Think also about the backgrounds
        ImageDirectory= '/Users/cdesgran/Documents/work/projects/instrus/SPHERE/SciOps/CDI/data/2026-08-15/Experiments/Exp0010/'
        #Directory where to save all those images
        processed_directory = '/Users/cdesgran/Documents/work/projects/instrus/SPHERE/SciOps/CDI/data/2026-08-15/Test/'



    if not os.path.exists(processed_directory):
            os.makedirs(processed_directory)

    #Which experiment to process in the following format: 'Experiment0007_'
    exp_name = 'Experiment0010_'

    onsky=0 #Set 0 for internal pup ; 1 for an on sky correction
    #Coronagraph that is used
    coro='APLC'
    #Dark hole size : param namemask in CreateMatrixfromModelEFConSPHERE.py
    dhsize=60
    correction_channel="equal_weight" #Either "longer_weight", "equal_weight", or a string of an integer
    corr_mode=500

    gain=0 # Keep 0 for CDI postprocessing!!

    #Algorithm for estimation. Should be either PWP or BTW
    estim_algorithm='PWP'

    #Number of probing actuator
    zone_to_correct='vertical' #vertical #horizontal #FDH

    #Type of probes used for PWP
    probe_type='individual_act' #'sinc' #'individual_act'

    #SizeProbes : can be 296, 400 or 500 (in nm)
    size_probes=200

    #Do you want to center your image at each iteration. Set 1 for yes, 0 for no.
    centeringateachiter=0
    #Do you want to rescale the coherent intensity to match the total intensity in the DH?
    rescaling=0

    # Path 
    #WORK_PATH0='/Users/apoitier/Documents/Research/CDI/SPHERE/cdi-jun04-2024/'

    centerx, centery = fits.getdata(ImageDirectory + exp_name + 'centerxy.fits')
    which_nd = 'ND_3.5'

    ModelDirectory = WORK_PATH1 + 'Model/'

    if onsky == 0:
        lightsource_estim = 'InternalPupil_'
    else:
        lightsource_estim = 'VLTPupil_'

    lightsource_estim = lightsource_estim + coro + '_'


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

    param = {
    "ImageDirectory": ImageDirectory,
    "MatrixDirectory": MatrixDirectory,
    "RootDirectory" : WORK_PATH1,

    "exp_name": exp_name,
    "dhsize": dhsize,
    "zone_to_correct":zone_to_correct,
    "centerx": centerx,
    "centery": centery,
    "which_nd": which_nd,
    "onsky": onsky,
    "size_probes": size_probes,
    "amplitudeEFCMatrix": 8, #required for the code to run but useless
    "corr_mode": corr_mode, #required for the code to run but useless
    "centeringateachiter": centeringateachiter,
    "dimimages": 200,
    "gain": gain,
    "rescaling": rescaling,
    "estim_algorithm": estim_algorithm,
    "probe_type": probe_type,

    "coro": coro,
    "live_matrix_measurement": False,
    "wave": 1.667e-6,
    "ModelDirectory": ModelDirectory,
    "lightsource_estim": lightsource_estim,
    "lightsource_corr": lightsource_estim, #required for the code to run but useless
    "posprobes": posprobes,
    "correction_channel": correction_channel,
    "detector":"IFS_OBS_H",
    "use_bkg": False, # CDe ! s

    }

    reprocess_experiment_IFS(param, processed_directory, refine=False) 