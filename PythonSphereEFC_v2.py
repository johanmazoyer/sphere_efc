#Python EFC code
#Version 2021/10/14

import os
import warnings
import SPHERE_EFC_Func as func
warnings.filterwarnings("ignore")

# Retrieve the shell variables
# RootDirectory = os.environ['WORK_PATH0']#'/vltuser/sphere/jmilli/test_EFC_20190830/PackageEFConSPHERE/'
# ImageDirectory = os.environ['WORK_PATH']#RootDirectory+'SlopesAndImages/'
# MatrixDirectory = os.environ['MATRIX_PATH']#RootDirectory+'MatricesAndModel/'
RootDirectory = '/vltuser/sphere/zwahhaj/efc/'
ImageDirectory = RootDirectory+'SlopesAndImages/'
MatrixDirectory = RootDirectory+'MatricesAndModel/'
nbiter = int(os.environ['nbiter']) #nb of the iteration (first iteration is 1)
exp_name = os.environ['EXP_NAME'] #Rootname of the Experiment
dhsize = os.environ['DHsize']#Dark hole shape and size choice
corr_mode = os.environ['corr_mode']#Correction mode (more or less agressive)
zone_to_correct = os.environ['zone_to_correct']
x0_up = int(os.environ['X0UP'])#x position in python of the upper PSF echo
y0_up = int(os.environ['Y0UP'])#y position in python of the upper PSF echo
x1_up = int(os.environ['X1UP'])#x position in python of the bottom PSF echo
y1_up = int(os.environ['Y1UP'])#y position in python of the bottom PSF echo
which_nd = os.environ['WHICH_ND']# which ND is used to record the Off-axis PSF
onsky = int(os.environ['ONSKY'])#If 1: On sky measurements ; If 0: Calibration source measurements
Assuming_VLT_PUP_for_corr = int(os.environ['Assuming_VLT_PUP_for_corr'])
size_probes = int(os.environ['size_probes'])# Set the size of the probes for the estimation
centeringateachiter = int(os.environ['centeringateachiter'])#If 1: Do the recentering at each iteration ; If 0: Do not do it
coro = os.environ['coro']
gain = float(os.environ['gain'])
slope_ini = os.environ['SLOPE_INI']
rescaling = int(os.environ['rescaling'])
estim_algorithm = os.environ['ESTIM_ALGORITHM']

print('Your working path is {0:s} and you are doing iteration number {1:d} of {2:s}'.format(RootDirectory,nbiter,exp_name), flush=True)


param = {
  "RootDirectory": RootDirectory,
  "ImageDirectory": ImageDirectory,
  "MatrixDirectory": MatrixDirectory,
  "nbiter": nbiter,
  "exp_name": exp_name,
  "dhsize": dhsize,
  "corr_mode": corr_mode,
  "zone_to_correct":zone_to_correct,
  "x0_up": x0_up,
  "y0_up": y0_up,
  "x1_up": x1_up,
  "y1_up": y1_up,
  "which_nd": which_nd,
  "onsky": onsky,
  "Assuming_VLT_PUP_for_corr": Assuming_VLT_PUP_for_corr,
  "size_probes": size_probes,
  "centeringateachiter": centeringateachiter,
  "amplitudeEFCMatrix": 8, #Amplitude in x nm/37 for the pokes to create the jacobian matrix such that pushact amplitude is equal to x nm (usually 296nm here)
  "dimimages": 200,
  "gain": gain,
  "slope_ini": slope_ini,
  "rescaling": rescaling,
  "estim_algorithm": estim_algorithm
}


if onsky == 0:
    lightsource_estim = 'InternalPupil_'
    
    if Assuming_VLT_PUP_for_corr == 0:
        lightsource_corr = 'InternalPupil_'
    else:
        lightsource_corr = 'VLTPupil_'
else:
    lightsource_estim = 'VLTPupil_'
    lightsource_corr = 'VLTPupil_'

lightsource_estim = lightsource_estim + coro + '_'
lightsource_corr = lightsource_corr + coro + '_'

param['lightsource_estim'] = lightsource_estim
param['lightsource_corr'] = lightsource_corr

if coro == 'APLC':
    if zone_to_correct == 'vertical':
        posprobes = [678 , 679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
    elif zone_to_correct == 'horizontal':
        posprobes = [893 , 934]

    
elif coro == 'FQPM':
    if zone_to_correct == 'vertical':
        posprobes = [678 , 679]#0.3cutestimation*squaremaxPSF*8/amplitude pour internal pup    #0.2*squaremaxPSF*8/amplitude pour on sky
    elif zone_to_correct == 'horizontal':
        posprobes = [1089 , 1125] #FQPM

param['posprobes'] = posprobes



func.FullIterEFC(param)

# file_path = ''
# func.record_slope_from_file(param, file_path, amplitude_in_nm, slope_ini, name_slope_file)

