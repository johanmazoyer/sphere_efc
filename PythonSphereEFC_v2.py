#Python EFC code
#Version 2021/10/14

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.optimize as opt
from scipy.optimize import fmin_powell as fmin_powell
import scipy.ndimage as snd
from astropy.io import fits
import glob
import warnings
warnings.filterwarnings("ignore")

# Retrieve the shell variables
# RootDirectory = os.environ['WORK_PATH0']#'/vltuser/sphere/jmilli/test_EFC_20190830/PackageEFConSPHERE/'
# ImageDirectory = os.environ['WORK_PATH']#RootDirectory+'SlopesAndImages/'
# MatrixDirectory = os.environ['MATRIX_PATH']#RootDirectory+'MatricesAndModel/'
RootDirectory = '/vltuser/sphere/zwahhaj/efc/sphere_efc-main/'
ImageDirectory = RootDirectory+'SlopesAndImages/'
MatrixDirectory = RootDirectory+'MatricesAndModel/'
nbiter = int(os.environ['nbiter']) #nb of the iteration (first iteration is 1)
exp_name = os.environ['EXP_NAME'] #Rootname of the Experiment
dhsize = os.environ['DHsize']#Dark hole shape and size choice
corr_mode = os.environ['corr_mode']#Correction mode (more or less agressive)
nbprobe = os.environ['nbprobe']#number of probing actuators
x0_up = int(os.environ['X0UP'])#x position in python of the upper PSF echo
y0_up = int(os.environ['Y0UP'])#y position in python of the upper PSF echo
x1_up = int(os.environ['X1UP'])#x position in python of the bottom PSF echo
y1_up = int(os.environ['Y1UP'])#y position in python of the bottom PSF echo
which_nd = os.environ['WHICH_ND']# which ND is used to record the Off-axis PSF
onsky = int(os.environ['ONSKY'])#If 1: On sky measurements ; If 0: Calibration source measurements
Assuming_VLT_PUP_for_corr = int(os.environ['Assuming_VLT_PUP_for_corr'])
size_probes = int(os.environ['size_probes'])# Set the size of the probes for the estimation
centeringateachiter = int(os.environ['centeringateachiter'])#If 1: Do the recentering at each iteration ; If 0: Do not do it

print('Your working path is {0:s} and you are doing iteration number {1:d} of {2:s}'.format(RootDirectory,nbiter,exp_name), flush=True)

# plotting options
matplotlib.rcParams['font.size'] = 17
colors = [color['color'] for color in list(plt.rcParams['axes.prop_cycle'])]
# Generic setup

ND35 = 1/0.00105
ND2 = 1/0.0179



def SHslopes2map(slopes, visu=True):
    """
    Takes in input a 1D vector of 2480 elements, map it to the SH WFS and returns
    2 maps of the slopes in x and y
    """
    if len(slopes) != 2480:
        raise IOError('The input vector must have 2480 elements (currently {0:d})'.format(len(slopes)))
    mapx = np.ndarray((40,40),dtype=float)*np.nan
    mapy = np.ndarray((40,40),dtype=float)*np.nan
    shackgrid = fits.getdata(MatrixDirectory+'shack.grid.fits')
    mapx[shackgrid>0] = slopes[np.arange(1240,dtype=int)*2]
    mapy[shackgrid>0] = slopes[np.arange(1240,dtype=int)*2+1]
    mapx = np.fliplr(np.rot90(mapx,-3))
    mapy = np.fliplr(np.rot90(mapy,-3))
    if visu: 
        fig, ax = plt.subplots(1,2)
        im = ax[0].imshow(mapx, cmap='CMRmap', origin='lower',interpolation='nearest',\
            vmin=np.nanmin(slopes),vmax=np.nanmax(slopes))
        ax[0].set_title('SH slopes X')
        ax[1].imshow(mapy, cmap='CMRmap', origin='lower',interpolation='nearest',\
            vmin=np.nanmin(slopes),vmax=np.nanmax(slopes))
        ax[1].set_title('SH slopes Y')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    return mapx,mapy


def SaveFits(image, head,doc_dir2, name):
    """ --------------------------------------------------
    Save fits file
    
    Parameters:
    ----------
    image: float, file to save
    head: list, header
    doc_dir2: str, directory where to save the file
    name: str, name of the saved file
    -------------------------------------------------- """
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdr = hdul[0].header
    hdr.set(head[0],head[1])
    hdu.writeto(doc_dir2+name+'.fits',overwrite=True)
    
def roundpupil(nbpix, prad1):
    """ --------------------------------------------------
    Create a round pupil (binary mask)
    
    Parameters:
    ----------
    nbpix: int, number of pixels for the 2D array created
    prad1: int, radius of the pupil created in pixels 

    Return:
    ------
    pupilnormal: 2D array, binary mask
    -------------------------------------------------- """
    xx, yy = np.meshgrid(np.arange(nbpix)-(nbpix)/2, np.arange(nbpix)-(nbpix)/2)
    rr     = np.hypot(yy, xx)
    pupilnormal = np.zeros((nbpix,nbpix))
    pupilnormal[rr<=prad1] = 1.
    return pupilnormal

def estimateEab(Difference, Vecteurprobes):
    """ --------------------------------------------------
    Estimate focal plane electric field with PW probing
    
    Parameters:
    ----------
    Difference: Difference of PW images
    Vecteurprobes: Matrix PW

    Return:
    ------
    Resultat: 2D array, complex, focal plane electric field
    -------------------------------------------------- """
    numprobe = len(Vecteurprobes[0,0])
    Differenceij = np.zeros((numprobe))
    Resultat=np.zeros((dimimages,dimimages),dtype=complex)
    l = 0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            Differenceij[:] = Difference[:,i,j]
            Resultatbis = np.dot(Vecteurprobes[l],Differenceij)
            Resultat[i,j] = Resultatbis[0]+1j*Resultatbis[1]
            
            l = l + 1  
    return Resultat/4
       
   
def solutiontocorrect(mask, ResultatEstimate, invertG):
    """ --------------------------------------------------
    Solution in nanometer to dig the DH at next iteration
    
    Parameters:
    ----------
    mask: 2D array, DH region
    ResultatEstimate: 2D array, focal plane elecric field
    invertG: inverted jacobian

    Return:
    ------
    solition: 1D array, floats with nanometers
    -------------------------------------------------- """
    Eab = np.zeros(2*int(np.sum(mask)))
    Resultatbis = (ResultatEstimate[np.where(mask==1)])
    Eab[0:int(np.sum(mask))] = np.real(Resultatbis).flatten()     
    Eab[int(np.sum(mask)):] = np.imag(Resultatbis).flatten()
    cool = np.dot(invertG,Eab)
    
    solution = np.zeros(int(1377))
    solution[WhichInPupil] = cool
    return solution
    

# def cost_function(xy_trans):
#     # Function can use image slices defined in the global scope
#     # Calculate X_t - image translated by x_trans
#     unshifted = fancy_xy_trans_slice(imagecorrection, xy_trans)
#     #Return mismatch measure for the translated image X_t
#     return correl_mismatch(imageref,unshifted)
    
def fancy_xy_trans_slice(img_slice, xy_trans):
    """ Return copy of `img_slice` translated by `x_vox_trans` pixels
    
    Parameters
    
    img_slice : array shape (M, N)
    2D image to transform with translation `x_vox_trans`
    x_vox_trans : float
    Number of pixels (pixels) to translate `img_slice`; can be
    positive or negative, and does not need to be integer value.
    """
    #Resample image using bilinear interpolation (order=1)
    trans_slice = snd.affine_transform(img_slice, [1, 1], xy_trans, order=1)
    return trans_slice

def my_callback(params):
    print("Trying parameters " + str(params), flush=True)


def correl_mismatch(slice0, slice1):
    """ Negative correlation between the two images, flattened to 1D """
    correl = np.corrcoef(slice0.ravel(), slice1.ravel())[0, 1]
    #print(correl)
    return -correl
    

def last(files):
    extract = sorted(glob.glob(files))[-1]
    return extract


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


def get_exptime(file):
    exptime = int(round(fits.getval(file,'EXPTIME')))
    return exptime


def reduceimageSPHERE(file, directory,  maxPSF, ctr_x, ctr_y, newsizeimg, exppsf, ND):
    """ --------------------------------------------------
    Processing of SPHERE images before being used and division by the maximum of the PSF
    
    Parameters:
    ----------
    image: 2D array, raw image
    back: 2D array, background image
    maxPSF: int, maximum of the raw PSF
    ctr_x: int, center of processed image in the x direction
    ctr_y: int, center of processed image in the y direction
    newsizeimg: int, size of the processed image (same dimension in x and y)
    expim: float, exposure time of the image in second
    exppsf: float, exposure time of the recorded PSF in second
    ND: float, neutral density attenuation factor used when recording PSF 

    Return:
    ------
    image: processed coronagraphic image, normalized by the max of the PSF
    -------------------------------------------------- """
    expim = get_exptime(file) #Get image exposure time
    back = fits.getdata(last(directory+'SPHERE_BKGRD_EFC_'+str(int(expim))+'s_*.fits'))[0] #Load dark that correspond to image exposure time
    image = fits.getdata(file)[0] #Load image
    image[:,:int(image.shape[1]/2)] = 0 #Cancel left part of image
    image = image - back #Subtract dark
    image = (image/expim)/(maxPSF*ND/exppsf)  #Divide by PSF max
    image = cropimage(image,ctr_x,ctr_y,newsizeimg) #Crop to keep relevant part of image
    return image


def createdifference(directory, filenameroot, posprobes, nbiter, centerx, centery):
    """ --------------------------------------------------
    Create difference cube of PW images
    
    Parameters:
    ----------
    directory: str, directory of the images lication
    filenameroot: str, name of the files
    posprobes: array, index of poked actuators used for PW
    nbiter: Iteration number to retrieve the good data
    ctr_x: int, center of processed image in the x direction
    ctr_y: int, center of processed image in the y direction

    Return:
    ------
    Difference: 3D array, cube of images
    -------------------------------------------------- """
    
    if which_nd == 'ND_3.5':
        ND = ND35
    elif which_nd == 'ND_2.0':
        ND = ND2
    else:
        ND = 1.

    
    #PSF
    file_PSF = last(directory+lightsource_estim+'OffAxisPSF*.fits')
    exppsf = get_exptime(file_PSF)
    PSF = reduceimageSPHERE(file_PSF, directory, 1,int(centerx),int(centery),dimimages,1,1)
    smoothPSF = snd.median_filter(PSF,size=3)
    maxPSF = PSF[np.unravel_index(np.argmax(smoothPSF, axis=None), smoothPSF.shape)[0] , np.unravel_index(np.argmax(smoothPSF, axis=None), smoothPSF.shape)[1] ]

    print('!!!! ACTION: MAXIMUM PSF HAS TO BE VERIFIED ON IMAGE: ',maxPSF, flush=True)
    
    #Correction
    
    #Traitement de l'image de référence (première image corono et recentrage subpixelique)
    fileref = last(directory+filenameroot+'iter0_coro_image*.fits')
    imageref = reduceimageSPHERE(fileref,directory,maxPSF,int(centerx),int(centery),dimimages,exppsf,ND)
    imageref = fancy_xy_trans_slice(imageref, [centerx-int(centerx), centery-int(centery)])
    imageref=cropimage(imageref,int(dimimages/2),int(dimimages/2),int(dimimages/2))

    filecorrection = last(directory+filenameroot+'iter'+str(nbiter-2)+'_coro_image*.fits')
    imagecorrection = reduceimageSPHERE(filecorrection, directory, maxPSF,int(centerx),int(centery),dimimages,exppsf,ND)
    imagecorrection = cropimage(imagecorrection,int(dimimages/2),int(dimimages/2),int(dimimages/2))
    Contrast = np.mean(imagecorrection[np.where(cropimage(maskDH,int(dimimages/2),int(dimimages/2),int(dimimages/2)))])
    print('Contrast in DH region at iter '+str(nbiter-2)+ ' = ' , Contrast, flush=True)
    
    with open(directory + filenameroot + 'Contrast_vs_iter', 'a') as f:
        f.write(str(Contrast) + '\n')

    def cost_function(xy_trans):
        # Function can use image slices defined in the global scope
        # Calculate X_t - image translated by x_trans
        unshifted = fancy_xy_trans_slice(imagecorrection, xy_trans)
        mask = roundpupil(int(dimimages/2),67)
        #Return mismatch measure for the translated image X_t
        return correl_mismatch(imageref[np.where(mask==0)], unshifted[np.where(mask==0)])
        
    
    if centeringateachiter == 1:
        #Calcul de la translation du centre par rapport à la réference: best param.
        #Les images probes sont ensuite translatées de best param
        best_params = fmin_powell(cost_function, [0, 0], disp=0, callback=None) #callback=my_callback
        print('   Shifting recorded images by: ',best_params,' pixels' , flush=True)
        imagecorrection = fancy_xy_trans_slice(imagecorrection, best_params)
    else:
        print('No recentering', flush=True)
        best_params = [centerx-int(centerx), centery-int(centery)]
        imagecorrection = fancy_xy_trans_slice(imagecorrection, best_params)
    
    
    #Probes
    numprobes = len(posprobes)
    Difference = np.zeros((numprobes,dimimages,dimimages))    
    k = 0
    j = 1
    
    display(imagecorrection, ax1, '', vmin=1e-7, vmax=1e-3, norm='log')
    ax1.text(1, 12, 'Contrast = '+str(format(Contrast,'.2e')), size=15,color ='red', weight='bold')
    PSF_to_display = cropimage(PSF,np.unravel_index(np.argmax(smoothPSF, axis=None), smoothPSF.shape)[0] , np.unravel_index(np.argmax(smoothPSF, axis=None), smoothPSF.shape)[1] , 70 )
    ax1bis = fig1.add_axes([0.65, 0.70, 0.25, 0.25])
    display(PSF_to_display, ax1bis, 'PSF' , vmin = np.amin(PSF_to_display), vmax = np.amax(PSF_to_display))

    

    for i in posprobes:
        image_name = last(directory+filenameroot+'iter'+str(nbiter-1)+'_Probe_'+'%04d' % j+'*.fits')
        #print('Loading the probe image {0:s}'.format(image_name), flush=True)
        Ikplus = reduceimageSPHERE(image_name, directory, maxPSF,int(centerx),int(centery),dimimages,exppsf,ND)
        Ikplus = fancy_xy_trans_slice(Ikplus, best_params)
        display(cropimage(Ikplus,int(dimimages/2),int(dimimages/2),int(dimimages/2))-imagecorrection, ax2.flat[j-1] , title='+ '+str(i), vmin=-1e-4, vmax=1e-4)
        j = j + 1
        image_name = last(directory+filenameroot+'iter'+str(nbiter-1)+'_Probe_'+'%04d' % j+'*.fits')
        #print('Loading the probe image {0:s}'.format(image_name), flush=True)
        Ikmoins = reduceimageSPHERE(image_name, directory, maxPSF,int(centerx),int(centery),dimimages,exppsf,ND)
        Ikmoins = fancy_xy_trans_slice(Ikmoins, best_params)
        display(cropimage(Ikmoins,int(dimimages/2),int(dimimages/2),int(dimimages/2))-imagecorrection, ax2.flat[j-1] , title='- '+str(i), vmin=-1e-4, vmax=1e-4)
        j = j + 1
        Difference[k] = (Ikplus-Ikmoins)
        k = k + 1

    return Difference


def display(image, axe, title, vmin, vmax , norm = None):
    """ --------------------------------------------------
    Create a binary mask.
    
    Parameters:
    ----------
    image: 2D array, image to display
    axe: name of the axe to display
    title: str, name of the image
    vmin: float, min scale of the display
    vmax: float, max scale of the display
    norm: Can be LogNorm

    -------------------------------------------------- """
    if norm == 'log':
        axe.imshow(image, norm=LogNorm(vmin, vmax))
    else:
        axe.imshow(image, vmin = vmin, vmax = vmax)
    #axe.imshow(image, norm = norm)
    axe.set_xticks([])
    axe.set_yticks([])
    axe.set_title(title,size=7)
    plt.draw()
    plt.pause(0.1)
    return 0

def resultEFC(directory, filenameroot, posprobes, nbiter, centerx, centery):
    """ --------------------------------------------------
    Give EFC solution in WFS slope
    
    Parameters:
    ----------
    directory:
    filenameroot:
    posprobes:
    nbiter:
    centerx:
    centery:

    Return:
    ------
    resultatestimation:
    slopes: 
    -------------------------------------------------- """
    print('- Creating difference of images...', flush=True)
    Difference = createdifference(directory, filenameroot, posprobes, nbiter, centerx, centery)
    print('- Estimating the focal plane electric field...', flush=True)
    resultatestimation = estimateEab(Difference, vectoressai)
    print('- Calculating slopes to generate the Dark Hole with EFC...', flush=True)
    gain = 0.5
    solution1 = solutiontocorrect(maskDH, resultatestimation, invertGDH)
    solution1 = solution1*amplitudeEFCMatrix/rad_632_to_nm_opt
    solution1 = -gain*solution1
    slopes = VoltToSlope(solution1)
    return resultatestimation, slopes
        



def recordslopes(slopes, dir, refslope, namerecord):
    """ --------------------------------------------------
    Save a slope.
    
    Parameters:
    ----------
    slopes:
    dir:
    refslope:
    namerecord:
    
    -------------------------------------------------- """
    ref = dir + refslope + '.fits'
    #hdul = fits.open(ref) #Charge la forme des pentes actuelles
    data, header1 = fits.getdata(ref, header=True)
    fits.writeto(dir+namerecord+'.fits', np.asarray(data+slopes,dtype=np.float32), header1, overwrite=True)
    return 0
    
   
   
    
def recordnewprobes(amptopush, acttopush, dir, refslope, nbiter):
    """ --------------------------------------------------
    Save new slopes to create PW probes on the DM
    
    Parameters:
    ----------
    amptopush:
    acttopush:
    dir:
    refslope:
    nbiter:

    -------------------------------------------------- """
    k = 1
    for j in acttopush:
        tensionDM = np.zeros(1377)
        tensionDM[j] = amptopush/37/rad_632_to_nm_opt
        slopetopush = VoltToSlope(tensionDM)
        recordslopes(slopetopush, dir, refslope, 'iter'+str(nbiter)+'probe'+str(k))
        k = k + 1
        
        tensionDM = np.zeros(1377)
        tensionDM[j] = -amptopush/37/rad_632_to_nm_opt
        slopetopush = VoltToSlope(tensionDM)
        recordslopes(slopetopush,dir,refslope,'iter'+str(nbiter)+'probe'+str(k))
        k = k + 1
    return 0
        
def FullIterEFC(dir, posprobes, nbiter, filenameroot, record=False):
    """ --------------------------------------------------
    PW + EFC Iteration wrapper
    
    Parameters:
    ----------
    dir:
    posprobes:
    nbiter:
    filenameroot:
    record:

    -------------------------------------------------- """

    #Check if the directory dir exists
    if os.path.isdir(dir) is False:
        #Create the directory
        os.mkdir(dir)

    if nbiter == 1:
        print('Creating slopes for Cosinus, PSFOffAxis and new probes...', flush=True)
        #Reference slopes
        refslope = 'VisAcq.DET1.REFSLP'
        #Copy the reference slope with the right name for iteration 0 of ExperimentXXXX
        recordslopes(np.zeros(2480),dir,refslope,filenameroot+'iter0correction')
        #Create the cosine of 10nm peak-to-valley amplitude for centering
        #Same name for all experiments because all iteration 0 are with the same reference slopes VisAcq.DET1.REFSLP
        recordCoswithvolt(10,dir,refslope)
        #Create DDTS slopes at (+10,+10) for off-axis PSF recording
        #Same name for all experiments because all iteration 0 are with the same reference slopes IRAcq.DET1.REFSLP
        #refslope='IRAcq.DET1.REFSLP'
        #recordslopes(10*np.ones(2),dir,refslope,'IRAcq.DET1.PSFOffAxis')
        #Add the probes to reference slopes
        #Same for all experiments but the filename is changing
        refslope = 'iter' + str(nbiter-1) + 'correction'
        recordnewprobes(size_probes, posprobes, dir+filenameroot, refslope,nbiter)
    else:
        #Calculate the center of the first coronagraphic image using the waffle
        if nbiter == 2:
            print('Calculating center of the first coronagraphic image:', flush=True)
            centerx,centery = findingcenterwithcosinus(dir+filenameroot)
            SaveFits([centerx,centery],['',0],dir+filenameroot,'centerxy')
        
        centerx,centery = fits.getdata(dir+filenameroot+'centerxy.fits')
        #Estimation of the electric field using the pair-wise probing (return the electric field and the slopes)
        print('Estimating the electric field using the pair-wise probing:', flush=True)
        estimation,pentespourcorrection = resultEFC(dir,filenameroot,posprobes,nbiter,centerx,centery)
        if record == True:
            #Record the slopes to apply for correction at the next iteration
            refslope = 'iter' + str(nbiter-2) + 'correction'
            recordslopes(pentespourcorrection, dir+filenameroot, refslope, 'iter'+str(nbiter-1)+'correction')
            #Record the slopes to apply for probing at the next iteration
            refslope = 'iter' + str(nbiter-1) + 'correction'
            recordnewprobes(size_probes, posprobes, dir+filenameroot, refslope, nbiter)
            
        estimation_to_display = cropimage(estimation*maskDH, int(dimimages/2), int(dimimages/2), int(dimimages/2))
        display(np.real(estimation_to_display), ax3.flat[0] , title='Real estim. iter' + str(nbiter-1), vmin=-1e-2, vmax=1e-2)
        display(np.imag(estimation_to_display), ax3.flat[2] , title='Imag estim. iter' + str(nbiter-1), vmin=-1e-2, vmax=1e-2)
        print('Done with recording new slopes!', flush=True)
        
        slopes_to_display = pentespourcorrection + fits.getdata(dir+filenameroot+refslope+'.fits')[0]
        display(SHslopes2map(slopes_to_display, visu=False)[0], ax3.flat[1], title = 'Slopes SH in X to apply for iter'+str(nbiter-1), vmin =np.amin(slopes_to_display), vmax = np.amax(slopes_to_display) )
        display(SHslopes2map(slopes_to_display, visu=False)[1], ax3.flat[3], title = 'Slopes SH in Y to apply for iter'+str(nbiter-1), vmin =np.amin(slopes_to_display), vmax = np.amax(slopes_to_display) )
        
    return 0


    
def VoltToSlope(Volt):
    """ --------------------------------------------------
    Conversion volt on the DM to slope on the WFS
    
    Parameters:
    ----------
    Volt:

    Return:
    ------
    Slopetopush: 
    -------------------------------------------------- """
    Slopetopush = V2S @ Volt
    return Slopetopush
    

    
def recordCoswithvolt(amptopushinnm, dir, refslope):
    """ --------------------------------------------------
    Creation of the cosinus (in slope) to apply on the DM    
    
    Parameters:
    ----------
    amptopushinnm:
    dir:
    refslope:

    -------------------------------------------------- """
    nam = ['cos_00deg','cos_30deg','cos_90deg']
    nbper = 10.
    dim = 240
    xx, yy = np.meshgrid(np.arange(240)-(240)/2, np.arange(240)-(240)/2)
    cc = np.cos(2*np.pi*(xx*np.cos(0*np.pi/180.))*nbper/dim)
    coe = (cc.flatten())@IMF_inv
    #print(np.amax(coe),np.amin(coe), flush=True)
    coe = coe*amptopushinnm/rad_632_to_nm_opt#/37/rad_632_to_nm_opt
    #print(np.amax(coe),np.amin(coe), flush=True)
    slopetopush = V2S@coe
    recordslopes(slopetopush,dir,refslope,nam[0]+'_'+str(amptopushinnm)+'nm')
    return 0
    
    
    
def findingcenterwithcosinus(dir):    
    """ --------------------------------------------------
    Find center of the coronagraphic image using previously apploed cosinus.
    
    Parameters:
    ----------
    dir: location of the cosinus

    Return:
    ------
    centerx, centery: floats, center of the coronagraphic image
    -------------------------------------------------- """
    
    def twoD_Gaussian(xy, amplitude, sigma_x, sigma_y, xo, yo,h):
        xo = float(xo)
        yo = float(yo) 
        theta=0   
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
        g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                + c*((y-yo)**2)))+h
        
        return (g).flatten()
    
    #LOOK THE FITS FILE AND CHANGE QUIKLY X0,Y0,X1,Y1
    cosinuspluscoro = last(dir+'CosinusForCentering*.fits')
    coro = last(dir+'iter0_coro_image*.fits')
    
    #Fit gaussian functions
    data = fits.getdata(cosinuspluscoro)[0]-fits.getdata(coro)[0]
    data1 = cropimage(data,x0_up,y0_up,30)
    data2 = cropimage(data,x1_up,y1_up,30)
    data1[np.where(data1<0)] = 0
    data2[np.where(data2<0)] = 0
   
    w,h = data1.shape
    x, y = np.mgrid[0:w, 0:h]
    xy = (x,y)
                
    #Fit 2D Gaussian with fixed parameters for the top PSF
    initial_guess = (np.amax(data1), 1 , 1 , np.unravel_index(np.argmax(data1, axis=None), data1.shape)[0] , np.unravel_index(np.argmax(data1, axis=None), data1.shape)[1] ,np.mean(data1))
    
    try:
        popt1, pcov = opt.curve_fit(twoD_Gaussian, xy, (data1).flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed top PSF", flush=True)

    #Fit 2D Gaussian with fixed parameters for the bottom PSF
    initial_guess = (np.amax(data2), 1 , 1 , np.unravel_index(np.argmax(data2, axis=None), data2.shape)[0] , np.unravel_index(np.argmax(data2, axis=None), data2.shape)[1] ,np.mean(data2))
    
    try:
        popt2, pcov = opt.curve_fit(twoD_Gaussian, xy, (data2).flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed bottom PSF", flush=True)
        
    centerx = ((popt1[3]+x0_up-15)+(popt2[3]+x1_up-15))/2
    centery = ((popt1[4]+y0_up-15)+(popt2[4]+y1_up-15))/2
    
    print('- centerx = ',centery, flush=True)
    print('- centery = ',centerx, flush=True)
    print('!!!! ACTION: CHECK CENTERX AND CENTERY CORRESPOND TO THE CENTER OF THE CORO IMAGE AND CLOSE THE WINDOW', flush=True)
    print('If false, change the guess in the MainEFCBash.sh file and run the iter again', flush=True)

    #I do not use crop of data here so the center index can be checked on the full image.
    display(data, ax4, 'Cosine for centering', vmin=-5e2, vmax=5e2)

    return centerx, centery


#Amplitude in x nm/37 for the pokes to create the jacobian matrix such that pushact amplitude is equal to x nm (usually 296nm here)
amplitudeEFCMatrix = 8
dimimages = 400

# Read SAXO calibrations
# static calibrations
IMF_inv = fits.getdata(MatrixDirectory + 'SAXO_DM_IFM_INV.fits', ignore_missing_end=True)  


#Unit conversion and normalisation
#influence matrix normalization = defoc meca en rad @ 632 nm
rad_632_to_nm_opt = 632/2/np.pi
#rad_632_to_nm_opt = 1 / 2 / np.pi * 1600 * 2   #ESAIIIIIII
PHI2V = (IMF_inv / rad_632_to_nm_opt).T   #Transpose


# daily calibrations !! Ask for HO_IM matrix before starting and save in directory!! ####################
V2S = fits.getdata(MatrixDirectory+ 'CLMatrixOptimiser.HO_IM.fits')
# renormalization for weighted center of gravity: 0.4 sensitivity
V2S = V2S / 0.4

if onsky == 0:
    lightsource_estim = 'InternalPupil_'
    
    if Assuming_VLT_PUP_for_corr == 0:
        lightsource_corr = 'InternalPupil_'
    else:
        lightsource_corr = 'VLTPupil_'
else:
    lightsource_estim = 'VLTPupil_'
    lightsource_corr = 'VLTPupil_'

if nbprobe == '2':
    posprobes = [678, 679]
if nbprobe == '3':
    posprobes = [678,679,680]
if nbprobe == '4':
    posprobes = [678,679,680,681]

vectoressai = fits.getdata(MatrixDirectory+lightsource_estim+'VecteurEstimation_'+nbprobe+'probes'+str(size_probes)+'nm.fits')
WhichInPupil = fits.getdata(MatrixDirectory+'WhichInPupil0_5.fits')


#Dark hole size
#DHsize = 0 for half dark hole 188mas to 625mas x -625mas to 625mas
#DHsize = 1 for half dark hole 125mas to 625mas x -625mas to 625mas
#DHsize = 2 for full dark hole -625mas to 625mas x -625mas to 625mas

## corr_mode: choose the SVD cutoff of the EFC correction
# corr_mode=0: stable correction but moderate contrast
# corr_mode=1: less stable correction but better contrast
# corr_mode=2: more aggressive correction (may be unstable)
plt.close()

fig = plt.figure(figsize=(12,7.5))
( fig1 , fig2 ) , ( fig3 , fig4 ) = fig.subfigures(2, 2)
fig1.suptitle('Coro image iter'+str(nbiter-2), size=10)
fig2.suptitle('Probe images iter'+str(nbiter-1), size=10)
#fig3.subtitle('Miscellaneous')
ax1 = fig1.subplots(1,1,sharex=True,sharey=True)
ax2 = fig2.subplots(2,2,sharex=True,sharey=True)
ax3 = fig3.subplots(2,2)
ax4 = fig4.subplots(1,1)
maskDH = fits.getdata(MatrixDirectory+'mask_DH'+str(dhsize)+'.fits')


invertGDH = fits.getdata(MatrixDirectory+lightsource_corr+'Interactionmatrix_DH'+str(dhsize)+'_SVD'+str(corr_mode)+'.fits')

FullIterEFC(ImageDirectory, posprobes, nbiter, exp_name, record=True)


if nbiter>1:
    if nbiter>2:
        pastContrast = []
        with open(ImageDirectory + exp_name + 'Contrast_vs_iter') as f:
            pastContrast = np.float64(f.readlines())
        ax4.plot(pastContrast, marker='o', markersize= 8, mfc='none')
        ax4.set_yscale('log')
        ax4.set_ylim(1e-7,1e-4)
        ax4.tick_params(axis='both', which='both', labelsize=8)
        ax4.set_title('Mean contrast in DH vs iteration',size=10)
    
    print('Close each image to proceed', flush=True)
    plt.draw()
    plt.show()
else:
    plt.close()

