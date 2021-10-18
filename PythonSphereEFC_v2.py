#Python EFC code
#Version 2019/11/27

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as cols

import scipy
import scipy.signal as scsi
import scipy.optimize as opt
import scipy.fftpack as fft
from scipy.optimize import curve_fit
from scipy.optimize import fmin_powell as fmin_powell
import scipy.ndimage as snd

import datetime

from pathlib import Path,PurePath

from astropy.io import fits
from astropy.time import Time
from astropy.modeling import models, fitting
from astropy.io import fits

import sys

import os
import cv2

import glob

import matplotlib.pyplot as plt
import numpy as np
from poppy import matrixDFT

import warnings
warnings.filterwarnings("ignore")

# Retrieve the shell variables
RootDirectory = os.environ['WORK_PATH0']#'/vltuser/sphere/jmilli/test_EFC_20190830/PackageEFConSPHERE/'
ImageDirectory =os.environ['WORK_PATH']#RootDirectory+'SlopesAndImages/'
MatrixDirectory=os.environ['MATRIX_PATH']#RootDirectory+'MatricesAndModel/'
nbiter = int(os.environ['nbiter']) #nb of the iteration (first iteration is 1)
exp_name=os.environ['EXP_NAME'] #Rootname of the Experiment
dhsize=os.environ['DHsize']#Dark hole shape and size choice
corr_mode=os.environ['corr_mode']#Correction mode (more or less agressive)
nbprobe=os.environ['nbprobe']#number of probing actuators
x0_up = int(os.environ['X0UP'])#x position in python of the upper PSF echo
y0_up = int(os.environ['Y0UP'])#y position in python of the upper PSF echo
x1_up = int(os.environ['X1UP'])#x position in python of the bottom PSF echo
y1_up = int(os.environ['Y1UP'])#y position in python of the bottom PSF echo
expimIRD = float(os.environ['DIT'])#DIT for the coronagraphic images
exppsfIRD = float(os.environ['DIT_PSF'])#DIT for the Off-axis PSF
which_nd = os.environ['WHICH_ND']# which ND is used to record the Off-axis PSF
onsky=int(os.environ['ONSKY'])#If 1: On sky measurements ; If 0: Calibration source measurements
size_probes=int(os.environ['size_probes'])# Set the size of the probes for the estimation
centeringateachiter=int(os.environ['centeringateachiter'])#If 1: Do the recentering at each iteration ; If 0: Do not do it

print('Your working path is {0:s} and you are doing iteration number {1:d} of {2:s}'.format(RootDirectory,nbiter,exp_name))

# plotting options
matplotlib.rcParams['font.size'] = 17
colors = [color['color'] for color in list(plt.rcParams['axes.prop_cycle'])]
# Generic setup

ND35=1/0.00105
ND2=1/0.0179

# Constants
# pupil dimensions
dim_pupil_saxo  = 240
dim_pupil_irdis = 384

# mas/pixel
pixel_irdis = 12.25
pixel_dtts  = 12.25
pixel_ifs   = 7.43

# wavelengths
wave_FeII = 1.642e-6
wave_H2   = 1.593e-6
wave_H3   = 1.667e-6

# lambda/D
loD_FeII = wave_FeII / 8 * 180 / np.pi * 3600 * 1000
loD_H2   = wave_H2 / 8 * 180 / np.pi * 3600 * 1000
loD_H3   = wave_H3 / 8 * 180 / np.pi * 3600 * 1000



def SHslopes2map(slopes,visu=True):
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


def LoadImageFits(docs_dir):
    openfits=fits.open(docs_dir)
    image=openfits[0].data
    return image

def SaveFits(image,head,doc_dir2,name):
    hdu=fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdr=hdul[0].header
    hdr.set(head[0],head[1])
    hdu.writeto(doc_dir2+name+'.fits',overwrite=True)
    
def roundpupil(nbpix,prad1):
    xx, yy = np.meshgrid(np.arange(nbpix)-(nbpix)/2, np.arange(nbpix)-(nbpix)/2)
    rr     = np.hypot(yy, xx)
    pupilnormal=np.zeros((nbpix,nbpix))
    pupilnormal[rr<=prad1]=1.
    return pupilnormal

def estimateEab(Difference,Vecteurprobes):
    numprobe=len(Vecteurprobes[0,0])
    Differenceij=np.zeros((numprobe))
    Resultat=np.zeros((dimimages,dimimages),dtype=complex)
    l=0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            Differenceij[:]=Difference[:,i,j]
            Resultatbis=np.dot(Vecteurprobes[l],Differenceij)
            Resultat[i,j]=Resultatbis[0]+1j*Resultatbis[1]
            
            l=l+1  
    return Resultat/4
       
   
def solutiontocorrect(mask,ResultatEstimate,invertG):
    Eab=np.zeros(2*int(np.sum(mask)))
    Resultatbis=(ResultatEstimate[np.where(mask==1)])
    Eab[0:int(np.sum(mask))]=real(Resultatbis).flatten()     
    Eab[int(np.sum(mask)):]=im(Resultatbis).flatten()
    cool=np.dot(invertG,Eab)
    
    solution=np.zeros(int(1377))
    solution[WhichInPupil]=cool
    return solution
    

def cost_function(xy_trans):
    # Function can use image slices defined in the global scope
    # Calculate X_t - image translated by x_trans
    unshifted = fancy_xy_trans_slice(imagecorrection, xy_trans)
    #Return mismatch measure for the translated image X_t
    return correl_mismatch(imageref,unshifted)
    
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
    print("Trying parameters " + str(params))


def correl_mismatch(slice0, slice1):
    """ Negative correlation between the two images, flattened to 1D """
    correl = np.corrcoef(slice0.ravel(), slice1.ravel())[0, 1]
    #print(correl)
    return -correl
    



def cropimage(img,ctr_x,ctr_y,newsizeimg):
    lenimg=len(img)
    imgs2=lenimg/2
    newimgs2=newsizeimg/2
    return img[int(ctr_x-newimgs2):int(ctr_x+newimgs2),int(ctr_y-newimgs2):int(ctr_y+newimgs2)]



def reduceimageSPHERE(image,back,maxPSF,ctr_x,ctr_y,newsizeimg,expim,exppsf,ND):
    image[:,:int(image.shape[1]/2)]=0 #Annule la partie gauche de l'image
    image=image-back #Soustrait le dark
    image=(image/expim)/(maxPSF*ND/exppsf)  #Divise par le max de la PSF
    image=cropimage(image,ctr_x,ctr_y,newsizeimg) #Récupère la partie voulue de l'image
    return image




def createdifference(directory,filenameroot,posprobes,nbiter,centerx,centery):
    
    expim = expimIRD
    exppsf = exppsfIRD
    if which_nd == 'ND_3.5':
        ND=ND35
    elif which_nd == 'ND_2.0':
        ND=ND2
    else:
        ND=1.
        
    #Dark
    backgroundcorono = LoadImageFits(sorted(glob.glob(directory+'SPHERE_BKGRD_EFC_CORO*.fits'))[-1])[0]
    if expim==exppsf:
        backgroundPSF = backgroundcorono
    else:
        backgroundPSF = LoadImageFits(sorted(glob.glob(directory+'SPHERE_BKGRD_EFC_PSF*.fits'))[-1])[0]
    #PSF
    PSFbrut = LoadImageFits(sorted(glob.glob(directory+'OffAxisPSF*.fits'))[-1])[0]
    PSF = reduceimageSPHERE(PSFbrut,backgroundPSF,1,int(centerx),int(centery),dimimages,1,1,1)
    smoothPSF=snd.median_filter(PSF,size=3)
    maxPSF=PSF[np.unravel_index(np.argmax(smoothPSF, axis=None), smoothPSF.shape)[0] , np.unravel_index(np.argmax(smoothPSF, axis=None), smoothPSF.shape)[1] ]

    print('!!!! ACTION: MAXIMUM PSF HAS TO BE VERIFIED ON FITS: ',maxPSF)
    
    #Correction
    
    #Traitement de l'image de référence (première image corono et recentrage subpixelique)
    fileref = LoadImageFits(sorted(glob.glob(directory+filenameroot+'iter0_coro_image*.fits'))[-1])[0]
    imageref = reduceimageSPHERE(fileref,backgroundcorono,maxPSF,int(centerx),int(centery),dimimages,expim,exppsf,ND)
    imageref = fancy_xy_trans_slice(imageref, [centerx-int(centerx), centery-int(centery)])
    imageref=cropimage(imageref,int(dimimages/2),int(dimimages/2),int(dimimages/2))

    filecorrection = LoadImageFits(sorted(glob.glob(directory+filenameroot+'iter'+str(nbiter-2)+'_coro_image*.fits'))[-1])[0]
    imagecorrection=reduceimageSPHERE(filecorrection,backgroundcorono,maxPSF,int(centerx),int(centery),dimimages,expim,exppsf,ND)
    imagecorrection=cropimage(imagecorrection,int(dimimages/2),int(dimimages/2),int(dimimages/2))

    def cost_function(xy_trans):
        # Function can use image slices defined in the global scope
        # Calculate X_t - image translated by x_trans
        unshifted = fancy_xy_trans_slice(imagecorrection, xy_trans)
        mask=roundpupil(int(dimimages/2),67)
        #Return mismatch measure for the translated image X_t
        return correl_mismatch(imageref[np.where(mask==0)],unshifted[np.where(mask==0)])
        
    
    if centeringateachiter==1:
        #Calcul de la translation du centre par rapport à la réference: best param.
        #Les images probes sont ensuite translatées de best param
        best_params = fmin_powell(cost_function, [0, 0], disp=0, callback=None) #callback=my_callback
        print('   Shifting recorded images by: ',best_params,' pixels' )
        imagecorrection = fancy_xy_trans_slice(imagecorrection, best_params)
    else:
        print('No recentering')
        best_params = [centerx-int(centerx), centery-int(centery)]
        imagecorrection = fancy_xy_trans_slice(imagecorrection, best_params)
    
    
    #Probes
    numprobes=len(posprobes)
    Difference=np.zeros((numprobes,dimimages,dimimages))    
    k=0
    j=1
    for i in posprobes:
        image_name = sorted(glob.glob(directory+filenameroot+'iter'+str(nbiter-1)+'_Probe_'+'%04d' % j+'*.fits'))[-1]
        #print('Loading the probe image {0:s}'.format(image_name))
        image = LoadImageFits(image_name)[0]
        Ikplus = reduceimageSPHERE(image,backgroundcorono,maxPSF,int(centerx),int(centery),dimimages,expim,exppsf,ND)
        Ikplus = fancy_xy_trans_slice(Ikplus, best_params)
        j=j+1
        image_name = sorted(glob.glob(directory+filenameroot+'iter'+str(nbiter-1)+'_Probe_'+'%04d' % j+'*.fits'))[-1]
        #print('Loading the probe image {0:s}'.format(image_name))
        image = LoadImageFits(image_name)[0]
        Ikmoins = reduceimageSPHERE(image,backgroundcorono,maxPSF,int(centerx),int(centery),dimimages,expim,exppsf,ND)
        Ikmoins = fancy_xy_trans_slice(Ikmoins, best_params)
        Difference[k] = (Ikplus-Ikmoins)
        j = j+1
        k = k+1
        
    return Difference



def resultEFC(directory,filenameroot,posprobes,nbiter,centerx,centery):
    print('- Creating difference of images...')
    Difference=createdifference(directory,filenameroot,posprobes,nbiter,centerx,centery)
    print('- Estimating the focal plane electric field...')
    resultatestimation=estimateEab(Difference,vectoressai)
    print('- Calculating slopes to generate the Dark Hole with EFC...')
    gain=0.5
    solution1=solutiontocorrect(maskDH,resultatestimation,invertGDH)
    solution1=solution1*amplitude/rad_632_to_nm_opt
    solution1=-gain*solution1
    slopes=VoltToSlope(solution1)
    return resultatestimation,slopes
        



def recordslopes(slopes,dir,refslope,namerecord):
    ref=dir+refslope+'.fits'
    hdul = fits.open(ref) #Charge la forme des pentes actuelles
    data, header1 = fits.getdata(ref, header=True)
    fits.writeto(dir+namerecord+'.fits', np.asarray(data+slopes,dtype=np.float32), header1, overwrite=True)
    return 0
    
   
   
    
def recordnewprobes(amptopush,acttopush,dir,refslope,nbiter):
    k=1
    for j in acttopush:
        tensionDM=np.zeros(1377)
        tensionDM[j]=amptopush/37/rad_632_to_nm_opt
        slopetopush=VoltToSlope(tensionDM)
        recordslopes(slopetopush,dir,refslope,'iter'+str(nbiter)+'probe'+str(k))
        k=k+1
        
        tensionDM=np.zeros(1377)
        tensionDM[j]=-amptopush/37/rad_632_to_nm_opt
        slopetopush=VoltToSlope(tensionDM)
        recordslopes(slopetopush,dir,refslope,'iter'+str(nbiter)+'probe'+str(k))
        k=k+1
    return 0
        
def FullIterEFC(dir,posprobes,nbiter,filenameroot,record=False):

    #Check if the directory dir exists
    if os.path.isdir(dir) is False:
        #Create the directory
        os.mkdir(dir)

    if nbiter==1:
        print('Creating slopes for Cosinus, PSFOffAxis and new probes...')
        #Reference slopes
        refslope='VisAcq.DET1.REFSLP'
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
        refslope='iter'+str(nbiter-1)+'correction'
        recordnewprobes(size_probes,posprobes,dir+filenameroot,refslope,nbiter)
    else:
        #Calculate the center of the first coronagraphic image using the waffle
        if nbiter==2:
            print('Calculating center of the first coronagraphic image:')
            centerx,centery=findingcenterwithcosinus(dir+filenameroot)
            SaveFits([centerx,centery],['',0],dir+filenameroot,'centerxy')
        
        centerx,centery=LoadImageFits(dir+filenameroot+'centerxy.fits')
        #Estimation of the electric field using the pair-wise probing (return the electric field and the slopes)
        print('Estimating the electric field using the pair-wise probing:')
        estimation,pentespourcorrection=resultEFC(dir,filenameroot,posprobes,nbiter,centerx,centery)
        if record==True:
            #Record the slopes to apply for correction at the next iteration
            refslope='iter'+str(nbiter-2)+'correction'
            recordslopes(pentespourcorrection,dir+filenameroot,refslope,'iter'+str(nbiter-1)+'correction')
            #Record the slopes to apply for probing at the next iteration
            refslope='iter'+str(nbiter-1)+'correction'
            recordnewprobes(size_probes,posprobes,dir+filenameroot,refslope,nbiter)
            
        #Propagate the estimation of electric field in the pupil plane and display the result using a nm scale
        estimationpad=np.pad(real(estimation),int((isz-400)/2),'constant',constant_values=(0,0))+1j*np.pad(im(estimation),int((isz-400)/2),'constant',constant_values=(0,0))
        estimationinpup=cropimage(im(shift(ifft(shift(estimationpad)))),int(isz/2),int(isz/2),384)
        SaveFits(cropimage(im(shift(ifft(shift(estimationpad)))),int(isz/2),int(isz/2),384),['',0],dir,'iter1phase')
        SaveFits(cropimage(real(shift(ifft(shift(estimationpad)))),int(isz/2),int(isz/2),384),['',0],dir,'iter1amp')
        #estimationinpup=estimationinpup*squaremaxPSF/2/np.pi*wave*1e9
        print('Done with recording new slopes!')
        
        
        #f1 = plt.figure(1, figsize=(10,5))
        #f1.clf()
        #ax1 = f1.add_subplot(121)
        #ax1.imshow(-estimationinpup, cmap="hot")
        #ax1.set_title('Im part of the PP Electric field')
        #refslope='iter'+str(nbiter-2)+'correction'
        #before=LoadImageFits(dir+filenameroot+refslope+'.fits')[0]
        #ax3 = f1.add_subplot(122)
        #ax3.plot(before)
        #ax3.plot(pentespourcorrection+1.5)
        #ax3.set_title('Slopes')
        #f1.tight_layout()
        #plt.show()
        #SHslopes2map(pentespourcorrection,visu=True)
        #plt.show()
    return 0


    
def VoltToSlope(Volt):
    Slopetopush=V2S@Volt
    return Slopetopush
    

    
def recordCoswithvolt(amptopushinnm,dir,refslope):
    nam=['cos_00deg','cos_30deg','cos_90deg']
    nbper=10.
    dim=240
    xx, yy = np.meshgrid(np.arange(240)-(240)/2, np.arange(240)-(240)/2)
    cc= np.cos(2*np.pi*(xx*np.cos(0*np.pi/180.))*nbper/dim)
    coe=(cc.flatten())@IMF_inv
    #print(np.amax(coe),np.amin(coe))
    coe=coe*amptopushinnm/rad_632_to_nm_opt#/37/rad_632_to_nm_opt
    #print(np.amax(coe),np.amin(coe))
    slopetopush=V2S@coe
    recordslopes(slopetopush,dir,refslope,nam[0]+'_'+str(amptopushinnm)+'nm')
    return 0
    
    
    
def findingcenterwithcosinus(dir):    
    
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
    cosinuspluscoro=sorted(glob.glob(dir+'CosinusForCentering*.fits'))[-1]
    coro=sorted(glob.glob(dir+'iter0_coro_image*.fits'))[-1]
    
    #Fit par les gaussiennes
    popt=np.zeros(8)
    data=LoadImageFits(cosinuspluscoro)[0]-LoadImageFits(coro)[0]
    data1=cropimage(data,x0_up,y0_up,30)
    data2=cropimage(data,x1_up,y1_up,30)
    data1[np.where(data1<0)]=0
    data2[np.where(data2<0)]=0
   
    w,h = data1.shape
    x, y = np.mgrid[0:w, 0:h]
    xy=(x,y)
                
    #Fit 2D Gaussian with fixed parameters for the top PSF
    initial_guess = (np.amax(data1), 1 , 1 , np.unravel_index(np.argmax(data1, axis=None), data1.shape)[0] , np.unravel_index(np.argmax(data1, axis=None), data1.shape)[1] ,np.mean(data1))
    
    try:
        popt1, pcov = opt.curve_fit(twoD_Gaussian, xy, (data1).flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed top PSF")

    #Fit 2D Gaussian with fixed parameters for the bottom PSF
    initial_guess = (np.amax(data2), 1 , 1 , np.unravel_index(np.argmax(data2, axis=None), data2.shape)[0] , np.unravel_index(np.argmax(data2, axis=None), data2.shape)[1] ,np.mean(data2))
    
    try:
        popt2, pcov = opt.curve_fit(twoD_Gaussian, xy, (data2).flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed bottom PSF")
        
    centerx=((popt1[3]+x0_up-15)+(popt2[3]+x1_up-15))/2
    centery=((popt1[4]+y0_up-15)+(popt2[4]+y1_up-15))/2
    
    print('- centerx = ',centery)
    print('- centery = ',centerx)
    print('!!!! ACTION: CHECK CENTERX AND CENTERY CORRESPOND TO THE CENTER OF THE CORO IMAGE')
    print('If not, change the guess in the MainEFCBash.sh file')

    fig1,ax1=plt.subplots(1,1)
    ax1.imshow(data,vmin=-5e2,vmax=5e2)
    plt.show()

    return centerx,centery
    


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
isz=int(pupsize*ld_p)#-1 # Nombre de pixels dans le plan pupille pour atteindre la résolution ld_p voulue



amplitude=8
dimimages=400

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

if onsky==0:
    lightsource='InternalPupil_'
else:
    lightsource='VLTPupil_'

if nbprobe == '2':
    posprobes = [678, 679]
if nbprobe == '3':
    posprobes = [678,679,680]
if nbprobe == '4':
    posprobes = [678,679,680,681]

vectoressai=LoadImageFits(MatrixDirectory+lightsource+'VecteurEstimation_'+nbprobe+'probes'+str(size_probes)+'nm.fits')
WhichInPupil=LoadImageFits(MatrixDirectory+'WhichInPupil0_5.fits')


#Dark hole size
#DHsize = 0 for half dark hole 188mas to 625mas x -625mas to 625mas
#DHsize = 1 for half dark hole 125mas to 625mas x -625mas to 625mas
#DHsize = 2 for full dark hole -625mas to 625mas x -625mas to 625mas

## corr_mode: choose the SVD cutoff of the EFC correction
# corr_mode=0: stable correction but moderate contrast
# corr_mode=1: less stable correction but better contrast
# corr_mode=2: more aggressive correction (may be unstable)


maskDH = LoadImageFits(MatrixDirectory+'mask_DH'+str(dhsize)+'.fits')
invertGDH=LoadImageFits(MatrixDirectory+lightsource+'Interactionmatrix_DH'+str(dhsize)+'_SVD'+str(corr_mode)+'.fits')

FullIterEFC(ImageDirectory,posprobes,nbiter,exp_name,record=True)

