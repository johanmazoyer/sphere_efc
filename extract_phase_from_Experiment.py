import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import Definitions_for_matrices as def_mat

WORK_PATH0='/home/rgalicher/projets_financements_proposals/projets/Sphere/2202_Sphere_EFC/All_data_from_efc_python/'

# mat1=fits.getdata(WORK_PATH0+'SAXO_DM_IFM_INV.fits')
#mat2=fits.getdata(WORK_PATH0+'CLMatrixOptimiser.HO_IM.fits')

#U, s, V = np.linalg.svd(mat1, full_matrices=False)
#S = np.diag(s)
#InvS=np.linalg.inv(S)
#pseudoinverse = np.dot(np.dot(np.transpose(V),InvS),np.transpose(U))
#fits.writeto(WORK_PATH0+'SAXO_DM_IFM_INV_inverse.fits', pseudoinverse)

#U, s, V = np.linalg.svd(mat2, full_matrices=False)
#S = np.diag(s)
#InvS=np.linalg.inv(S)
#pseudoinverse = np.dot(np.dot(np.transpose(V),InvS),np.transpose(U))
#fits.writeto(WORK_PATH0+'CLMatrixOptimiser.HO_IM_inverse.fits', pseudoinverse)

mat = fits.getdata(WORK_PATH0+'CLMatrixOptimiser.HO_IM_inverse.fits')*.4*632/2/np.pi
pushact = fits.getdata('/home/rgalicher/software/python/EFC_Sphere/sphere_efc/Model/PushActInPup384SecondWay.fits')

expname = 'Experiment0019'

a=fits.getdata(WORK_PATH0+expname+'_iter0correction.fits')[0]
v= np.matmul(mat,a)
tmp = v@pushact.reshape(1377,384*384)
ph0 = tmp.reshape(384,384)


ph=np.zeros((6,384,384))
for k in range(6):
    a=fits.getdata(WORK_PATH0+expname+'_iter'+str(int(k))+'correction.fits')[0]-fits.getdata(WORK_PATH0+expname+'_iter0correction.fits')[0]
    v= np.matmul(mat,a)
    tmp = v@pushact.reshape(1377,384*384)
    if k==0:
        ph[k] = ph0
    else:
        ph[k]=tmp.reshape(384,384)
    #plt.imshow(ph[k])
    #plt.show()

fits.writeto(WORK_PATH0+expname+'_phase_added_on_DM_vs_iteration.fits', ph,overwrite=True)

ph=np.zeros((4,384,384))
for k in range(4):
    a=fits.getdata(WORK_PATH0+expname+'_iter1probe'+str(int(k+1))+'.fits')[0]-fits.getdata(WORK_PATH0+expname+'_iter0correction.fits')[0]
    v= np.matmul(mat,a)
    tmp = v@pushact.reshape(1377,384*384)
    ph[k]=tmp.reshape(384,384)
    #plt.imshow(ph[k])
    #plt.show()

fits.writeto(WORK_PATH0+expname+'_phase_probe.fits', ph,overwrite=True)