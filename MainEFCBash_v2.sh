#!/bin/bash

#Version 2019/11/27

: '
This script should be run on the sparta gateway.
It will launch the EFC python code and record IRDIS data

Preliminary steps to perfornm before running this script

1/ In the folder called SlopesAndImages in WORK_PATH0: 
   a/ Save the Visible WFS ref slopes  (VisAcq.DET1.REFSLP.fits)

2/ In the folder called MatricesAndModel in WORK_PATH0:
   Save the interaction matrix (CLMatrixOptimiser.HO_IM.fit) from the CDMS browser

3/ Adjust the user parameters below to nbiter=1 at start (then increment), adjust (maybe) the path.
'

#Coronagraph that is used
coro = 'APLC'
#coro = 'FQPM'

#Number of the current iteration
#Each time nbiter=1, a new file rootname 'ExperimentXXXX'
#is automatically created
nbiter=2
# First try with nbiter= 1 to see if initialization runs
# then nbiter=2 to see at least 1 full loop 


#Do you want to save automatically an off-axis PSF and different backgrounds? Set 1 for yes, 0 for no.
create_bkgrd=0
create_PSF=0
create_coro=1


## IRDIS parameters

#coronagraphic image
DIT_image=1
NDIT_image=1

#Image diversity
DIT_probe=1
NDIT_probe=1

#Off-axis PSF
DIT_PSF=1
NDIT_PSF=1
WHICH_ND='ND_3.5' #can be 'ND_3.5' or 'ND_2.0' (to be checked for 2.0!)

#Background
DIT_bkgrd=1
NDIT_bkgrd=2

#Other images
DIT_cosinus=1
NDIT_cosinus=2


ONSKY=0 #Set 0 for internal pup ; 1 for an on sky correction
Assuming_VLT_PUP_for_corr=0 
#Work only if ONSKY=0. If  Assuming_VLT_PUP_for_corr=1 assume ONSKY=1 for EFC correction only

#Dark hole size : param namemask in CreateMatrixfromModelEFConSPHERE.py
#DHsize = 0 for half dark hole 188mas to 625mas x -625mas to 625mas
#DHsize = 1 for half dark hole 125mas to 625mas x -625mas to 625mas
#DHsize = 2 for full dark hole -625mas to 625mas x -625mas to 625mas
#DHsize = 3 for half dark hole -625mas to 625mas x 125mas to 625mas
DHsize=1

#Correction mode
# corr_mode=0: stable correction but moderate contrast
# corr_mode=1: less stable correction but better contrast
# corr_mode=2: more aggressive correction (may be unstable)
corr_mode=1

#Number of probing actuator
#nbprobe=2
zone_to_correct='horizontal'#vertical

#SizeProbes : can be 296, 400 or 500 (in nm)
size_probes=400

# First guess for the PSF echo position used for image centering (WARNING X and Y are inverted here)
# (adding a cosine to DM phase)
X0UP=548 #544 #Y position of the upper PSF echo in python #553
Y0UP=1511 #X position of the upper PSF echo in python
X1UP=478 #474 #Y position of the bottom PSF echo in python #485
Y1UP=1511 #X position of the bottom PSF echo in python

#Do you want to center your image at each iteration. Set 1 for yes, 0 for no.
centeringateachiter=0


# Path common to wsre and wsrsgw
DATA_PATH=/data/SPHERE/INS_ROOT/SYSTEM/DETDATA
#WORK_PATH0=/vltuser/sphere/jmilli/test_EFC_20190830/PackageEFConSPHERE/
WORK_PATH0=/vltuser/sphere/zwahhaj/efc/sphere_efc-main
#WORK_PATH0=~/Documents/Research/SPHERE/Git_Software/sphere_efc
#WORK_PATH0=~/Documents/Recherche/DonneesTHD/EFConSPHERE/sphere_efc


###################################################################
###################################################################
################# NO CHANGE BELOW
###################################################################
###################################################################

MATRIX_PATH=$WORK_PATH0'/MatricesAndModel'
WORK_PATH=$WORK_PATH0'/SlopesAndImages'


if [ "$ONSKY" -eq "1" ]; then
	echo "The NCPA compensation is done from on-sky measurements"
	lightsource_estim = 'VLTPupil_'
else
	echo "The NCPA compensation is done from the calibration source measurements"
	lightsource_estim = 'InternalPupil_'
fi

lightsource_estim = lightsource_estim + coro+'_'

if [ "$create_bkgrd" -eq "1" ]; then
#FOR BACKGROUND
    # close shutter
    echo "Close shutter (instrument shutter on sky or IRDIS shutter for internal lamp"
    if [ "$ONSKY" -eq "1" ]; then
	msgSend -n wsre sroControl SETUP "-function INS.SHUT.ST F"
    else
	msgSend -n wsre sroControl SETUP "-function OCS1.INS.OPTI1.NAME CLOSED"
    fi
    /bin/sleep 3
    echo "The shutter should be closed now (check)"
    #echo "Press enter to continue and take a background..."
    #read -n1 -r key

    # acquire background
    echo "Acquire background"
    msgSend -n wsre sroControl SETUP "-expoId 0 -file SPHERE_irdis_tec_exp.ref -function OCS1.DET1.READ.CURNAME Nondest  OCS1.DET1.SEQ1.DIT ${DIT_bkgrd} OCS1.DET1.NDIT ${NDIT_bkgrd} DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IMAGE OCS1.OCS.DET1.IMGNAME SPHERE_BKGRD_EFC_${DIT_bkgrd}s_ OCS1.DET1.FRAM1.STORE F OCS1.DET1.FRAM2.STORE T OCS1.DET1.ACQ1.QUEUE 0 OCS.DET1.IMGNAME SPHERE_IRDIS_OBS"
    msgSend -n wsre sroControl START "-detId IRDIS"
    msgSend -n wsre sroControl WAIT "-detId IRDIS"


    # open shutter
    echo "Open shutter (IRDIS shutter for internal test or instrument shutter on sky)"
    if [ "$ONSKY" -eq "1" ]; then
	msgSend -n wsre sroControl SETUP "-function INS.SHUT.ST T"
    else
	msgSend -n wsre sroControl SETUP "-function OCS1.INS.OPTI1.NAME ST_ALC2"
    fi
    /bin/sleep 3

    echo "The shutter should be opened now (check)"

fi

if [ "$create_PSF" -eq "1" ]; then

    # FOR OFF-AXIS PSF (!!!!!!!!!! ADD one ND !!!!!!)
    echo "Off-Axis PSF!"
    echo "Add "$WHICH_ND
    #echo "Press enter to continue and introduce the ND and take the off axis PSF"
    #read -n1 -r key
    
    msgSend -n wsre sroControl SETUP "-function INS.FILT2.NAME "$WHICH_ND
    echo "Waiting 3s for the ND to be loaded"
    /bin/sleep 3
    


    echo "!!!! ACTION: Modify the DTTS to create an off-axis PSF"
    echo "!!!! ACTION: Check the off axis PSF is not saturating with the current DIT. If not, press ENTER to acquire the Off-Axis PSF. Else, abort and change DIT_PSF"
    read -n1 -r key


    echo "Acquire OFF-Axis PSF"
    echo ' * acquiring image'
    msgSend -n wsre sroControl SETUP "-expoId 0 -file SPHERE_irdis_tec_exp.ref -function OCS1.DET1.READ.CURNAME Nondest  OCS1.DET1.SEQ1.DIT ${DIT_PSF} OCS1.DET1.NDIT ${NDIT_PSF} DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IMAGE OCS1.OCS.DET1.IMGNAME ${lightsource_estim}OffAxisPSF_ OCS1.DET1.FRAM1.STORE F OCS1.DET1.FRAM2.STORE T OCS1.DET1.ACQ1.QUEUE 0 OCS.DET1.IMGNAME SPHERE_IRDIS_OBS"
    msgSend -n wsre sroControl START "-detId IRDIS"
    msgSend -n wsre sroControl WAIT "-detId IRDIS"


    echo "!!!! ACTION: Bring the PSF back behind the corono by changing DTTS and press ENTER when well-centered"
    read -n1 -r key

    echo "Remove "$WHICH_ND
    msgSend -n wsre sroControl SETUP "-function INS.FILT2.NAME OPEN"
    echo "Waiting 3s for the ND to be removed"
    /bin/sleep 3
	
fi


if [ "$create_coro" -eq "1" ]; then
	#Find the name of the last experiment and change the name if nbiter==1
	#Check if at least one experiment was launched
	if [ -f "$WORK_PATH/Experiment0000_iter0correction.fits" ]
	then
		#tmpplus is used to increment the rootname if nbiter == 1
		if (($nbiter > 1))
		then
		tmpplus=1
		else
		tmpplus=0
		fi
		#Find the last file starting by Experiment in WORK_PATH
		TMP=$(ls ${WORK_PATH}/Experiment*iter0correction.fits|wc -l)
		let TMP=$TMP-$tmpplus
		#This number is the number of the next experiment
		EXP_NAME=Experiment$(printf "%04d" $TMP)'_'
	else
		# First Experiment
		EXP_NAME='Experiment0000_'
	fi 

	#Export the variables so that they can be retrived from python
	export WORK_PATH0
	export WORK_PATH
	export MATRIX_PATH
	export nbiter
	export EXP_NAME
	export DHsize
	export corr_mode
	export zone_to_correct
	export X0UP
	export Y0UP
	export X1UP
	export Y1UP
	export WHICH_ND
	export ONSKY
	export Assuming_VLT_PUP_for_corr
	export size_probes
	export centeringateachiter
	export coro

	#Launch the EFC code to prepare all the required files (slopes to apply on the DM and on DTTS)
	echo "Launch python EFC code"
	python3 PythonSphereEFC_v2.py #Should be python3 on SPHERE

	if (($nbiter == 2))
	then
		echo "!!!! ACTION: If bad centering, press Ctrl-C to stop the process and modify the guess. Else, press enter to continue and to acquire the images"
		read -n1 -r key

	fi

	echo "Launch image acquisition on SPHERE from the new slopes"
	# list all files
	FILES_WFS=( `/bin/ls ${WORK_PATH}/${EXP_NAME}iter*correction.fits` )
	#echo ${#FILES_WFS[@]}
	echo "List of input files:"
	for FILE in "${FILES_WFS[@]}"
	do
		echo " * ${FILE}"
	done


	if [ ${#FILES_WFS[@]} -eq "1" ]; then

		
		# FOR COSINUS TO CENTER

		echo "COSINUS!"
		
		FILE_COS=( `/bin/ls ${WORK_PATH}/cos_00deg_10nm.fits` )
		for FILE in "${FILE_COS[@]}"
		do
			echo " * loading ref slopes: ${FILE}"
			rsh wsrsgw cdmsLoad -f ${FILE} -r VisAcq.DET1.REFSLP 
			rsh wsrsgw "msgSend \"\" CommandGateway EXEC \"VisAcq.update ALL\""

			echo "Waiting 3s for the slopes to be loaded"
			/bin/sleep 3
			echo ' * slopes loaded'

			echo "Acquire Cosinus"
			echo ' * acquiring image'
			msgSend -n wsre sroControl SETUP "-expoId 0 -file SPHERE_irdis_tec_exp.ref -function OCS1.DET1.READ.CURNAME Nondest  OCS1.DET1.SEQ1.DIT ${DIT_cosinus} OCS1.DET1.NDIT ${NDIT_cosinus} DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IMAGE OCS1.OCS.DET1.IMGNAME ${EXP_NAME}CosinusForCentering_ OCS1.DET1.FRAM1.STORE F OCS1.DET1.FRAM2.STORE T OCS1.DET1.ACQ1.QUEUE 0 OCS.DET1.IMGNAME SPHERE_IRDIS_OBS"
			msgSend -n wsre sroControl START "-detId IRDIS"
			msgSend -n wsre sroControl WAIT "-detId IRDIS"

			/bin/sleep 2
		done
	fi

	# New coro image

	# wait for user confirmation
	echo "Loading the last correction slopes"
	#echo "Press enter to load the last correction slopes..."
	#read -n1 -r key

	# load the LAST reference slopes in the list
	FILE_WFS=${FILES_WFS[${#FILES_WFS[@]}-1]}
	echo " * loading ref slopes: ${FILE_WFS}"
	rsh wsrsgw cdmsLoad -f ${FILE_WFS} -r VisAcq.DET1.REFSLP 
	rsh wsrsgw "msgSend \"\" CommandGateway EXEC \"VisAcq.update ALL\""

	echo "Waiting 3s for the slopes to be loaded"
	/bin/sleep 3
	echo ' * slopes loaded'


	echo "Acquire coronagraphic image"
	echo ' * acquiring image'
	let imgnb=$nbiter-1
	#echo ${imgnb}
	msgSend -n wsre sroControl SETUP "-expoId 0 -file SPHERE_irdis_tec_exp.ref -function OCS1.DET1.READ.CURNAME Nondest  OCS1.DET1.SEQ1.DIT ${DIT_image} OCS1.DET1.NDIT ${NDIT_image} DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IMAGE OCS1.OCS.DET1.IMGNAME ${EXP_NAME}iter${imgnb}_coro_image_ OCS1.DET1.FRAM1.STORE F OCS1.DET1.FRAM2.STORE T OCS1.DET1.ACQ1.QUEUE 0 OCS.DET1.IMGNAME SPHERE_IRDIS_OBS"
	msgSend -n wsre sroControl START "-detId IRDIS"
	msgSend -n wsre sroControl WAIT "-detId IRDIS"


	#echo "Press enter to take the probe images..."
	#read -n1 -r key


	# Probe images
	k=1
	FILES_probes=( `/bin/ls ${WORK_PATH}/${EXP_NAME}iter${nbiter}probe*.fits` )
	for FILE in "${FILES_probes[@]}"
	do
		echo " * loading ref slopes: ${FILE}"
		rsh wsrsgw cdmsLoad -f ${FILE} -r VisAcq.DET1.REFSLP 
		rsh wsrsgw "msgSend \"\" CommandGateway EXEC \"VisAcq.update ALL\""

		echo "Waiting 3s for the slopes to be loaded"
		/bin/sleep 3
		echo ' * slopes loaded'

		echo "Acquire Probe"
		echo ' * acquiring image'
		msgSend -n wsre sroControl SETUP "-expoId 0 -file SPHERE_irdis_tec_exp.ref -function OCS1.DET1.READ.CURNAME Nondest  OCS1.DET1.SEQ1.DIT ${DIT_probe} OCS1.DET1.NDIT ${NDIT_probe} DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IMAGE OCS1.OCS.DET1.IMGNAME ${EXP_NAME}iter${nbiter}_Probe_000${k}_ OCS1.DET1.FRAM1.STORE F OCS1.DET1.FRAM2.STORE T OCS1.DET1.ACQ1.QUEUE 0 OCS.DET1.IMGNAME SPHERE_IRDIS_OBS"
		msgSend -n wsre sroControl START "-detId IRDIS"
		msgSend -n wsre sroControl WAIT "-detId IRDIS"
		let k=$k+1
	done
fi


# copy all science files into working directory
echo "Copy science files"

# this line is necessary otherwise the cp of existing files fails because of permission problems
chmod uga+rwx ${WORK_PATH}/${EXP_NAME}*fits
scp sphere@wsre:${DATA_PATH}/${EXP_NAME}iter[0-9]_coro_image_*.fits ${WORK_PATH}/
scp sphere@wsre:${DATA_PATH}/${EXP_NAME}iter[0-9]_Probe_000[0-9]_*.fits ${WORK_PATH}/
scp sphere@wsre:${DATA_PATH}/${EXP_NAME}CosinusForCentering_*.fits ${WORK_PATH}/
scp sphere@wsre:${DATA_PATH}/SPHERE_BKGRD_EFC_*.fits ${WORK_PATH}/
scp sphere@wsre:${DATA_PATH}/${lightsource_estim}OffAxisPSF_*.fits ${WORK_PATH}/
#./send.sh

# end
echo "Done!"





















