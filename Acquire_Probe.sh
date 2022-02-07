#!/bin/bash

: '
This script should be run on the sparta gateway. It aims to acquire an individual probe image to reprocess EFC code if needed.
'

#Number of the current iteration
nbiter=2 #should be nbiter-1 with respect to MainEFCBash
which_probe=1 #from 1 to 4

## IRDIS parameters
#Image diversity
DIT_probe=1
NDIT_probe=1

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



	# Probe images
	FILES_probes=( `/bin/ls ${WORK_PATH}/${EXP_NAME}iter${nbiter}probe${which_probe}.fits` )
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
		msgSend -n wsre sroControl SETUP "-expoId 0 -file SPHERE_irdis_tec_exp.ref -function OCS1.DET1.READ.CURNAME Nondest  OCS1.DET1.SEQ1.DIT ${DIT_probe} OCS1.DET1.NDIT ${NDIT_probe} DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IMAGE OCS1.OCS.DET1.IMGNAME ${EXP_NAME}iter${nbiter}_Probe_000${which_probe}_ OCS1.DET1.FRAM1.STORE F OCS1.DET1.FRAM2.STORE T OCS1.DET1.ACQ1.QUEUE 0 OCS.DET1.IMGNAME SPHERE_IRDIS_OBS"
		msgSend -n wsre sroControl START "-detId IRDIS"
		msgSend -n wsre sroControl WAIT "-detId IRDIS"
	done


# copy all science files into working directory
echo "Copy science files"

# this line is necessary otherwise the cp of existing files fails because of permission problems
chmod uga+rwx ${WORK_PATH}/${EXP_NAME}*fits
scp sphere@wsre:${DATA_PATH}/${EXP_NAME}iter${nbiter}_Probe_000${which_probe}_*.fits ${WORK_PATH}/
#./send.sh

# end
echo "Done!"





















