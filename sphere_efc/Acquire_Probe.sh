#!/bin/bash

: '
This script should be run on the sparta gateway. It aims to acquire an individual probe image to reprocess EFC code if needed.
'

#Number of the current iteration
nbiter=2 #should be nbiter-1 with respect to MainEFCBash
which_probe=1 #from 1 to 4

## IRDIS parameters
#Image diversity
DIT_IRDIS_probe=0
NDIT_IRDIS_probe=1
DIT_IFS_probe=1
NDIT_IFS_probe=1

#Which instrument is used
detector="IFS_OBS_YJ" #Can be IRDIS_H3 or IFS_OBS_YJ or IFS_OBS_H

# If detector = "IFS_OBS_YJ" or "IFS_OBS_H"
# If IFS_only = "True": IRDIS_DIT = 0 and IRDIS_NDIT=1
# IF IFS_only = "False": the parameters set by the user are used for the IRDIS DIT and NDIT
IFS_only = "True"


# Path common to wsre and wsrsgw
DATA_PATH=/data/SPHERE/INS_ROOT/SYSTEM/DETDATA
#WORK_PATH0=/vltuser/sphere/jmilli/test_EFC_20190830/PackageEFConSPHERE/
WORK_PATH0=/vltuser/sphere/zwahhaj/efc
#WORK_PATH0=~/Documents/Research/SPHERE/Git_Software/sphere_efc
#WORK_PATH0=~/Documents/Recherche/DonneesTHD/EFConSPHERE/sphere_efc


###################################################################
###################################################################
################# NO CHANGE BELOW
###################################################################
###################################################################

MATRIX_PATH=$WORK_PATH0'/MatricesAndModel'
WORK_PATH=$WORK_PATH0'/SlopesAndImages'

SX=1
SY=1220
N=300

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
		rsh wsrsgw "msgSend \"\" CommandGateway EXEC \"VisAcq.update ALL\" "

		echo "Waiting 3s for the slopes to be loaded"
		/bin/sleep 3
		echo ' * slopes loaded'

		echo "Acquire Probe"
		echo ' * acquiring image'
			if [[ "$detector" == "IRDIS_H3" ]]; then
				ssh wsre "msgSend -n wsre sroControl SETUP \"-expoId 0 -file SPHERE_irdis_tec_exp.ref -function OCS1.DET1.READ.CURNAME Nondest  OCS1.DET1.SEQ1.DIT ${DIT_IRDIS_probe} OCS1.DET1.NDIT ${NDIT_IRDIS_probe} DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IMAGE OCS1.OCS.DET1.IMGNAME ${EXP_NAME}iter${nbiter}_Probe_000${which_probe}_ OCS1.DET1.FRAM1.STORE F OCS1.DET1.FRAM2.STORE T OCS1.DET1.ACQ1.QUEUE 0 OCS.DET1.IMGNAME SPHERE_IRDIS_OBS OCS1.DET1.SEQ1.WIN.STRX ${SX} OCS1.DET1.SEQ1.WIN.STRY ${SY} OCS1.DET1.SEQ1.WIN.NX 2048 OCS1.DET1.SEQ1.WIN.NY ${N}\" "
				ssh wsre "msgSend -n wsre sroControl START \"-detId IRDIS\" "
				ssh wsre "msgSend -n wsre sroControl WAIT \"-detId IRDIS\" "

			elif [[ "$detector" == "IFS_OBS_YJ" || "$detector" == "IFS_OBS_H" ]]; then
				ssh wsre "msgSend -n wsre sroControl SETUP \"-expoId 0 -function OCS2.DET1.SEQ1.DIT ${DIT_IFS_probe} OCS2.DET1.NDIT ${NDIT_IFS_probe} OCS2.DET1.FRAM1.BREAK ${NDIT_IFS_probe} OCS2.DET1.FRAM2.BREAK 0  OCS2.DET1.ACQ1.QUEUE ${NDIT_IFS_probe} OCS2.DET1.READ.CURNAME Nondest OCS1.DET1.SEQ1.DIT ${DIT_IRDIS_probe} OCS1.DET1.NDIT ${NDIT_IRDIS_probe} OCS1.DET1.ACQ1.QUEUE ${NDIT_IRDIS_probe} OCS1.DET1.READ.CURNAME Nondest -file SPHERE_irdifs_obs.ref \" "
				
				# Record IFS data
				ssh wsre "msgSend -n wsre sroControl SETUP \"-expoId 0 -function INS.MODE IRDIFS DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IFU OCS2.INS.DITH.POSX 0 OCS2.INS.DITH.POSY 0 OCS2.OCS.DET1.IMGNAME ${EXP_NAME}IFS_iter${nbiter}_Probe_000${which_probe}_ \" "
				ssh wsre "msgSend -n wsre sroControl START \"-expoId 0 -detId IFS \" "

				if [[ "$IFS_only" == "False" ]]; then
					ssh wsre "msgSend -n wsre sroControl SETUP \"-expoId 0 -function INS.MODE IRDIFS DPR.CATG TEST DPR.TYPE OBJECT DPR.TECH IMAGE,DUAL OCS1.INS.DITH.POSX 0 OCS1.INS.DITH.POSY 0  OCS1.OCS.DET1.IMGNAME ${EXP_NAME}IRDIS_iter${nbiter}_Probe_000${which_probe}_ \" "
					ssh wsre "msgSend -n wsre sroControl START \"-expoId 0 -detId IRDIS \""
				fi
				# line using the solo file
				# ssh wsre "msgSend -n wsre sroControl SETUP \"-expoId 0 -file SPHERE_gen_obs_solo_ifs.ref -function OCS2.DET1.READ.CURNAME Nondest OCS2.DET1.SEQ1.DIT ${DIT_IFS_probe} OCS2.DET1.NDIT ${NDIT_IFS_probe} OCS2.OCS.DET1.IMGNAME ${EXP_NAME}IFS_iter${nbiter}_Probe_000${k}_ \" "
 				# ssh wsre "msgSend -n wsre sroControl START \"-detId IFS\" "
				ssh wsre "msgSend -n wsre sroControl WAIT \"-detId IFS\" "

			fi
	done


# copy all science files into working directory
echo "Copy science files"

# this line is necessary otherwise the cp of existing files fails because of permission problems
chmod uga+rwx ${WORK_PATH}/${EXP_NAME}*fits
scp sphere@wsre:${DATA_PATH}/${EXP_NAME}iter${nbiter}_Probe_000${which_probe}_*.fits ${WORK_PATH}/
#./send.sh

# end
echo "Done!"





















