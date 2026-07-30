#!/bin/bash
WORK_PATH=~/Documents/Recherche/DonneesTHD/EFConSPHERE/sphere_efc/SlopesAndImages
FILE_COS=( `/bin/ls ${WORK_PATH}/VisAcq.DET1.REFSLP.fits` )
for FILE in "${FILE_COS[@]}"
do
	echo " * loading ref slopes: ${FILE}"
	rsh wsrsgw cdmsLoad -f ${FILE} -r VisAcq.DET1.REFSLP 
	rsh wsrsgw "msgSend \"\" CommandGateway EXEC \"VisAcq.update ALL\""
	echo "Waiting 3s for the slopes to be loaded"
	/bin/sleep 3
	echo ' * slopes loaded'
done
