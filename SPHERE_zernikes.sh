#!/bin/bash

declare -a alpha=()
alpha+=(5 10)
alpha+=(7 10)

SLOPE_INI='VisAcq.DET1.REFSLP'

WORK_PATH0=/vltuser/sphere/zwahhaj/efc
WORK_PATH0=~/Documents/Research/SPHERE/Git_Software/sphere_efc
MATRIX_PATH=$WORK_PATH0'/MatricesAndModel/'
WORK_PATH=$WORK_PATH0'/SlopesAndImages/'


# Convert the array to a string
my_array_str=$(printf '%s\n' "${alpha[@]}")

# Export the array as an environment variable
export MY_ARRAY="$my_array_str"
export WORK_PATH
export MATRIX_PATH
export SLOPE_INI


echo "Run python Zernike script"
python SPHERE_zernikes.py #Should be python3 on SPHERE

read -n1 -r key

echo "Introduce zernikes on the DM"
FILE_COS=( `/bin/ls ${WORK_PATH}/Zernikes.fits` )
		for FILE in "${FILE_COS[@]}"
		do
			echo " * loading ref slopes: ${FILE}"
			rsh wsrsgw cdmsLoad -f ${FILE} -r VisAcq.DET1.REFSLP 
			rsh wsrsgw "msgSend \"\" CommandGateway EXEC \"VisAcq.update ALL\""

			echo "Waiting ${PAUSE_TIME}s for the slopes to be loaded"
			/bin/sleep 2
			echo ' * slopes loaded'

		done

