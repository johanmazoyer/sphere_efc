

"""
Created on  March 4 2022
code to fin the position of an object to dig the dark hole at the good position with SPHERE
@author: J Mazoyer

required atroplan
conda install -c conda-forge astroplan
don't forget to cite https://ui.adsabs.harvard.edu/abs/2018AJ....155..128M/abstract
"""

from astropy.time import Time
import astropy.units as u
from astroplan import Observer, FixedTarget

# from astropy.coordinates import EarthLocation
# print(EarthLocation.get_site_names())

vlt = Observer.at_site('Paranal Observatory', timezone="UTC")

PUPOFFSET = 135.87 # ±0.03 deg 
True_North = -1.75 #+/-0.08 deg (https://arxiv.org/pdf/1609.06681.pdf)


# test beta-pictoris irdis dataset 2015-02-05
target_name = "Beta Pictoris"
timestring = '2015-02-05T03:25:02' # start time
estimated_onsky_PA_of_the_planet = 212.58 # https://doi.org/10.1051/0004-6361/201834302

time = Time(timestring, format='isot', scale='utc')
target = FixedTarget.from_name(target_name)

PARANGLE_deg = vlt.parallactic_angle(time, target).to_value(u.deg)
print(PARANGLE_deg)

### zeformula given by Gael
### PA_onsky = PA_detector + PARANGLE_deg + True_North + PUPOFFSET 

PA_detector = estimated_onsky_PA_of_the_planet - PARANGLE_deg - True_North - PUPOFFSET 
print(PA_detector)