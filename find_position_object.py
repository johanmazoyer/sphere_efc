

"""
Created on  March 4 2022
code to fin the position of an object to dig the dark hole at the good position with SPHERE
@author: J Mazoyer

# test beta-pictoris irdis dataset 2015-02-05
# timestring = '2015-02-05T00:24:51.25' # start time in cube
# timestring = '2015-02-05T03:55:24.57' # end time in cube
# and ~3 hours of observations checks out with the cube where we see the planet by eye


requires atroplan
conda install -c conda-forge astroplan
don't forget to cite https://ui.adsabs.harvard.edu/abs/2018AJ....155..128M/abstract
"""


import numpy as np 

import matplotlib.pyplot as plt

from astropy.time import Time
import astropy.units as u
from astroplan import Observer, FixedTarget

# from astropy.coordinates import EarthLocation
# print(EarthLocation.get_site_names())


def PA_on_detector(target_name, time_now, estimated_onsky_PA_of_the_planet, verbose = False):
    
    """ --------------------------------------------------
    Measure the planet position in SPHERE detector at the given time

    Author: J Mazoyer
    
    Parameters:
    ----------
    target_name: string
        name of the object

    time_now: Time object (astropy)
        Time of observation

    estimated_onsky_PA_of_the_planet: float degrees
        Estimated position angle of the planet in degrees
        
    verbose : bool
        if True print values and intermediate

    Return:
    
    Position angle of the planet

    -------------------------------------------------- """

    vlt = Observer.at_site('Paranal Observatory', timezone="UTC")
    PUPOFFSET = 135.87 # ±0.03 deg 
    True_North = -1.75 #+/-0.08 deg (https://arxiv.org/pdf/1609.06681.pdf)


    target = FixedTarget.from_name(target_name)

    PARANGLE_deg = vlt.parallactic_angle(time_now, target).to_value(u.deg)

    # the values actually written in the rot sphere files are
    # -PARANGLE_deg - PUPOFFSET - True_North

    ### zeformula given by Gael
    ### PA_onsky = PA_detector + PARANGLE_deg + True_North + PUPOFFSET 

    PA_detector = estimated_onsky_PA_of_the_planet - PARANGLE_deg - True_North - PUPOFFSET 

    if verbose:
        print("")
        print("")
        print("target: ",target_name)
        print("planet estimated Postion angle is:", estimated_onsky_PA_of_the_planet )
        print("at observation time: ",time_now)
        # print("parralactic angle given in SPHERE reduc parang files (to check): ", round(-PARANGLE_deg - PUPOFFSET - True_North,4))
        print("PA on detector angle is: ", round(PA_detector,2))
        print("")
        print("")
    return PA_detector




def plot_pos_planet(target_name, time_now, estimated_onsky_PA_of_the_planet,estimated_sep_of_the_planet, time_in_hour = 1):
    """ --------------------------------------------------
    Plot the planet position in SPHERE detector in the next hour of the given time

    Author: J Mazoyer
    
    Parameters:
    ----------
    target_name: string
        name of the object

    time_now: Time object (astropy)
        Time at the begining of observation

    estimated_onsky_PA_of_the_planet: float degrees
        Estimated position angle of the planet in degrees
        
    estimated_sep_of_the_planet: float mas
        Estimated separation of the planet in mas

    Return:

    NA
    -------------------------------------------------- """
    
    sphere_plate_scale = 12.25 #mas.pix−1

    t1 = Time('2010-01-01 00:00:00')
    t2 = Time('2010-01-01 00:10:00')
    dt10minutes = t2 - t1  # Difference between two Times

    # print(dt10minutes.sec)

    total_timein10min = np.ceil(time_in_hour*60/10)

    time_plots = time_now + dt10minutes * np.arange(0,total_timein10min + 1)


    radiuses_plot = np.zeros(len(time_plots)) + estimated_sep_of_the_planet/sphere_plate_scale
    positions_angles_plot = np.zeros(len(time_plots)) 

    for i, timehere in enumerate(time_plots):
        positions_angles_plot[i] = np.radians(PA_on_detector(target_name, time_plots[i], estimated_onsky_PA_of_the_planet, verbose = False))
    
    print("")
    print("")
    print("at ", time_plots[0])
    print("angle on detector is ", np.degrees(positions_angles_plot[0] ) )

    print("at ", time_plots[-1])
    print("angle on detector is ",  np.degrees(positions_angles_plot[-1] ) )
    print("")
    print("")

    colors = positions_angles_plot


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")
    ax.plot(positions_angles_plot, radiuses_plot)
    c = ax.scatter(positions_angles_plot[0], radiuses_plot[0], c='red', alpha=0.75)

    ax.set_rmax(60)
    ax.set_rticks([10, 20, 30, 40, 50, 60])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    plt.text(np.radians(225), 90, target_name, fontsize = 12,ha ='center')
    plt.text(np.radians(135), 90,  time_plots[0], fontsize = 12,ha ='center',c='red')


    ax.set_title("Position planet in next {0} hour(s) (radius in pixel, dot is now)".format(time_in_hour), va='bottom')
    # plt.show()
    plt.savefig("/Users/jmazoyer/Desktop/graphe_pos_planets/" +f"{target_name}_start{timestring}_duration{time_in_hour}h.pdf" )



target_name = "HD 4113 C"
timestring = '2022-10-06T04:00:00' # start time in cube

#beta pic b 
# estimated_onsky_PA_of_the_planet = 212.58 # degree https://doi.org/10.1051/0004-6361/201834302
# estimated_sep_of_the_planet = 332.42 # mas https://doi.org/10.1051/0004-6361/201834302

# HR 87 99 d JAson site http://whereistheplanet.com/
# estimated_onsky_PA_of_the_planet = 238 # degree
# estimated_sep_of_the_planet = 695.8 # mas 

# HR 87 99 e JAson site http://whereistheplanet.com/
# estimated_onsky_PA_of_the_planet = 319.6 # degree
# estimated_sep_of_the_planet = 398 # mas 


# HD 984b / HIP 1134 JAson site http://whereistheplanet.com/
# estimated_onsky_PA_of_the_planet = 39.5 # degree
# estimated_sep_of_the_planet = 253 # mas 

# HD 4113 C
# https://www.aanda.org/articles/aa/pdf/2018/06/aa30136-16.pdf
estimated_onsky_PA_of_the_planet = 42 # degree
estimated_sep_of_the_planet = 530 # mas 


time_now = Time(timestring, format='isot', scale='utc')
PA_on_detector(target_name, time_now, estimated_onsky_PA_of_the_planet, verbose = True)
plot_pos_planet(target_name, time_now, estimated_onsky_PA_of_the_planet,estimated_sep_of_the_planet, time_in_hour = 1)


# just to check in betapic cube
# posplanet = [488,504] # position of seen planet in the cube in pixel at start
# posplanet = [505,538] # position of seen planet in the cube in pixel at end


# posstar = [512,512] # pixel

# measured_angle_to_north = np.degrees(np.arctan2(posplanet[1] - posstar[1],posplanet[0] - posstar[0])) - 90
# print("measure on image", measured_angle_to_north)

# print("")
# print("")
# print("")