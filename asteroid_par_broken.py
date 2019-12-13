import numpy as np
import numpy.linalg as sl
from armageddon.solver import Planet
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.interpolate as si
import time

random.seed(time.time())

def parse_data(input_radius,input_velocity,input_density,input_strength,input_angle,profile_name,input_radians,N,num):
    '''
    input_data: directory
        radius : float
            The radiu of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

    profile_name: str

        dedz : 

        altitude:

    num : str{'explicit euler',
            'implicit euler',
            'midpoint implicit euler',
            'runge kutta'}

    '''
    start_time = time.time()
    data = pd.read_csv(profile_name,header=None)
    
    height_input = np.array(pd.to_numeric(data.iloc[1:,0]))*1e3
    dedz_input = np.array(pd.to_numeric(data.iloc[1:,1]))
    lp = np.polyfit(height_input, dedz_input,15)
    f = np.poly1d(lp)
    radius = np.zeros(N+1)
    velocity = np.zeros(N+1)
    density = np.zeros(N+1)
    strength = np.zeros(N+1)
    angle = np.zeros(N+1)
    error = np.zeros(N+1)
    find_altitude = []
    find_dedz = []
    start_loop_time = time.time()
    #print('before simultation passed:',start_loop_time-start_time)
    P = Planet()
    for i in range(N+1):
        if input_radius ==None:
            radius[i] = random.randrange(5,50,1)
        else:
            radius[i] = input_radius
        if input_velocity==None:
            velocity[i] = random.randrange(1e3,1e5,1)
        else:    
            velocity[i] =input_velocity
        if input_density == None:
            density[i] = random.randrange(1,7000,1)
        else:
            density[i] = input_density
        if input_strength == None:
            strength[i] = random.randrange(1000,1e7,1)
        else:
            strength[i] = input_strength
        if input_angle == None:
            angle[i] = random.randrange(0,90,1)
        else:
            angle[i] = input_angle

        result, out = P.impact(radius[i], velocity[i], density[i], strength[i], angle[i],
               init_altitude=100e3, dt=0.05, radians=input_radians,
               fragmentation=True, num_scheme=num)
        dedz = np.array(result.dedz[:])
        alts= np.array(result.altitude[:])
        test_altitude = alts[np.where((alts<=max(height_input))&(alts>=min(height_input)))]
        test_dedz = dedz[np.where((alts<=max(height_input))&(alts>=min(height_input)))]
        compare_dedz = f(test_altitude)
        error[i] = sl.norm(test_dedz-compare_dedz)/np.sqrt(len(test_dedz))
        find_altitude.append(test_altitude)
        find_dedz.append(test_dedz)
    print('loops for:',time.time()-start_loop_time)

    loc = np.argmin(error)
    outcome = {'radius':radius[loc],'velocity':velocity[loc],'density':density[loc],
                  'strength':strength[loc],'angle':angle[loc]}

    best_fit_dedz = find_dedz[loc]#fit_dedz2
    best_fit_alt = find_altitude[loc]

    fig = plt.figure(figsize=(8, 6))
    plt.title('Event data over Simulation', fontsize =14)
    plt.plot(dedz_input, height_input,'bo', label='Event data')
    plt.plot(best_fit_dedz, best_fit_alt, 'r-', label='Simulation data')
    plt.legend(loc = 'upper right')
    plt.grid()

    plt.show()
    

    return outcome
