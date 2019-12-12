import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def anal_sol(H_plot, radius=10, velocity=20e3, density=3000, strength=10e5, angle=45,
               init_altitude=100e3, radians=False):
    '''
    Solves analytical solution for meteroid impact

    Parameters
    ----------

    radius : float
        The radius of the asteroid in meters

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

    radians : logical, optional
        Whether angles should be given in degrees or radians. Default=False
        Angles returned in the DataFrame will have the same units as the
        input
    

    Returns
    -------
    Result : DataFrame
            pandas dataFrame with collumns:
            altitude, velocity, dedz

    '''
    # define constants
    Cd = 1 # drag coefficient
    H = 8000 # atomspheric consatnt
    rho = 1.2 # air density at the ground

    # define initial conditions

    m = 4/3 * np.pi * radius**3 * density # mass, asteroid to be assumed as spheric shape
    A = np.pi * radius**2 # cross-sectional area

    if radians is False: # converts degrees to radians
        angle = angle * (np.pi)/180
    
    # constant in analytical solution
    c = velocity/(np.exp((-Cd * A * rho * H / (2 * m * np.sin(angle))) * np.exp(-init_altitude/H)))

    def v_h(h):
        return c * np.exp((-Cd * A * rho * H / (2 * m * np.sin(angle))) * np.exp(-h/H))

    C2 = -Cd * A * rho * H / (2 * m * np.sin(angle))
    
    def dEdz(z):
        return c * np.exp(C2 * np.exp(-z/H)) * C2 * np.exp(-z/H) * (-1/H) * m * v_h(z)

    #H_plot = np.linspace(100000, 0, 200)
    v_plot = v_h(H_plot)

    dedz = np.zeros((len(v_plot),)) # create array to store dedz results
    dedz[0] = 0 # initial dedz
    for i in range(1,len(v_plot)): # loop through all rows of result
            energy = ((1/2 * m * v_plot[i]**2) - (1/2 * m * v_plot[i-1]**2))/4.184e12
            alt = (H_plot[i] - H_plot[i-1])/1e3
            dedz[i] = energy / alt
    #dEdz_plot = dedz(H_plot)

    result = pd.DataFrame({'altitude':H_plot, 'velocity':v_plot, 'dedz':dedz})
    #result = result.sort_values(by='altitude', ascending=False)

    return result
