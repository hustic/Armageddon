import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

def sci_sol( velocity=20e3, density=3000, angle=45, z=100e3, distance=0, radius=10, fragmentation=False, radians=False):
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
    C_D = 1.
    C_H = 0.1
    Q = 1e7
    C_L = 1e-3
    R_p = 6371e3
    g = 9.81
    rho_0 = 1.2
    rho_m = 3000
    H = 8000
    Y = 1e15
    alpha = 0.3



    if radians is False:  # converts degrees to radians
        angle = angle * (np.pi) / 180


    def dmove(t,Point):
        velocity, density, angle, z, distance, radius = Point
        m = 4 / 3 * np.pi * radius ** 3 * density  # mass, asteroid to be assumed as spheric shape
        A = np.pi * (radius ** 2)  # cross-sectional area
        rho_a = rho_0 * np.exp(-z / H)
        # new_r = 0
        # if rho_a * (velocity**2) >= Y:
        #     new_r = velocity*((7/2*alpha*rho_a/rho_m)**(1/2))
        return np.array([(-C_D * rho_a * A * (velocity ** 2)) / (2 * m) + g * np.sin(angle),
                         (-C_H * rho_a * A * (velocity ** 3)) / (2 * Q),
                         (g * np.cos(angle)) / (velocity) - (C_L * rho_a * A * velocity) / (2 * m) - (velocity * np.cos(angle)) / (R_p + z),
                         -velocity * np.sin(angle),
                         (velocity * np.cos(angle)) / (1 + z / R_p),
                         0
                         ])
    tmax = 120
    t = np.arange(0, tmax, 0.05)
    P1 = solve_ivp(dmove, (0, tmax),(velocity, density, angle, z, distance, radius), method='RK45', t_eval=t)
    sci_result = P1['y']
    #print(P1['y'][3])
    #print(sci_result[3])



    dedz = np.zeros(len(sci_result[0]))
    ke = ((1 / 2 * sci_result[1, 1:] * sci_result[0, 1:] ** 2) - (1 / 2 * sci_result[1, :-1] * sci_result[0, :-1] ** 2)) / 4.184e12
    alt = (sci_result[3, 1:] - sci_result[3, :-1]) / 1e3
    dedz[1:] = ke / alt
    i = np.where(dedz < 0)
    dedz[i] = 0

    '''
    0: velocity
    1: mass
    2: angle
    3: altitude
    4: distance
    5: radius
    '''

    result = pd.DataFrame({'altitude':sci_result[3], 'velocity':sci_result[0], 'mass':sci_result[1], 'angle':sci_result[2], 'distance':sci_result[4],'radius':sci_result[5], 'dedz':dedz})
    #result = result.sort_values(by='altitude', ascending=False)

    return result
