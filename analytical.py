import numpy as np
import matplotlib.pyplot as plt

def anal_sol(self, radius, velocity, density, strength, angle,
               init_altitude=100e3, dt=0.05, radians=False):
    # define constants
    Cd = 1 # drag coefficient
    H = 8000 # atomspheric consatnt
    rho = 3000 # asteroid density

    # define initial conditions
    velocity = 20e3 # velocity
    radius = 10 # radius
    m = 4/3 * np.pi * radius**3 * rho # mass, asteroid to be assumed as spheric shape
    angle = 45 # angle
    init_altitude = 100000 # initial height
    A = np.pi * radius**2 # cross-sectional area

    if radians is False: # converts degrees to radians
        angle = angle * (np.pi)/180

    # define atomspheric density
    vv = np.exp((-Cd * A * 1.2 * H / (2 * m * np.sin(angle))) * np.exp(-init_altitude/H))
    c = velocity / vv # vv is a substitution constant

    def v_h(h): 
        return c * np.exp((-Cd * A * 1.2 * H / (2 * m * np.sin(angle))) * np.exp(-h/H))

    C2 = -Cd * A * 1.2 * H / (2 * m * np.sin(angle))

    def dEdz(z):
        return c * np.exp(C2 * np.exp(-z/H)) * C2 * np.exp(-z/H) * (-1/H) * m * v_h(z)

    H_plot = np.linspace(0, 100000, 200)
    dh = 500
    E = 0.5 * m * v_h(H_plot)**2
    de = (E[1:] - E[0:-1]) / dh

    

    #if radians is False:
     #       Y[:, 2] = list(map(lambda x: x * 180/np.pi, Y[:, 2]))
