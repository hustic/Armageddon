import numpy as np
import pandas as pd
from scipy.special import erf

#from .solver import *

def solve_ensemble(
        planet,
        fiducial_impact,
        variables,
        radians=False,
        rmin=8, rmax=12,
        ):
    """
    Run asteroid simulation for a distribution of initial conditions and
    find the burst distribution

    Parameters
    ----------

    planet : object
        The Planet class instance on which to perform the ensemble calculation

    fiducial_impact : dict
        Dictionary of the fiducial values of radius, angle, strength, velocity
        and density

    variables : list
        List of strings of all impact parameters to be varied in the ensemble
        calculation

    rmin : float, optional
        Minimum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    rmax : float, optional
        Maximum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    Returns
    -------

    ensemble : DataFrame
        DataFrame with columns of any parameters that are varied and the
        airburst altitude
    """

    nval = int(20) # Number of values to sample from
    N = int(1e3)  # Choose 1000 samples

    # Initialize parameter arrays with fiducial values for variables not varied
    radii = np.full((N,),fiducial_impact['radius'])
    angles_f = np.full((N,),fiducial_impact['angle'])
    strengths_f = np.full((N,),fiducial_impact['strength'])
    velocities_f = np.full((N,),fiducial_impact['velocity'])
    densities_f = np.full((N,),fiducial_impact['density'])

    # Dummy values to pass pytest while solver.py not implemented
    burst_altitudes = np.zeros((N,))

    # Initialize final data array to add varied variables to
    data = np.zeros((N,))

    # Random sampling given values and respective probabilities
    for var in variables:
        if var == 'radius':
            # p(x=X) = 1/4
            r = np.linspace(rmin,rmax,nval)
            r_dist = np.full(r.shape,0.25)
            radii = np.random.choice(r, size=N, p=r_dist)
            data = np.vstack((data, radii))
        if var == 'angle':
            # p(x=X) = d (1-cos^2(X)) / dX = 2 sin(X)cos(X)
            theta = np.linspace(0,90,nval)
            theta_dist = 2*np.sin(theta)*np.cos(theta)
            angles = np.random.choice(theta, size=N, p=theta_dist)
            data = np.vstack((data, angles))
        if var == 'strength':
            # p(x=X) = 1/(x*log(10000)), assume log10
            str = np.linspace(1e3,1e7,nval)
            # s_CPD = np.log(str/smin) / np.log(smax/smin)
            s_dist = 1/(str*4)
            strengths = np.random.choice(str, size=N, p=s_dist)
            data = np.vstack((data, strengths))
        if var == 'velocity':
            # p(x=X) = (sqrt(2/pi)*exp(-x**2/242)*x**2)/1331
            v = np.linspace(0,50,nval) # At infinite distance
            vi = np.sqrt(11.2**2 + v**2) # Impact velocity
            a = 1.1e4
            v_dist = (np.sqrt(2/np.pi)*np.exp(-v**2/242)*v**2)/1331
            velocities = np.random.choice(vi, size=N, p=v_dist)
            data = np.vstack((data, velocities))
            # v_CPD = erf(v/(a*np.sqrt(2))) \
                      # - (v/a)*np.exp(-(v**2)/(2*a**2))*np.sqrt(2/np.pi)
        if var == 'density':
            # p(x=X) = exp(-(x-3e3)**2/2e6)/(1000*sqrt(2*pi))
            rho = np.linspace(0,7000,nval)
            rho_dist = np.exp(-(rho-3e3)**2/2e6)/(1000*np.sqrt(2*np.pi))
            densities = np.random.choice(rho,size=N, p=rho_dist)
            data = np.vstack((data, densities))
            #rho_CPD = 0.5*(1+erf((rho-3e3)/(1e3*np.sqrt(2))))

    # Run the simulation with the above arrays of parameters
    # Makes an ndarray of result dataframes and an ndarray of outcome dicts

    """Will uncomment the code below after solver.py fully implemented"""
    # simulation = np.vectorize(planet.impact)

    # results, outcomes = simulation(radius=radii,angle=angles,
                                   # strength=strengths,
                                   # velocity=velocities,density=densities)

    # # Convert array of dicts into pandas DataFrame
    # outcomes = pd.DataFrame(outcomes)

    # # Extract 'burst_altitude' column, cases with no airburst will have NaN
    # burst_altitudes = np.array(outcomes['burst_altitude'])

    data = np.vstack((data, burst_altitudes))
    data = data[1:,:]

    return pd.DataFrame(data.T, columns=variables+['burst_altitude'])

# def plot_histogram(planet, fiducial_impact, variables): 