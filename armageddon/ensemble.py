import numpy as np
import pandas as pd
from scipy.special import erf
import dask

from .solver import Planet as planet

__all__ = ['solve_ensemble']

def solve_ensemble(
        planet,
        fiducial_impact,
        variables,
        radians=False,
        rmin=8, rmax=12,
        N=int(200)):
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

    N : int, optional
        Number of times to sample input parameters and, correspondingly, number
        of times the simulation is run.

    Returns
    -------

    ensemble : DataFrame
        DataFrame with columns of any parameters that are varied and the
        airburst altitude
    """

    nval = int(11) # Number of input values to sample from

    # Initialize parameter arrays with fiducial values for variables not varied
    radii = np.full((N,),fiducial_impact['radius'])
    angles = np.full((N,),fiducial_impact['angle'])
    strengths = np.full((N,),fiducial_impact['strength'])
    velocities = np.full((N,),fiducial_impact['velocity'])
    densities = np.full((N,),fiducial_impact['density'])

    # Dummy values to pass pytest while solver.py not implemented
    burst_altitudes = np.zeros((N,))

    # Initialize list to add varied variables and burst altitudes to
    data = []

    # Random sampling given values and respective probabilities
    for var in variables:
        if var == 'radius':
            # p(x=X) = 1/4, this is a uniform distribution
            radii = np.random.uniform(rmin, rmax, N) # uniform random sampling
            data.append(radii)

        # For non-uniform probability distributions, use np.random.choice
        if var == 'angle':
            # p(x=X) = d (1-cos^2(X)) / dX = 2 sin(X)cos(X)
            # range of possible input values
            theta = np.linspace(0,90,nval)
            # calculate probabilities corresponding to every possible input value
            theta_dist = 2*np.sin(np.radians(theta))*np.cos(np.radians(theta))
            theta_dist = theta_dist/np.sum(theta_dist) # normalize it to add up to 1
            angles = np.random.choice(theta, size=N, p=theta_dist) # sampling
            data.append(angles)

        # Repeat process for other input parameters
        if var == 'strength':
            # p(x=X) = 1/(x*log(10000)), assume log10
            str = np.linspace(1e3,1e7,nval)
            s_dist = 1/(str*4)
            s_dist = s_dist/np.sum(s_dist)
            strengths = np.random.choice(str, size=N, p=s_dist)
            data.append(strengths)

        if var == 'velocity':
            # p(x=X) = (sqrt(2/pi)*exp(-x**2/242)*x**2)/1331
            v = np.linspace(0,50000,nval) # At infinite distance, in m/s
            vi = np.sqrt(11200**2 + v**2) # Impact velocity, in m/s
            v_dist = (np.sqrt(2/np.pi)*np.exp(-(v/1000)**2/242)*(v/1000)**2)/1331
            v_dist = v_dist/np.sum(v_dist)
            velocities = np.random.choice(vi, size=N, p=v_dist)
            data.append(velocities)

        if var == 'density':
            # p(x=X) = exp(-(x-3e3)**2/2e6)/(1000*sqrt(2*pi))
            rho = np.linspace(1,7001,nval)
            rho_dist = np.exp(-(rho-3e3)**2/2e6)/(1000*np.sqrt(2*np.pi))
            rho_dist = rho_dist/np.sum(rho_dist)
            densities = np.random.choice(rho,size=N, p=rho_dist)
            data.append(densities)

    # Create array of input parameters
    params = np.array([radii, angles, strengths, velocities, densities])

    # Run parallelized simulation
    dask.config.set(scheduler='processes')
    lazies = [dask.delayed(planet.solve_atmospheric_entry)(*x, num_scheme='EE') for x in params.T]
    results = [dask.delayed(planet.calculate_energy)(lazy) for lazy in lazies]
    outcomes = [dask.delayed(planet.analyse_outcome)(result) for result in results]
    outcomes = dask.compute(*outcomes)

    bursts = [o['burst_altitude'] if 'burst_altitude' in o else 0 for o in outcomes]

    # Append burst altitude results to final output data
    data.append(bursts)
    # Convert to array and return in pandas DataFrame
    data = np.array(data)
    return pd.DataFrame(data.T, columns=variables+['burst_altitude'])

def plot_ensemble(ensemble):
    """
    Generate histogram plots for input parameters and burst altitude

    Parameters
    ----------

    ensemble : DataFrame
        pandas DataFrame with specified varied parameters and simulated burst
        altitudes. 

    Returns
    -------

    Figure 1 : plot
        matplotlib plot of histograms of input parameters and burst altitudes.
    """
    fig = plt.figure(figsize=(12, 8))
    fig.tight_layout()
    ax1 = plt.subplot(321)
    ax2 = plt.subplot(322)
    ax3 = plt.subplot(323)
    ax4 = plt.subplot(324)
    ax5 = plt.subplot(325)
    ax6 = plt.subplot(326)

    burst_altitude = np.array(ensemble['burst_altitude'])
    
    plt.show()