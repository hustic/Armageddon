import numpy as np
import pandas as pd
from scipy.special import erf
import dask
import dask.dataframe as dd

from .solver import Planet as planet

__all__ = ['solve_ensemble']

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

    nval = int(11) # Number of values to sample from
    N = int(100)  # Choose 500 samples for now

    # Initialize parameter arrays with fiducial values for variables not varied
    radii = np.full((N,),fiducial_impact['radius'])
    angles = np.full((N,),fiducial_impact['angle'])
    strengths = np.full((N,),fiducial_impact['strength'])
    velocities = np.full((N,),fiducial_impact['velocity'])
    densities = np.full((N,),fiducial_impact['density'])

    # Dummy values to pass pytest while solver.py not implemented
    burst_altitudes = np.zeros((N,))

    # Initialize final data array to add varied variables to
    data = np.zeros((N,))

    # Random sampling given values and respective probabilities

    for var in variables:
        if var == 'radius':
            # p(x=X) = 1/4, this is a uniform distribution
            radii = np.random.uniform(rmin, rmax, N)
            data = np.vstack((data, radii))
        if var == 'angle':
            # p(x=X) = d (1-cos^2(X)) / dX = 2 sin(X)cos(X)
            theta = np.linspace(0,90,nval)
            theta_dist = 2*np.sin(np.radians(theta))*np.cos(np.radians(theta))
            theta_dist = theta_dist/np.sum(theta_dist) # normalize it to add up to 1
            angles = np.random.choice(theta, size=N, p=theta_dist)
            data = np.vstack((data, angles))
        if var == 'strength':
            # p(x=X) = 1/(x*log(10000)), assume log10
            str = np.linspace(1e3,1e7,nval)
            # s_CPD = np.log(str/smin) / np.log(smax/smin)
            s_dist = 1/(str*4)
            s_dist = s_dist/np.sum(s_dist)
            strengths = np.random.choice(str, size=N, p=s_dist)
            data = np.vstack((data, strengths))
        if var == 'velocity':
            # p(x=X) = (sqrt(2/pi)*exp(-x**2/242)*x**2)/1331
            v = np.linspace(0,50000,nval) # At infinite distance, in m/s
            vi = np.sqrt(11200**2 + v**2) # Impact velocity, in m/s
            # a = 1.1e4
            v_dist = (np.sqrt(2/np.pi)*np.exp(-(v/1000)**2/242)*(v/1000)**2)/1331
            v_dist = v_dist/np.sum(v_dist)
            velocities = np.random.choice(vi, size=N, p=v_dist)
            data = np.vstack((data, velocities))
            # v_CPD = erf(v/(a*np.sqrt(2))) \
                      # - (v/a)*np.exp(-(v**2)/(2*a**2))*np.sqrt(2/np.pi)
        if var == 'density':
            # p(x=X) = exp(-(x-3e3)**2/2e6)/(1000*sqrt(2*pi))
            rho = np.linspace(0,7000,nval)
            rho_dist = np.exp(-(rho-3e3)**2/2e6)/(1000*np.sqrt(2*np.pi))
            rho_dist = rho_dist/np.sum(rho_dist)
            densities = np.random.choice(rho,size=N, p=rho_dist)
            data = np.vstack((data, densities))
            #rho_CPD = 0.5*(1+erf((rho-3e3)/(1e3*np.sqrt(2))))

    # Run the simulation with the above arrays of parameters
    # Makes an ndarray of result dataframes and an ndarray of outcome dicts

    """Will attempt to vectorize or parallelize"""

#    params = np.array([radii,angles,strengths,velocities,densities])
#    param_df = pd.DataFrame(data=params,columns=['radius','angle','strength',
#                                               'velocity','density'])

    #simulation = np.vectorize(planet.solve_atmospheric_entry)

    #results, outcomes = simulation(radius=radii,angle=angles,
    #                               strength=strengths,
    #                               velocity=velocities,density=densities)

    params = np.array([radii,angles,strengths,velocities,densities])
    param_df = pd.DataFrame(data=params.T, columns=['radius','angle','strength',
                                                 'velocity','density'])
    print(param_df)
    param_dd = dd.from_pandas(param_df, npartitions=10)

#    p = param_df.apply(lambda x: planet.solve_atmospheric_entry(radius=param_df['radius'],
#                                        angle=param_df['angle'], strength=param_df['strength'], velocity=param_dd['velocity'],
#                                        density=param_df['density']), axis=1)

    solve_parallel = np.vectorize(planet.solve_atmospheric_entry)
    p = np.full((N,), solve_parallel(radius=radii,
                                        angle=angles, strength=strengths, velocity=velocities,
                                        density=densities))
    calculate_parallel = np.vectorize(planet.calculate_energy)
    p_e = np.full((N,), calculate_parallel(p))
    parallel_outcomes = np.vectorize(planet.analyse_outcome)
    outcome = np.full((N,), parallel_outcomes(p_e))
    print(outcome)
    #param_df['outcomes'] = 

    #outcomes = []

    #for i in range(N):
    #    result, outcome = dask.delayed(planet.impact, nout=2)(radius=radii[i],angle=angles[i],
     #                                   strength=strengths[i],
     #                                   velocity=velocities[i],density=densities[i])
     #   outcomes.append(outcome)

       
    #futures = dask.persist(*outcomes)

#    results = dask.compute(*futures)
    # Convert array of dicts into pandas DataFrame
    outcomes = pd.DataFrame(outcome)
    print(outcomes)

    # Extract 'burst_altitude' column, cases with no airburst will have NaN
    burst_altitudes = np.array(outcomes['burst_altitude'])

    data = np.vstack((data, burst_altitudes))
    data = data[1:,:]

    return pd.DataFrame(data.T, columns=variables+['burst_altitude'])

# def plot_histogram(planet, fiducial_impact, variables): 
