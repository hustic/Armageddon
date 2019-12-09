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
    
    result = planet.solve_atmospheric_entry(radius=fiducial_impact['radius'],
                                            angle=fiducial_impact['angle'],
                                            strength=fiducial_impact['strength'],
                                            velocity=fiducial_impact['velocity'],
                                            density=fiducial_impact['density'])

    for var in variables:
        if var == 'radius':
            radius_series = result['radius']
            radius_dist = (radius_series - rmin) / (rmax - rmin)
        if var == 'angle':
            angle = result['angle']
            angle_dist = np.cos(angle)**2
        if var == 'strength':
            str = result['strength']
            smin = 1e3
            smax = 1e7
            s_dist = np.log(str/smin) / np.log(smax/smin)
        if var == 'velocity':
            v = result['velocity']
            a = 1.1e4
            # at infinity:
            v_dist = erf(v/(a*np.sqrt(2))) \
                     - (v/a)*np.exp(-(v**2)/(2*a**2))*np.sqrt(2/np.pi)
            # I don't know what the bit about impact velocity means
        if var == 'density':
            rho = result['density']
            rho_dist = 0.5*(1+erf((rho-3e3)/(1e3*np.sqrt(2))))

    # Implement your ensemble function here

    return pd.DataFrame(columns=variables+['burst_altitude'], index=range(0))
