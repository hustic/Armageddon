from collections import OrderedDict
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from pytest import fixture


# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly

@fixture(scope='module')
def armageddon():
    """Perform the module import"""
    import armageddon
    return armageddon


@fixture(scope='module')
def planet(armageddon):
    """Return a default planet with a constant atmosphere"""
    return armageddon.Planet(atmos_func='constant')


@fixture(scope='module')
def input_data():
    input_data = {'radius': 1.,
                  'velocity': 1.0e5,
                  'density': 3000.,
                  'strength': 1e32,
                  'angle': 30.0,
                  'init_altitude': 100e3,
                  'dt': 0.05,
                  'radians': False
                  }
    return input_data


@fixture(scope='module')
def result(planet, input_data):
    """Solve a default impact for the default planet"""

    result = planet.solve_atmospheric_entry(**input_data)
    print(result)
    return result


# def test_import(armageddon):
#     """Check package imports"""
#     assert armageddon


# def test_planet_signature(armageddon):
#     """Check planet accepts specified inputs"""
#     inputs = OrderedDict(atmos_func='constant',
#                          atmos_filename=None,
#                          Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
#                          alpha=0.3, Rp=6371e3,
#                          g=9.81, H=8000., rho0=1.2)
#
#     # call by keyword
#     planet = armageddon.Planet(**inputs)
#
#     # call by position
#     planet = armageddon.Planet(*inputs.values())


# def test_attributes(planet):
#     """Check planet has specified attributes."""
#     for key in ('Cd', 'Ch', 'Q', 'Cl',
#                 'alpha', 'Rp', 'g', 'H', 'rho0'):
#         assert hasattr(planet, key)


# def test_solve_atmospheric_entry(result, input_data):
#     """Check atmospheric entry solve.
#
#     Currently only the output type for zero timesteps."""
#
#     assert type(result) is pd.DataFrame
#
#     for key in ('velocity', 'mass', 'angle', 'altitude',
#                 'distance', 'radius', 'time'):
#         assert key in result.columns
#
#     assert np.isclose(result.velocity.iloc == P1['y'][0])
#     assert np.isclose(result.angle.iloc == P1['y'][2])
#     assert np.isclose(result.altitude.iloc == P1['y'][3])
#     assert result.distance.iloc[0] == 0.0
#     assert np.isclose(result.radius.iloc == P1['y'][5])
#     assert result.time.iloc[0] == 0.0


# def test_calculate_energy(planet, result):
#     energy = planet.calculate_energy(result=result)
#
#     print(energy)
#
#     assert type(energy) is pd.DataFrame
#
#     for key in ('velocity', 'mass', 'angle', 'altitude',
#                 'distance', 'radius', 'time', 'dedz'):
#         #assert key in energy.columns
#         assert energy == energy_z
#
#
# def test_analyse_outcome(planet, result):
#     outcome = planet.analyse_outcome(result)
#
#     assert type(outcome) is dict
#
#
# def test_ensemble(planet, armageddon):
#     fiducial_impact = {'radius': 0.0,
#                        'angle': 0.0,
#                        'strength': 0.0,
#                        'velocity': 0.0,
#                        'density': 0.0}
#
#     ensemble = armageddon.ensemble.solve_ensemble(planet,
#                                                   fiducial_impact,
#                                                   variables=[], radians=False,
#                                                   rmin=8, rmax=12)
#
#     assert 'burst_altitude' in ensemble.columns


# def test_solve_atmospheric_entry_with_scipy_odeint(planet, reslut, input_data):
#     C_D, g, C_H, Q, C_L, R_p, alpha, rho_m, rho_0, H, Y = (1.,9.81,0.1,1e7,1e-3,6371000.,0.3,3000,1.2,8000.,1e32)
#     v, m, theta, z, x, r = Point
#     A = np.pi*(r**2)
#     rho_a = rho_0*np.exp(-z/H)
#     new_r = 0
#     if rho_a * (v**2) >= Y:
#         new_r = v*((7/2*alpha*rho_a/rho_m)**(1/2))
#     return np.array([(-C_D*rho_a*A*(v**2))/(2*m) + g*np.sin(theta),
#                      (-C_H*rho_a*A*(v**3))/(2*Q),
#                      (g*np.cos(theta))/(v) - (C_L*rho_a*A*v)/(2*m) - (v*np.cos(theta))/(R_p + z),
#                      -v*np.sin(theta),
#                      (v*np.cos(theta))/(1 + z/R_p),
#                      new_r
#                     ])
# tmax = 60
# t = np.arange(0,tmax,0.05)
# P1 = solve_ivp(dmove,(0,tmax),(1.0e5,4000*np.pi*(10**3),np.pi/6,100e3,0.,1.),method='RK45', t_eval=t)
# energy_t = -(0.5*P1['y'][1, 1:]*(P1['y'][0, 1:]**2) - 0.5*P1['y'][1, :-1]*(P1['y'][0, :-1]**2))/(t[1:]-t[:-1])
# energy_z = (0.5*P1['y'][1, 1:]*(P1['y'][0, 1:]**2) - 0.5*P1['y'][1, :-1]*(P1['y'][0, :-1]**2))/(P1['y'][3, 1:]-P1['y'][3, :-1])
# # from mpl_toolkits.mplot3d import Axes3D
# # import matplotlib.pyplot as plt

def test_solve_atmospheric_entry_with_scipy_odeint(planet, result, input_data):
    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns

    t = np.arange(0, 1000, input_data['dt'])
    init_state = (input_data['velocity'], input_data['density']*4/3*np.pi*(input_data['radius']**3),
                  input_data['angle'], input_data['init_altitude'], 0., input_data['radius'])
    P1 = odeint(planet.dmove_odeint, init_state, t, args=([input_data['strength'], input_data['density']],))
    # we don't need to invert angle delete something here (about angle)

    result = pd.DataFrame({'velocity': P1[:, 0],
                              'mass': P1[:, 1],
                              'angle': P1[:, 2],
                              'altitude': P1[:, 3],
                              'distance': P1[:, 4],
                              'radius': P1[:, 5],
                              'time': t}, index=range(len(P1)))
    assert result.velocity.iloc[0] == input_data['velocity']
    assert result.angle.iloc[0] == input_data['angle']
    assert result.altitude.iloc[0] == input_data['init_altitude']
    assert result.distance.iloc[0] == 0.0
    assert result.radius.iloc[0] == input_data['radius']
    assert result.time.iloc[0] == 0.0


def test_solve_atmospheric_entry_with_analytic(result, input_data):
    assert  type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns

    assert result.velocity.iloc[0] == input_data['velocity']
    assert result.angle.iloc[0] == input_data['angle']
    assert result.altitude.iloc[0] == input_data['init_altitude']
    assert result.distance.iloc[0] == 0.0
    assert result.radius.iloc[0] == input_data['radius']
    assert result.time.iloc[0] == 0.0

