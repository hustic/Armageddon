import numpy as np
import pandas as pd

class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------

        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function ``rho = rho0 exp(-z/H)``.
            Options are ``exponential``, ``tabular``, ``constant`` and ``mars``

        atmos_filename : string, optional
            If ``atmos_func`` = ``'tabular'``, then set the filename of the table
            to be read in here.

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient
o90okl
        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        Returns
        -------

        None
        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0

        if atmos_func == 'exponential':
            self.rhoa = lambda x: rho0 * np.exp(-x/self.H)
        elif atmos_func == 'tabular':
            raise NotImplementedError
        elif atmos_func == 'mars':
            raise NotImplementedError
        elif atmos_func == 'constant':
            self.rhoa = lambda x: rho0
        else:
            raise NotImplementedError

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact event.
        Also calculates the kinetic energy lost per unit altitude and
        analyses the result to determine the outcome of the impact.

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

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        Returns
        -------

        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``, ``dedz``

        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.

            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        result = self.solve_atmospheric_entry(radius, velocity, density, strength, angle)
        result = self.calculate_energy(result)
        outcome = self.analyse_outcome(result)

        return result, outcome

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact scenario

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

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        Returns
        -------
        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``
        """

        # Enter your code here to solve the differential equations
        if radians is False:
            angle = angle * (np.pi)/180

        T = 120
        T_arr = []
        t = 0
        T_arr.append(0)
        mass = density * 4/3 * radius**3 * np.pi
        init_distance = 0
        y = np.array([velocity, mass, angle, init_altitude, init_distance, radius])
        Y = []
        Y.append(y)

        while t <= T:
            t = t + dt
            T_arr.append(t)

            if strength >= (self.rhoa(y[3]) * y[0]**2):
                fragmented = True
            else:
                fragmented = False

            y = self.midpoint_implicit_euler(y, self.f, dt, fragmented, density)
            Y.append(y)

            if y[3] <= 0:
                break

        Y = np.array(Y)
        return pd.DataFrame({'velocity': Y[:, 0],
                             'mass': Y[:, 1],
                             'angle': Y[:, 2],
                             'altitude': Y[:, 3],
                             'distance': Y[:, 4],
                             'radius': Y[:, 5],
                             'time': T_arr})#, index=range(1))

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------

        result : DataFrame
            A pandas DataFrame with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns
        -------

        Result : DataFrame
            Returns the DataFrame with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame

        result = result.copy()

        dedz = np.zeros((len(result),))
        dedz[0] = 0
        for i in range(1,len(result)):
            dedz[i] = ((1/2 * result.mass[i] * result.velocity[i]**2) - (1/2 * result.mass[i-1] * result.velocity[i-1]**2)) / (result.altitude[i] - result.altitude[i-1])

        result.insert(len(result.columns), 'dedz', dedz)

        return result

    def analyse_outcome(self, result):
        """
        Inspect a prefound solution to calculate the impact and airburst stats

        Parameters
        ----------

        result : DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time

        Returns
        -------

        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.

            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        # Enter your code here to process the result DataFrame and
        # populate the outcome dictionary.
        outcome = {}
        event = 0
        index_max = result.dedz.idxmax()
        if result.altitude[index_max] > 0: # check for Airburst
            burst_peak_dedz = result.dedz[index_max]
            burst_altitude = result.altitude[index_max]
            burst_total_ke_lost = 1/2 * ((result.mass[0] * result.velocity[0]**2) - (result.mass[index_max] * result.velocity[index_max]**2))#sum(result.iloc['dedz'][:index_max]

            outcome['burst_peak_dedz'] = burst_peak_dedz
            outcome['burst_altitude'] = burst_altitude
            outcome['burst_total_ke_lost'] = burst_total_ke_lost
            
            event += 1
    
        if result.mass.iloc[-1] != 0: # checl for Cratering
            impact_time = result.time.iloc[-1]
            impact_mass = result.mass.iloc[-1]
            impact_speed = result.velocity.iloc[-1]

            outcome['impact_time'] = impact_time
            outcome['impact_mass'] = impact_mass
            outcome['impact_speed'] = impact_speed

            event += 2

        if event == 1:
            outcome['outcome'] = 'Airburst'
        elif event == 2:
            outcome['outcome'] = 'Cratering'
        elif event == 3:
            outcome['outcome'] = 'Airburst and cratering'
        else:
            raise ValueError
        return outcome

    def f(self, y, fragmented, density):
        '''
        0: velocity
        1: mass
        2: angle
        3: altitude
        4: distance
        5: radius
        '''
        f = np.zeros_like(y)
        f[0] = - (self.Cd * self.rhoa(y[3]) * y[0]**2 * np.pi * y[5]**2) / (2 * y[1]) + (self.g * np.sin(y[2]))
        f[1] = - (self.Ch * self.rhoa(y[3]) * np.pi * y[5]**2 * y[0]**3) / (2 * self.Q)
        f[2] = (self.g * np.cos(y[2])) / y[0]  - (self.Cl * self.rhoa(y[3]) * np.pi * y[5]**2 * y[0]) / (2 * y[1]) - (y[0] * np.cos(y[2])) / (self.Rp + y[3])
        f[3] = - y[0] * np.sin(y[2])
        f[4] = (y[0] * np.cos(y[2])) / (1 + y[3] / self.Rp)
        if fragmented == True:
            f[5] = (7/2 * self.alpha * (self.rhoa(y[3]) / density))**(1/2) * y[0]
        else:
            f[5] = 0
        return f

    def explicit_euler(self, y, f, dt, fragmented, density):
        y = y + f(y, fragmented, density) * dt
        return y
        
    def implicit_euler(self, y, f, dt, fragmented, density):
        y_dummy = y + f(y, fragmented, density) * dt
        y = y + f(y_dummy, fragmented, density) * dt
        return y
        
    def midpoint_implicit_euler(self, y, f, dt, fragmented, density):
        y_dummy = y + f(y, fragmented, density) * dt
        y = y + (f(y, fragmented, density) + f(y_dummy, fragmented, density)) * 0.5 * dt
        return y
