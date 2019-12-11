import numpy as np
import pandas as pd
import os
class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2, fragmentation=True, num_scheme='EE'):
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
            Dispersion coefficient o90okl

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
            BASE_PATH = os.path.dirname(os.path.dirname(__file__))
            atmos_filename= os.sep.join((BASE_PATH,'/data/AltitudeDensityTable.csv'))
            table = pd.read_csv(atmos_filename,header =None, delim_whitespace=True,skiprows=6)
            #'Altitude':table.iloc[:,0],'Density':iloc[:,1],'Scale_Height':iloc[:,2]
            self.rhoa = lambda x: table.iloc[int(x/10),1]*np.exp((table.iloc[int(x/10),0]-x)/table.iloc[int(x/10),2])
            
        elif atmos_func == 'mars':
            self.rhoa = lambda x: 0.699*np.exp(-0.00009*x)/(0.1921*((249.7-0.00222*x)*(x>=7000)+(242.1-0.000998*x)*(x<7000)))
        elif atmos_func == 'constant':
            self.rhoa = lambda x: rho0
        else:
            raise NotImplementedError

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100e3, dt=0.05, radians=False,
               fragmentation=True, num_scheme='EE'):
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
        result = self.solve_atmospheric_entry(radius, velocity, density, strength, angle,
        init_altitude, dt, radians, fragmentation, num_scheme)
        result = self.calculate_energy(result)
        outcome = self.analyse_outcome(result)

        return result, outcome

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False,
            fragmentation=True, num_scheme='EE'):
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
        num_scheme_dict = {
            'EE': self.explicit_euler,
            'IE': self.implicit_euler,
            'MIE': self.midpoint_implicit_euler,
            'RK': self.runge_kutta
            }

        if radians is False: # converts degrees to radians
            angle = angle * (np.pi)/180

        T = 120 # max duration of simulation in seconds
        T_arr = [] # list to store the all timesteps
        t = 0 # inital time assumed to be zero
        T_arr.append(0) # storing first time

        mass = density * 4/3 * radius**3 * np.pi # defining the mass of astroid assuming a sphere shape
        init_distance = 0 # intial distance assumed to be zero
        y = np.array([velocity, mass, angle, init_altitude, init_distance, radius]) # defining initial condition array

        Y = [] # empty list to store solution array for every timestep
        Y.append(y) # store initial condition

        while t <= T: # initiate timeloop
            t = t + dt # move up to next timestep
            T_arr.append(t) # store new timestep

            if strength <= (self.rhoa(y[3]) * y[0]**2) and fragmentation is True:
                fragmented = True # define status of fragmentation
            else:
                fragmented = False

            y = num_scheme_dict[num_scheme](y, self.f, dt, fragmented, density) # compute values for next timestep
            Y.append(y) #store caomputed values

            if y[1] <= 0: # stop simulation if mass becomes zero
                break

            if y[3] <= 0: # stop simulation if mass reaches ground
                break

        Y = np.array(Y)

        if radians is False:
            Y[:, 2] = np.round_(list(map(lambda x: x * 180/np.pi, Y[:, 2])), decimals=1)

        return pd.DataFrame({'velocity': Y[:, 0], # return all the stored values in pd.DataFrame
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
        dedz = np.zeros((len(result),)) # create array to store dedz results
        dedz[0] = 0 # initial dedz
        for i in range(1,len(result)): # loop through all rows of result
            energy = ((1/2 * result.mass[i] * result.velocity[i]**2) - (1/2 *result.mass[i-1] * result.velocity[i-1]**2))/4.184e12
            alt = (result.altitude[i] - result.altitude[i-1])/1e3
            dedz[i] = energy / alt
            # get dedz as released energy per altitude
        result.insert(len(result.columns), 'dedz', dedz) # add dedz to DataFrame 'result'

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
        outcome = {} # set empty dictionary
        event = 0 # set classifying index to zero

        index_max = result.dedz.idxmax() # determine index at which airburst happens
        if result.altitude[index_max] > 0: # check for Airburst
            burst_peak_dedz = result.dedz[index_max] # released energy at airbusrt
            burst_altitude = result.altitude[index_max] # altitude of airburst
            burst_total_ke_lost = 1/2 * ((result.mass[0] * result.velocity[0]**2) - (result.mass[index_max] * result.velocity[index_max]**2))#sum(result.iloc['dedz'][:index_max]
            # above is: total released energy
            # add the above three parameters to dictionary below
            outcome['burst_peak_dedz'] = burst_peak_dedz
            outcome['burst_altitude'] = burst_altitude
            outcome['burst_total_ke_lost'] = burst_total_ke_lost
            
            event += 1 # increase classifying index to by one
    
        if result.mass.iloc[-1] != 0: # check for Cratering with mass being zero when simulation is finished
            impact_time = result.time.iloc[-1] # difference in seconds between entering atmosphere and impact
            impact_mass = result.mass.iloc[-1] # 
            impact_speed = result.velocity.iloc[-1]

            outcome['impact_time'] = impact_time
            outcome['impact_mass'] = impact_mass
            outcome['impact_speed'] = impact_speed

            event += 2 # increase classifying index to by two

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

    def runge_kutta(self, y, f, dt, fragmented, density):
        k1 = f(y, fragmented, density) * dt
        k2 = f(y+k1/2, fragmented, density) * dt
        k3 = f(y+k2/2, fragmented, density) * dt
        k4 = f(y+k3, fragmented, density) * dt

        y = y + (k1 + 2 * (k2 + k3) + k4) / 6
        return y
