import armageddon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

earth = armageddon.Planet()

# last entry: index 9202, 26 velocity,

fiducial_impact = {'radius': 10.0,
                   'angle': 45.0,
                   'strength': 100000.0,
                   'velocity': 21000.0,
                   'density': 3000.0}

print('Starting simulation now...')
    
#ensemble = armageddon.ensemble.solve_ensemble(earth,
#                                              fiducial_impact,
#                                              variables=['angle', 'radius', 'strength', 'velocity', 'density'], radians=False,
#                                              rmin=8, rmax=12)
                                              
#print(ensemble)

for i in range(10):
    start_time = time.time()
    ensemble = armageddon.ensemble.solve_ensemble(earth,
                                              fiducial_impact,
                                              variables=['angle', 'radius', 'strength', 'velocity', 'density'], radians=False,
                                              rmin=8, rmax=12)
    print("--- %s seconds ---" % (time.time() - start_time))
print(ensemble)
# round to nearest integer for now
#burst_altitude = np.array(ensemble['burst_altitude']).astype('int')
var = np.array(ensemble['angle'])

#y = np.bincount(burst_altitude)
#ii = np.nonzero(y)[0]
#counts = np.vstack((ii,y[ii])).T

#plt.plot(counts[:,0],counts[:,1])
plt.hist(var)
#plt.ylim((0,np.max(counts[:,1])*1.3))
plt.show()
