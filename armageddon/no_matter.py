from solver import Planet
'''import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analytical import anal_sol
'''
import timeit

def costly_func():
    earth = Planet(atmos_func='exponential', atmos_filename='../data/AltitudeDensityTable.csv')
    df, out = earth.impact(10, 20e3, 3000, 10e5, 45, num_scheme='RK', fragmentation=True, dt=0.01, init_altitude=80000)
    return df, out
df, out = costly_func()
print(df, out)
#H_plot = np.array(df.altitude)
#anal_df = anal_sol(H_plot, 10, 20e3, 3000, 3000, 45, 100e3, False)
#print(out)
#data = pd.read_csv('../data/AltitudeDensityTable.csv', header=None, delim_whitespace=True, skiprows=6)
#print(data)
'''schemes = [
            'EE',
            'IE',
            'MIE',
            'RK'
            ]
'''
'''dts = [0.1, 0.05, 0.01, 0.005, 0.001]
dt_rms = []
for dt in dts:
    RMS_velocity = []
    RMS_dedz = []
    for scheme in schemes:
        df, dic = earth.impact(5, 10e3, 1200, 1e5, 30, 100e3, dt, False, False, scheme)
        H_plot = np.array(df.altitude)
        anal_df = anal_sol(H_plot, 5, 10e3, 1200, 1e5, 30, 100e3, False)
        RMS_velocity.append(np.sqrt(1/len(df) * sum((anal_df.velocity - df.velocity)**2)))
    dt_rms.append(RMS_velocity)

dt_rms = np.array(dt_rms).T
'''
'''fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
#ax3 = plt.subplot(213)
'''
#äax1.scatter(df.altitude, df.velocity, marker='.', color='r', label='num')
#ax1.plot(H_plot, anal_df.velocity, label='anal')
'''ax1.plot(dts, dt_rms[0], label='EE')
ax1.plot(dts, dt_rms[1], label='IE')
ax1.plot(dts, dt_rms[2], label='MIE')
ax1.plot(dts, dt_rms[3], label='RK')
ax1.set_xlabel('altitude', fontsize=14)
ax1.set_ylabel('v', fontsize=14)
#ax1.set_title("RMSs")
#ax1.set_ylim(0, 40e3)
ax1.grid()
ax1.legend()'''
'''
ax2.scatter(df.altitude, df.dedz, marker='.', color='r', label='num')
#ax2.plot(H_plot, anal_df.dedz, label='anal')
ax2.set_xlabel('altitude', fontsize=14)
ax2.set_ylabel('dedz', fontsize=14)
#ax2.set_title("numerical")
ax2.grid()
ax2.legend()
#ax2.set_xlim((0, 40e3))'''
'''
ax3.plot(data.iloc[0], data.iloc[1], label='tabular density')
ax3.grid()
ax3.legend()
'''
#plt.show()

'''
0: velocity
1: mass
2: angle
3: altitude
4: distance
5: radius
'''
