from solver import Planet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

earth = Planet(atmos_func='exponential')
df = earth.solve_atmospheric_entry(10, 20e3, 3000, 1e5, 45)
df = earth.calculate_energy(df)
print(df)
#earth.analyse_outcome(df)

fig = plt.figure(figsize=(7,7))
plt.plot(df['dedz'], df['altitude'])
plt.xlabel('dedz')
plt.ylabel('altitude')
#plt.ylim(0,20000)
plt.grid()
plt.show()


'''
0: velocity
1: mass
2: angle
3: altitude
4: distance
5: radius
'''