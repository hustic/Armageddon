from solver import Planet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

earth = Planet(atmos_func='exponential')
df = earth.solve_atmospheric_entry(10, 20e3, 3000, 1e5, 45)
df = earth.calculate_energy(df)
print(df)
fig = plt.figure(figsize=(5,5))
plt.plot(df['dedz'], df['altitude'])
plt.xlabel('velocity')
plt.ylabel('altitude')
plt.show()


'''
0: velocity
1: mass
2: angle
3: altitude
4: distance
5: radius
'''