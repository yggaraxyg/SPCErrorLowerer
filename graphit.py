import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')

if (len(sys.argv)!=2):
    print("Filename please!")
    exit(-1)

filename = sys.argv[1]

info = np.load(filename)
data = info['data']
error = info['error']
    
plt.plot(data,error)
bigInterval = np.linspace(data[0],data[-1], int(len(data)/5)+1, True)
littleInterval = np.linspace(data[0],data[-1], len(data), True)
plt.grid(True, which = 'both',)
plt.xticks(bigInterval)
plt.xticks(littleInterval, minor=True)
#plt.gca().set_yscale("log")
plt.gca().set_ylim([-0.25,0.25])
plt.fill_between(data, 0, error,facecolor='C0', alpha=1)
plt.xlabel("Attenuation")
plt.ylabel("Error Level (normalized via division by tbins)")
plt.title("Error levels for attenuation in graph from file\n"+filename)
plt.show()
