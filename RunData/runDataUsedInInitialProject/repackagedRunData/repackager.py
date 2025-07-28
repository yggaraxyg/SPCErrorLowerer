import numpy as np
import sys

if (len(sys.argv)<3):
    print("newname and Filename(s) please!")
    exit(-1)

box = []
for i in range(len(sys.argv)-2):
    filename = sys.argv[i+2]
    curfile = np.load(filename)
    #'''
    curerr = curfile['errorWidth']
    for i in range(len(curerr)):
        box.append(curerr[i])
    #'''
    curerr = curfile['errorDepth']
    for i in range(len(curerr)):
        box.append(curerr[i])
    #'''
np.savez(sys.argv[1], error=box)
