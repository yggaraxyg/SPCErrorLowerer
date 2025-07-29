import numpy as np
import sys

if (len(sys.argv)<4):
    print("keepIndividual(0,1) newname and Filename(s) please!")
    exit(-1)

keepIndividual=(int(sys.argv[1])>0)

if(keepIndividual):
    box = []
    for i in range(len(sys.argv)-3):
        filename = sys.argv[i+3]
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
else:
    box = [0 for i in np.load(sys.argv[3])['errorWidth']]
    for i in range(len(sys.argv)-3):
        filename = sys.argv[i+3]
        curfile = np.load(filename)
        #'''
        curerr = curfile['errorWidth']
        for i in range(len(curerr)):
            box[i]+=curerr[i]
        #'''
        curerr = curfile['errorDepth']
        for i in range(len(curerr)):
            box[i]+=curerr[i]
        #'''
    for i in range(len(box)):
        box[i] = box[i]/(len(sys.argv)-3)

np.savez(sys.argv[2], error=box)
