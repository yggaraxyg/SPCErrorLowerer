import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')

if (len(sys.argv)<4):
    print("Reverse (0,1), Same y (0,1), dist (for labeling purposes), and Filename(s) please!")
    exit(-1)

smalltext = {'size' : 5} 

matplotlib.rc('font', **smalltext)

reverse = (int(sys.argv[1])>0)
samey = (int(sys.argv[2])>0)
dist = sys.argv[3]

if(int(len(sys.argv)-4)==6):
    box, graphs = plt.subplots(2,3,sharey=samey)
    ov6=True
else:
    box, graphs = plt.subplots(1,int((len(sys.argv)-2)/2),sharey=samey)
    ov6=False

box.set_figheight(10)
box.set_figwidth(20)
if(False==ov6):
    for i in range(int((len(sys.argv)-2)/2)):
    
        filename = sys.argv[i+4]

        info = np.load(filename)
        data = info['data']
        error = info['error']
        
        if(reverse):
            atad = [0 for j in range(len(data))]
            for j in range(len(data)):
                atad[j] = 1-data[j]
            data = atad

            graphs[i].plot(data,error)
        bigInterval = np.linspace(data[0],data[-1], int(len(data)/5)+1, True)
        littleInterval = np.linspace(data[0],data[-1], len(data), True)
        graphs[i].grid(True, which = 'both',)
        graphs[i].set_xticks(bigInterval)
        graphs[i].set_xticks(littleInterval, minor=True)
        graphs[i].fill_between(data, 0, error,facecolor='C0', alpha=1)
        graphs[i].set_title("Error for data in file\n"+filename)
else:
    for i in range(2):
        for l in range(3):
            filename = sys.argv[(3*i+l)+4]

            info = np.load(filename)
            data = info['data']
            error = info['error']
        
            if(reverse):
                atad = [0 for j in range(len(data))]
                for j in range(len(data)):
                    atad[j] = 1-data[j]
                data = atad
            
            graphs[i,l].plot(data,error)
            bigInterval = np.linspace(data[0],data[-1], int(len(data)/5)+1, True)
            littleInterval = np.linspace(data[0],data[-1], len(data), True)
            graphs[i,l].grid(True, which = 'major', linewidth=1.5)
            graphs[i,l].grid(True, which = 'minor', linewidth=0.75)
            graphs[i,l].set_xticks(bigInterval)
            graphs[i,l].set_xticks(littleInterval, minor=True)
            graphs[i,l].fill_between(data, 0, error,facecolor='C0', alpha=1)
            titlestring = "freerun "
            if(i==0):
                titlestring+="off "
            else:
                titlestring+="on "
            titlestring+=", a centre at "
            titlestring+=dist
            titlestring+=" and "
            if(l==0):
                titlestring+="0"
            elif(l==1):
                titlestring+="15"
            else:
                titlestring+="75"
            titlestring+="ns deadtime."
            
            graphs[i,l].set_title(titlestring,fontsize=12)

    
        
box.supxlabel("Level Of Attenuation (fraction of light blocked out)", fontsize=18)
box.supylabel("Error Level (normalized via division by tbins)", fontsize=18)
box.suptitle("Error Levels With...", fontsize=18)
#plt.figure(figsize=(12,4))
if(samey):
    samstring = "Sharedy"
else:
    samstring = "Zoomedy"
plt.savefig("allGraphs/dist-"+str("point".join(dist.rsplit(".")))+"-"+samstring+"-"+".png")
#plt.show()
