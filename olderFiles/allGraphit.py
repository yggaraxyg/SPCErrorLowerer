import matplotlib.pyplot as plt
import numpy as np
import statistics
import sys
import matplotlib
matplotlib.use('TkAgg')
#plt.style.use('seaborn-v0_8-poster')

if (len(sys.argv)<2):
    print("Same y (0,1), and fwhm please!")
    exit(-1)
    
smalltext = {'size' : 8} 

matplotlib.rc('font', **smalltext)

samey = (int(sys.argv[1])>0)
FWHM = sys.argv[2]

box, graphs = plt.subplots(2,3,sharey=False)

box.set_figheight(8)
box.set_figwidth(24)
MaxErr=0.0
MinErr=0.0
for i in range(2):
    for l in range(3):
        error=[0 for i in range(101)]
        
        for m in np.arange(0.05, 1.01, 0.05):
            filestring = "RunData/free-"
            if(i==0):
                filestring+="False-"
            else:
                filestring+="True-"
            filestring+="sig-1.0000-bkg-4.0000-tmax-100.0-tbins-1000-FWHM-"
            filestring+=FWHM
            filestring+="-reps-10-cent-"
            filestring+=("{0:0.2f}".format(m))
            filestring+="-bins-16-pulses-10000-deadtime-"
            if(l==0):
                filestring+="0"
            elif(l==1):
                filestring+="15"
            else:
                filestring+="75"
            filestring+=".npz"
            #print(filestring)
            info = np.load(filestring)
            data = info['data']
            terror = info['error']
            for n in range(len(terror)):
                error[n]+=terror[n]
                
        for m in range(len(error)):
            error[m]=error[m]/20

        lmin = 0
        lmax = 0
        if(samey==False):
            MaxErr=0.0
            MinErr=0.0
        for j in range(len(error)):
            if(MaxErr<error[j]):
                MaxErr=error[j]
            if(MinErr>error[j]):
                MinErr=error[j]
            if(abs(error[lmin])>abs(error[j])):
                lmin=j
            if(abs(error[lmax])<abs(error[j])):
                lmax=j
           
        atad = [0 for j in range(len(data))]
        for j in range(len(data)):
            atad[j] = 1-data[j]
        data = atad
        
        mean = [statistics.mean(error) for j in range(len(data))]
        
        graphs[i,l].plot(data, error, data, mean)
        bigInterval = np.linspace(data[0],data[-1], int(len(data)/5)+1, True)
        littleInterval = np.linspace(data[0],data[-1], len(data), True)
        #graphs[i,l].grid(False, which = 'major', linewidth=1.5)
        #graphs[i,l].grid(False, which = 'minor', linewidth=0.75)
        graphs[i,l].grid(False)
        graphs[i,l].set_xticks(bigInterval)
        graphs[i,l].set_xticks(littleInterval, minor=True)
        graphs[i,l].set_yticks([MinErr, (MinErr*3/4), MinErr/2, MinErr/4,0, MaxErr/4, MaxErr/2, (MaxErr*3/4), MaxErr])
        graphs[i,l].fill_between(data, 0, error,facecolor='C0', alpha=1)
        if(samey):
            graphs[i,l].vlines(data[lmin],-1,1,colors='g')
            graphs[i,l].vlines(data[lmax],-1,1,colors='r')
        else:
            graphs[i,l].vlines(data[lmin],MinErr,MaxErr,colors='g')
            graphs[i,l].vlines(data[lmax],MinErr,MaxErr,colors='r')
            
        titlestring = "FreeRun "
        if(i==0):
            titlestring+="off"
        else:
            titlestring+="on"
        titlestring+=", "
        if(l==0):
            titlestring+="0"
        elif(l==1):
            titlestring+="15"
        else:
            titlestring+="75"
        titlestring+="ns DeadTime"
            
        graphs[i,l].set_title(titlestring,fontsize=12)
        
if(samey):
    for i in range(2):
        for l in range(3):
            scale = min(abs(MinErr),abs(MaxErr))*1.5
            if(scale==0):
                scale=max(abs(MinErr),abs(MaxErr))
            graphs[i,l].set_yticks([-scale, (-scale*3/4), -scale/2, -scale/4, 0, scale/4, scale/2, (scale*3/4), scale])
            graphs[i,l].set_ylim([-scale,scale])
    
        
box.supxlabel("Level Of Attenuation (fraction of light blocked out)", fontsize=24)
box.supylabel("Normalized Error Level\n(Predicted peak location\n minus true peak location)", fontsize=24)
box.suptitle("Average Effect of optical attenuation for distance fraction across all distances, phi_sig: 1, phi_bkg: 4, FWHM: "+FWHM+".", fontsize=24)
#plt.figure(figsize=(12,4))
if(samey):
    samstring = "Sharedy"
else:
    samstring = "Zoomedy"
plt.savefig("5FWHMGraphs/all-"+samstring+".png")


