import matplotlib.pyplot as plt
import numpy as np
import statistics
import sys
import matplotlib
matplotlib.use('TkAgg')
#plt.style.use('seaborn-v0_8-poster')

if (len(sys.argv)<5):
    print("Dep/Width (0,1), Same y (0,1), signal, bkg, dist, fwhm (for labeling purposes), and Filename(s) please!")
    exit(-1)

smalltext = {'size' : 8} 

matplotlib.rc('font', **smalltext)

DepWid = (int(sys.argv[1])>0)
samey = (int(sys.argv[2])>0)
sig =  sys.argv[3]
bkg =  sys.argv[4]
dist = sys.argv[5]
FWHM = sys.argv[6]

numGraphs= int(len(sys.argv)-7)


if(numGraphs==6):
    #box, graphs = plt.subplots(2,3,sharey=samey)
    box, graphs = plt.subplots(2,3,sharey=False)
    ov6=True
else:
    box, graphs = plt.subplots(1,numGraphs,sharey=samey)
    ov6=False

#ov6=false    
    
box.set_figheight(8)
box.set_figwidth(24)
if(False==ov6):
    for i in range(numGraphs):
        filename = sys.argv[i+7]

        info = np.load(filename)
        data = info['data']
        if(DepWid):
            erStr = "errorWidth"
        else:
            erStr = "errorDepth"    
        error = info[erStr]
        
        atad = [0 for j in range(len(data))]
        for j in range(len(data)):
            atad[j] = 1-data[j]
        data = atad
            
        mean = [statistics.mean(error) for j in range(len(data))]

        if(numGraphs>1):
            graphs[i].plot(data,error,data,mean)
            bigInterval = np.linspace(data[0],data[-1], int(len(data)/5)+1, True)
            littleInterval = np.linspace(data[0],data[-1], len(data), True)
            graphs[i].grid(True, which = 'both',)
            graphs[i].set_xticks(bigInterval)
            graphs[i].set_xticks(littleInterval, minor=True)
            graphs[i].fill_between(data, 0, error,facecolor='C0', alpha=1)
            graphs[i].set_title("Error for data in file\n"+filename)
        else:
            graphs.plot(data,error,data,mean)
            bigInterval = np.linspace(data[0],data[-1], int(len(data)/5)+1, True)
            littleInterval = np.linspace(data[0],data[-1], len(data), True)
            graphs.grid(True, which = 'both',)
            graphs.set_xticks(bigInterval)
            graphs.set_xticks(littleInterval, minor=True)
            graphs.fill_between(data, 0, error,facecolor='C0', alpha=1)
            graphs.set_title("Error for data in file\n"+filename)
else:
    MaxErr=0.0
    MinErr=0.0
    for i in range(2):
        for l in range(3):
            filename = sys.argv[(3*i+l)+7]

            info = np.load(filename)
            data = info['data']
            if(DepWid):
                erStr = "errorWidth"
            else:
                erStr = "errorDepth"    
            error = info[erStr]
            
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
            graphs[i,l].set_yticks([MinErr, (MinErr*3/4), MinErr/2, MinErr/4, 0, MaxErr/4, MaxErr/2, (MaxErr*3/4), MaxErr])
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
                #graphs[i,l].set_yticks([MinErr, 0, MaxErr])
                #graphs[i,l].set_yticks([MinErr, (MinErr*3/4), MinErr/2, MinErr/4, 0, MaxErr/4, MaxErr/2, (MaxErr*3/4), MaxErr])
                graphs[i,l].set_yticks([-scale, (-scale*3/4), -scale/2, -scale/4, 0, scale/4, scale/2, (scale*3/4), scale])
                graphs[i,l].set_ylim([-scale,scale])
    

if(DepWid):
    labStr = "Width"
else:
    labStr = "Depth"    
box.supxlabel("Level Of Attenuation (fraction of light blocked out)", fontsize=24)
box.supylabel("Normalized Error Level\n(Predicted peak location\n minus true peak location)", fontsize=24)
box.suptitle("Effect of optical attenuation for distance fraction "+dist+"\n phi_sig: "+str(sig)+", phi_bkg: "+str(bkg)+", FWHM: "+FWHM+" Histogram type: "+labStr+".", fontsize=24)
#plt.figure(figsize=(12,4))
if(samey):
    samstring = "Sharedy"
else:
    samstring = "Zoomedy"

plt.savefig(str("Graphs/sig-"+str(sig)+"-bkg-"+str(bkg)+"-FWHM-"+str(FWHM)+"-dist-"+dist+"-"+samstring+"-"+labStr+".png"))
print(dist+" done.")
#plt.show()
