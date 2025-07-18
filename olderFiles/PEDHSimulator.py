from SPCSimLib.SPCSim.sensors import PEDHBaseClass
from SPCSimLib.SPCSim.utils import plot_transient, plot_edh, plot_edh_traj

##the code refused to work until I put these here
from SPCSimLib.SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSimLib.SPCSim.data_loaders.transient_loaders import TransientGenerator
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')

##this I put in so I could easily run the thing multiple times from the terminal
import sys

if (len(sys.argv)!=10):
    print("nine arguments please! (laser illumination, background illumination, start of attenuation multiplier, end of attenuation multiplier, interval of attenuation, repetitions of each value, centre of graph, histogram bins, and pulses)")
    exit(-1)

las_il = float(sys.argv[1])
bkg_il = float(sys.argv[2])

startOfGraph = float(sys.argv[3])
endOfGraph  = float(sys.argv[4])
intervalOfGraph = float(sys.argv[5])

reps = int(sys.argv[6])
cent = float(sys.argv[7])
bins = int(sys.argv[8])
puls = int(sys.argv[9])

def main(sig, bkg):
    ##this, as it states, simulates the SPC scene.
    PixLdr = PerPixelLoader(min_dist = cent,
                            tmax = 100,
                            sig_bkg_list = [
                            [sig,bkg]],
                            device = "cpu")

    ##this generates the data for the scene
    # Generate the per pixel data
    data = PixLdr.get_data()

    ##this prepares the arch part of the simulation
    # Creating transient generator with laser time period of 100ns, FWHM 1 and with
    # laser time period divided into 1000 equal time-bins
    tr_gen = TransientGenerator(N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 1)


    ##this gets the background inputs set up
    # Using the get function to generate the transient
    # for a given distance, albedo, intensity, and illumination condition
    phi_bar = tr_gen.get_transient(data["gt_dist"],
                                   data["albedo"],
                                   data["albedo"],
                                   data["alpha_sig"],
                                   data["alpha_bkg"])


    ##this prepares for lasers
    # Setting the dimensions, device, number of EWH bins per pixel
    # and number of laser pulses
    Nr, Nc, N_tbins = phi_bar.shape
    device = PixLdr.device
    N_edhbins = int(bins)
    N_pulses = int(puls)
    #print(N_pulses)

    ##this is the laser simulator
    spc1 = PEDHBaseClass(Nr,
                         Nc,
                         N_pulses,
                         device,
                         N_tbins,
                         N_edhbins,
                         step_params={
                             "k":1
                         })

    captured_data = spc1.capture(phi_bar)

    pedh_data = captured_data["edh"]
    gtedh_data = captured_data["gtedh"]
    #ewh_data = captured_data["ewh"]
    #edh_list = captured_data["traj"]
    #edh_list = np.array(edh_list)

    predictedHistogramBorders=(pedh_data.numpy()[0][0])
    groundTruthHistogramBorders=(gtedh_data.numpy()[0][0])
    
    '''
    print("pedh_data")
    print(PredictedHistogramBorders)

    print("gtedh_data[0][0]")
    print(GroundTruthHistogramBorders)
    '''
    smallestPrePred = 0
    smallestPostPred = 1000
    smallestPreTrue = 0
    smallestPostTrue = 1000

    predBucket=0;
    trueBucket=0;
    
    for i in range(len(predictedHistogramBorders)-1):
        if((predictedHistogramBorders[i+1]-predictedHistogramBorders[i])<(smallestPostPred-smallestPrePred)):
            smallestPrePred = predictedHistogramBorders[i]
            smallestPostPred = predictedHistogramBorders[i+1]
            predBucket=i;
        if((groundTruthHistogramBorders[i+1]-groundTruthHistogramBorders[i])<(smallestPostTrue-smallestPreTrue)):
            smallestPreTrue = groundTruthHistogramBorders[i]
            smallestPostTrue = groundTruthHistogramBorders[i+1]
            trueBucket=i;

    print("Predicted bucket: "+str(predBucket)+" True Bucket: "+str(trueBucket))
    return (abs(smallestPostPred-smallestPostTrue)+abs(smallestPrePred-smallestPreTrue))
    #return (abs(predBucket-trueBucket))

'''
##all this is setup for displayogn the info
ROW, COL = [0,0]

fig, ax = plt.subplots(1,1, figsize=(20,11))
ymax = ((torch.sum(ewh_data[ROW,COL,:])/N_edhbins)).item()
plot_edh(pedh_data[ROW,COL,:],
         ax,
         tr = ewh_data[ROW, COL,:].numpy(),
         ymax = ymax)
plot_edh(gtedh_data[ROW,COL,:], ax,
         tr = phi_bar[ROW, COL,:].numpy()*spc1.N_pulses,
        #  crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax,
         ymax = ymax, ls='--')
ax.set_title(r'Final EDH boundaries for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,COL], data["alpha_bkg"][ROW,COL], spc1.N_pulses))

fig1, ax1 = plt.subplots(1,1, figsize=(20,11))
plot_edh_traj(ax1, edh_list, gtedh_data[ROW,COL,1:-1], ewh_data[0,0,:].cpu().numpy())
ax1.set_title(r'PEDH CV trajectories for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,COL], data["alpha_bkg"][ROW,COL], spc1.N_pulses))

fig.savefig("Photos/PEDH-sig-"+str(sig)+"-bkg-"+str(bkg)+"-mindist-"+str(sys.argv[3])+"-tmax-"+str(sys.argv[4])+"-bins-"+str(N_edhbins)+"-pulses-"+str(N_pulses)+".png")

fig1.savefig("Photos/PEDH-sig-"+str(sig)+"-bkg-"+str(bkg)+"-mindist-"+str(sys.argv[3])+"-tmax-"+str(sys.argv[4])+"-bins-"+str(N_edhbins)+"-pulses-"+str(N_pulses)+"-trajectory.png")

##as signal and background increase and decrease, the PEDH changes like the EDH changes.

##in terms of tracjectory:
##as signal increases the trajectory of each histogram bucket border becomes sharper and quicker, and the line itself as well as the arch have reduced variation, though the arch becomes larger
##as background increases the trajectory of each histogram bucket border becomes slower and more unpredictable, and the line and arch become far more varied and unpredictable 
#'''


pointsInGraph = int(((endOfGraph-startOfGraph)/intervalOfGraph))

data = [0 for _ in range(pointsInGraph+1)]
error = [0 for _ in range(pointsInGraph+1)]

for i in range(pointsInGraph+1):
    avgSum=float(0);
    for j in range(reps):
        avgSum+=main((las_il*(startOfGraph+(intervalOfGraph*i))),(las_il*(startOfGraph+(intervalOfGraph*i))))
    avgSum=avgSum/reps
    data[i]=(startOfGraph+(intervalOfGraph*i))
    error[i]=avgSum
    print("attenuation multiplier: "+str(data[i])+" error: "+str(error[i]))

    
plt.plot(data,error)
plt.show()
