from SPCSimLib.SPCSim.sensors import BaseEDHSPC
from SPCSimLib.SPCSim.utils import plot_transient, plot_ewh, plot_edh

##the code refused to work until I put these here                                                                    
from SPCSimLib.SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSimLib.SPCSim.data_loaders.transient_loaders import TransientGenerator
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

##this I put in so I could easily run the thing multiple times from the terminal                                     
import sys

if (len(sys.argv)!=11):
    print("ten arguments please! (laser illumination, background illumination, start of attenuation multiplier, end of attenuation multiplier, interval of attenuation, max distance, time bins, repetitions of each value, histogram bins, and pulses)")
    exit(-1)

las_il = float(sys.argv[1])
bkg_il = float(sys.argv[2])

startOfGraph = float(sys.argv[3])
endOfGraph  = float(sys.argv[4])
intervalOfGraph = float(sys.argv[5])

tmax = float(sys.argv[6])
tbins = int(sys.argv[7])
reps = int(sys.argv[8])
bins = int(sys.argv[9])
puls = int(sys.argv[10])


def main(sig, bkg, cent):
    ##this, as it states, simulates the SPC scene.
    # Simulating results for distance = 0.1*dmax
    PixLdr = PerPixelLoader(min_dist = cent,
                            tmax = int((tmax/15)*100),
                            sig_bkg_list = [
                                [sig,bkg]],
                            device = "cpu")

    ##this gets the scene data for the full simulation.
    # Generate the per pixel data
    data = PixLdr.get_data()


    ##once more, this prepares for photon shots
    # Creating transient generator with laser time period of 100ns, FWHM 1 and with
    # laser time period divided into 1000 equal time-bins
    tr_gen = TransientGenerator(N_tbins = tbins, tmax = PixLdr.tmax, FWHM = 2)

    ##this is even more prep for the full simulation.
    # Using the get function to generate the transient
    # for a given distance, albedo, intensity, and illumination condition
    phi_bar = tr_gen.get_transient(
        data["gt_dist"],
        data["albedo"],
        data["albedo"],
        data["alpha_sig"],
        data["alpha_bkg"])

    #print("gt_dist")
    #print(data["gt_dist"].numpy()[0][0])
    
    ##this defines how many pulses are sent and how many bins are in the histogram.
    # Setting the dimensions, device, number of EWH bins per pixel
    # and number of laser pulses
    Nr, Nc, N_tbins = phi_bar.shape
    device = PixLdr.device
    N_edhbins = bins
    N_pulses = puls
    
    ##this simualtes the SPC shots
    spc1 = BaseEDHSPC(Nr,
                      Nc,
                      N_pulses,
                      device,
                      N_tbins,
                      N_edhbins)

    ##these variables hold the data
    captured_data = spc1.capture(phi_bar)
    oedh_data = captured_data["oedh"]
    gtedh_data = captured_data["gtedh"]


    predictedHistogramBorders=(oedh_data.numpy()[0][0])
    groundTruthHistogramBorders=(gtedh_data.numpy()[0][0])
    histogramPeakLocation=(data["gt_dist"].numpy()[0][0])
    
    '''
    print("oedh")
    print(oedh_data)
    print("gtedh")
    print(gtedh_data)
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
            trueBucket=i
    
    #print("Predicted bucket: "+str(predBucket)+" True Bucket: "+str(trueBucket))
    #return (abs(smallestPostPred-smallestPostTrue)+abs(smallestPrePred-smallestPreTrue))
    #return (abs(predBucket-trueBucket)) 
    #return (abs(((predictedHistogramBorders[predBucket+1]+predictedHistogramBorders[predBucket])/2)-((groundTruthHistogramBorders[trueBucket+1]+groundTruthHistogramBorders[trueBucket])/2)))
    return abs(((predictedHistogramBorders[predBucket+1]+predictedHistogramBorders[predBucket])/2)-((histogramPeakLocation/tmax)*tbins))#/tbins
    #return ((predictedHistogramBorders[predBucket+1]+predictedHistogramBorders[predBucket])/2)
    #return ((histogramPeakLocation/tmax)*tbins)

pointsInGraph = int(((endOfGraph-startOfGraph)/intervalOfGraph))

data = [0 for _ in range(pointsInGraph)]
error = [0 for _ in range(pointsInGraph)]

mindex=0;

for i in range(pointsInGraph):
    avgSum=float(0);
    data[i]=(startOfGraph+(intervalOfGraph*i))
    #'''
    for j in range(reps):
        #print("\rattenuation multiplier: "+str("{:.3f}".format(data[i]))+" rep: "+str(j+1)+"/"+str(reps),end="")
        avgSum+=main((las_il*(startOfGraph+(intervalOfGraph*(i+1)))),(bkg_il*(startOfGraph+(intervalOfGraph*i+1))),(float(j+1)/reps))
        print("\r                                                                       ",end="")
        print("\rattenuation multiplier: "+str("{:.4f}".format(data[i]))+" rep: "+str(j+1)+"/"+str(reps),end="")
    #'''

    avgSum=avgSum/reps
    print(" "+str(avgSum))
    error[i]=avgSum
    if(error[i]<error[mindex]):
        mindex=i

print("Minimum Datapoint: "+str("{:.4f}".format(data[mindex]))+" Minimum Error: "+str("{:.4f}".format(error[mindex])))
print("")

plt.plot(data,error)
plt.show()
