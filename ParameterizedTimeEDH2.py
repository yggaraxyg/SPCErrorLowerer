from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
import matplotlib.pyplot as plt
from SPCSim.sensors.dtof import BaseEDHSPC
import torch
from SPCSim.utils.plot_utils import plot_transient, plot_edh
import numpy as np
import math
import sys
import matplotlib
matplotlib.use('TkAgg')

if (len(sys.argv)!=16):
    print("fifteen arguments please! (signal illumination, background illumination, start of attenuation multiplier, end of attenuation multiplier, interval of attenuation, tmax, time bins, FWHM  repetitions of each value, histogram bins, pulses, deadtime, FreeRun, and Multi)")
    exit(-1)

for i in sys.argv:
    print(i+" ",end="")
print()
    
las_il = float(sys.argv[1]) #0.5
bkg_il = float(sys.argv[2]) #5

startOfGraph = float(sys.argv[3]) #0
endOfGraph  = float(sys.argv[4]) #1
intervalOfGraph = float(sys.argv[5]) #0.1

tmaxp = float(sys.argv[6]) #100
tbinsp = int(sys.argv[7]) #1000
FWHMp = float(sys.argv[8]) #1.5-?
reps = int(sys.argv[9]) #10
distFraction = float(sys.argv[10]) #0.4
bins = int(sys.argv[11]) #16
puls = int(sys.argv[12]) #100000
deadtime = int(sys.argv[13]) #75
freerun = (int(sys.argv[14])>0) #true
multi = (int(sys.argv[15])>0) #true

def main(sig, bkg, cent, crep=0):
    # Simulating results for distance = 0.1*dmax
    PixLdr = PerPixelLoader(min_dist = cent,
                            tmax = tmaxp,
                            sig_bkg_list = [
                                [sig,bkg]],
                            device = "cpu")

    # Generate the per pixel data
    data = PixLdr.get_data()

    # Creating transient generator with laser time period of 100ns, FWHM 1 and with
    # laser time period divided into 1000 equal time-bins
    tr_gen = TransientGenerator(Nr = PixLdr.Nr, Nc = PixLdr.Nc, N_tbins = tbinsp, tmax = PixLdr.tmax, FWHM = FWHMp) #100 #3.5

    # Using the get function to generate the transient
    # for a given distance, albedo, intensity, and illumination condition
    phi_bar = tr_gen.get_transient(
        data["gt_dist"],
        data["albedo"],
        data["albedo"],
        data["alpha_sig"],
        data["alpha_bkg"])
    
    
    # Setting the dimensions, device, number of EWH bins per pixel
    # and number of laser pulses

    Nr, Nc, N_tbins = phi_bar.shape
    device = PixLdr.device
    N_edhbins = bins #16
    N_pulses = puls #100000

    spc1 = BaseEDHSPC(Nr, 
                      Nc,
                      N_pulses,
                      device,
                      N_tbins,
                      N_edhbins)

    # Define the SPAD dead time in nano seconds
    dead_time_ns = deadtime
    # Compute the size of a single time bin
    bin_size = tr_gen.tmax*1.0/tr_gen.N_tbins
    # Compute the dead time in number of bins from dead time in nano seconds
    dead_time_bins = int(dead_time_ns/bin_size)
    # Finally...
    # Make sure to pass dead_time_bins and not dead_time_ns to spc1.set_dead_time_bins()
    spc1.set_dead_time_bins(dead_time_bins)
    if(freerun):
        spc1.set_free_running_mode()
        
    captured_data = spc1.capture(phi_bar)
    oedh_data = captured_data["oedh"]
    gtedh_data = captured_data["gtedh"]

    ewh_data = captured_data["ewh"]

    fig, ax = plt.subplots(1,1, figsize=(12,4))
            
    ROW, COL = [0,0]

    norm_ewh = ewh_data[ROW,COL,:].numpy()
    norm_ewh = norm_ewh/np.sum(norm_ewh)
        
    norm_phi_bar = phi_bar[ROW, COL,:].numpy()
    norm_phi_bar = norm_phi_bar/np.sum(norm_phi_bar)

    # ymax = 1.0/N_edhbins
    ymax = max(norm_ewh.max(), norm_phi_bar.max())
        
    plot_edh(oedh_data[ROW,COL,:],
             ax,
             ymax = ymax)
    
    plot_transient(ax, norm_ewh, plt_type = '-k', label="%d-bin EWH"%N_tbins)
        
    plot_edh(gtedh_data[ROW,COL,:], ax,
             tr = norm_phi_bar,
             #  crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax, # uncoment this line to zoom into peak
             ymax = ymax, ls='--')
    plt.legend()
    #'''
    if(multi):    
        if(freerun):
            fig.savefig("TimePhotos/TimedEDH-FreeRun-sig-"+str("{:.4f}".format(sig))+"-bkg-"+str("{:.4f}".format(bkg))+"-tmax-"+str(tmaxp)+"-tbins-"+str(tbinsp)+"-FWHM-"+str("{:.1f}".format(FWHMp))+"-cent-"+str("{:.2f}".format(cent))+"-bins-"+str(bins)+"-pulses-"+str(puls)+"-deadtime-"+str(deadtime)+"-rep-"+str(crep)+".png")
        else:
            fig.savefig("TimePhotos/TimedEDH-NormRun-sig-"+str("{:.4f}".format(sig))+"-bkg-"+str("{:.4f}".format(bkg))+"-tmax-"+str(tmaxp)+"-tbins-"+str(tbinsp)+"-FWHM-"+str("{:.1f}".format(FWHMp))+"-cent-"+str("{:.2f}".format(cent))+"-bins-"+str(bins)+"-pulses-"+str(puls)+"-deadtime-"+str(deadtime)+"-rep-"+str(crep)+".png")
    else:
        if(freerun):
            fig.savefig("TimePhotos/TimedEDH-FreeRun-sig-"+str("{:.4f}".format(sig))+"-bkg-"+str("{:.4f}".format(bkg))+"-tmax-"+str(tmaxp)+"-tbins-"+str(tbinsp)+"-FWHM-"+str("{:.1f}".format(FWHMp))+"-cent-"+str("{:.2f}".format(cent))+"-bins-"+str(bins)+"-pulses-"+str(puls)+"-deadtime-"+str(deadtime)+".png")
        else:
            fig.savefig("TimePhotos/TimedEDH-NormRun-sig-"+str("{:.4f}".format(sig))+"-bkg-"+str("{:.4f}".format(bkg))+"-tmax-"+str(tmaxp)+"-tbins-"+str(tbinsp)+"-FWHM-"+str("{:.1f}".format(FWHMp))+"-cent-"+str("{:.2f}".format(cent))+"-bins-"+str(bins)+"-pulses-"+str(puls)+"-deadtime-"+str(deadtime)+".png")
    #'''
    matplotlib.pyplot.close()
    
    if(multi==True):    
        
        predictedDepthHistogramBorders=(oedh_data.numpy()[0][0])
        predictedWidthHistogramBorders=(ewh_data.numpy()[0][0])
        
        histogramPeakLocation=((data["gt_dist"].numpy()[0][0])/(0.15*tmaxp))*tbinsp

        smallestPrePred = 0
        smallestPostPred = tbinsp

        predBucket=0;

        for i in range(len(predictedDepthHistogramBorders)-1):
            if((predictedDepthHistogramBorders[i+1]-predictedDepthHistogramBorders[i])<(smallestPostPred-smallestPrePred)):
                smallestPrePred = predictedDepthHistogramBorders[i]
                smallestPostPred = predictedDepthHistogramBorders[i+1]
                predBucket=i;
        #'''
        print(str(predictedDepthHistogramBorders[predBucket])+" "+str((predictedDepthHistogramBorders[predBucket+1]+predictedDepthHistogramBorders[predBucket])/2)+" "+str(predictedDepthHistogramBorders[predBucket+1])+" "+str(histogramPeakLocation)+" "+str(((predictedDepthHistogramBorders[predBucket+1]+predictedDepthHistogramBorders[predBucket])/2)-(histogramPeakLocation)))
        #'''
        
        DepthErr = (((predictedDepthHistogramBorders[predBucket+1]+predictedDepthHistogramBorders[predBucket])/2)-(histogramPeakLocation))/tbinsp

        maxindex = 0
        for i in range(len(predictedWidthHistogramBorders)):
            if(predictedWidthHistogramBorders[maxindex]<predictedWidthHistogramBorders[i]):
                maxindex=i
                
        WidthErr = ((maxindex+1)-(histogramPeakLocation))/tbinsp
        
        return [DepthErr,WidthErr]
        
if(multi):
    pointsInGraph = int(((endOfGraph-startOfGraph)/intervalOfGraph))

    data = [0 for _ in range(pointsInGraph+1)]
    errorDepth = [0 for _ in range(pointsInGraph+1)]
    errorWidth = [0 for _ in range(pointsInGraph+1)]
    
    mindexDepth=0
    mindexWidth=0
    
    for i in range(pointsInGraph+1):
        avgDepthSum=float(0)
        avgWidthSum=float(0)
        data[i]=(startOfGraph+(intervalOfGraph*i))
        for j in range(reps):
            err=main((las_il*(startOfGraph+(intervalOfGraph*(i+1)))),(bkg_il*(startOfGraph+(intervalOfGraph*(i+1)))),distFraction,j+1)
            avgDepthSum+=err[0]
            avgWidthSum+=err[1]
            print("\r                                                                       ",end="")
            print("\rattenuation multiplier: "+str("{:.4f}".format(data[i]))+" rep: "+str(j+1)+"/"+str(reps))
        print(" "+str(avgDepthSum)+" "+str(avgWidthSum), end="")
        avgDepthSum=avgDepthSum/float(reps)
        avgWidthSum=avgWidthSum/float(reps)
        print(" "+str(avgDepthSum)+" "+str(avgWidthSum), end="")
        errorDepth[i]=avgDepthSum
        errorWidth[i]=avgWidthSum
        if(abs(errorDepth[i])<errorDepth[mindexDepth]):
            mindexDepth=i
        if(abs(errorWidth[i])<errorWidth[mindexWidth]):
            mindexWidth=i
        print(" "+str(distFraction))
        
    print("Minimum Datapoint: "+str("{:.4f}".format(data[mindexDepth]))+" Minimum ErrorDepth: "+str("{:.4f}".format(errorDepth[mindexDepth]))+" Minimum ErrorWidth: "+str("{:.4f}".format(errorWidth[mindexWidth])))
    print("")
    
    ##save file
    filestr = "RunData/free-"+str(freerun)+"-sig-"+str("{:.4f}".format(las_il))+"-bkg-"+str("{:.4f}".format(bkg_il))+"-tmax-"+str(tmaxp)+"-tbins-"+str(tbinsp)+"-FWHM-"+str("{:.1f}".format(FWHMp))+"-reps-"+str(reps)+"-cent-"+str("{:.2f}".format(distFraction))+"-bins-"+str(bins)+"-pulses-"+str(puls)+"-deadtime-"+str(deadtime)+".npz"
    open(filestr, 'w')
    np.savez(filestr, data=data, errorDepth=errorDepth, errorWidth=errorWidth)
    #filestr.close()

else:
    main(las_il, bkg_il, distFraction)
