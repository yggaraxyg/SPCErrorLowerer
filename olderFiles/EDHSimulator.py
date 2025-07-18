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

if (len(sys.argv)!=5):
    print("four arguments please! (laser illumination, background illumination, pulses and bins)")
    exit(-1)
else:
    sig = float(sys.argv[1])
    bkg = float(sys.argv[2])

##this, as it states, simulates the SPC scene.
# Simulating results for distance = 0.1*dmax
PixLdr = PerPixelLoader(min_dist = 0.4,
                        tmax = 100,
                        sig_bkg_list = [
                            [sig,bkg]],
                        device = "cpu")

##this gets the scene data for the full simulation.
# Generate the per pixel data                                                                                        
data = PixLdr.get_data()


##once more, this prepares for photon shots
# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 2)

##this is even more prep for the full simulation.
# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(
                                data["gt_dist"],
                                data["albedo"],
                                data["albedo"],
                                data["alpha_sig"],
                                data["alpha_bkg"])


##this defines how many pulses are sent and how many bins are in the histogram.
# Setting the dimensions, device, number of EWH bins per pixel
# and number of laser pulses

Nr, Nc, N_tbins = phi_bar.shape
device = PixLdr.device
N_edhbins = int(sys.argv[3])
N_pulses = int(sys.argv[4])

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
##graph prep
ewh_data = captured_data["ewh"]
fig, ax = plt.subplots(1,1, figsize=(12,4))

ROW, COL = [0,0]

ymax = ((torch.sum(ewh_data[ROW,COL,:])/N_edhbins)).item()

plot_edh(oedh_data[ROW,COL,:],
         ax,
         ymax = ymax)

plot_edh(gtedh_data[ROW,COL,:], ax,
         tr = phi_bar[ROW, COL,:].numpy()*spc1.N_pulses,
         #crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax, # uncoment this line to zoom into peak
         ymax = ymax, ls='--')

ax.set_xlabel("Time (a.u.)")
ax.set_ylabel("Photon counts")
plt.legend()
##graph saving and info giving
fig.savefig("Photos/EDH-sig-"+str(sig)+"-bkg-"+str(bkg)+"-bins-"+str(N_edhbins)+"-pulses-"+str(N_pulses)+".png")
print("\rSig: "+str(sig)+" Bkg: "+str(bkg))

##as signal increases, the height of the arch increases and the width of the histogram buckets around the arch decrease.
##as background increases, the height of the arch decreases and the width of all histogram buckets equalises.
