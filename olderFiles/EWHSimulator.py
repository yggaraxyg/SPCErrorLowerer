from SPCSimLib.SPCSim.sensors import BaseEWHSPC
from SPCSimLib.SPCSim.utils import plot_transient, plot_ewh

##the code refused to work until I put these here
from SPCSimLib.SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSimLib.SPCSim.data_loaders.transient_loaders import TransientGenerator
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

##this I put in so I coudl easily run the thing multiple times from the terminal
import sys

if (len(sys.argv)!=3):
    print("Two arguments please! (laser illumination and background illumination)")
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

##this just gets the data from the simulation.
# Generate the per pixel data
data = PixLdr.get_data()


##this actually prepares the simulator for the photon shots
# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 2)

##this is more prep
# Using the get function to generate the transient
# for a given distance, albedo, intensity, and illumination condition
phi_bar = tr_gen.get_transient(data["gt_dist"],
                               data["albedo"],
                               data["albedo"],
                               data["alpha_sig"],
                               data["alpha_bkg"])

# Setting the dimensions, device, number of EWH bins per pixel
# and number of laser pulses

##this is for making the histogram. also, it defines how many pulses are sent
Nr, Nc, N_tbins = phi_bar.shape
device = PixLdr.device
N_ewhbins = 50
N_pulses = 500

##this does the simulation.
spc1 = BaseEWHSPC(
                Nr,
                Nc,
                N_pulses,
                device,
                N_tbins,
                N_ewhbins
               )

##this preps the data for graphing
captured_data = spc1.capture(phi_bar)
ewh_data = captured_data["ewh"].cpu().numpy()
phi_bar = phi_bar.cpu()

ewh_bins_axis = torch.linspace(0,N_tbins-N_tbins//N_ewhbins,N_ewhbins)

##this graphs all the data
fig, ax = plt.subplots(1,1,figsize=(8,4))
ROW, COL = [0,0]
plot_ewh(ax, ewh_bins_axis, ewh_data[ROW, COL,:], label = "EWH histogram", color = 'w')
plot_transient(ax, phi_bar[ROW, COL,:].numpy()*spc1.N_pulses, plt_type = '-r', label="True Transient")
ax.set_xlabel("Time (a.u.)")
ax.set_ylabel("Photon counts")
plt.legend()
fig.savefig("Photos/EWH-sig-"+str(sig)+"-bkg-"+str(bkg)+".png")

##as signal increases, the height of the arch of the line and height of the bar where the distance is increases,
##as background increases, the height of the arch of the line decreases and the height of all bars increases, though it does so so unevenly.
##it takes about a 1000-to-1 ratio of background illumination to signal illumination to make the arch of the line disappear
