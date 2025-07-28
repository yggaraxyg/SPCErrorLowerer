from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
import matplotlib.pyplot as plt
from SPCSim.sensors.dtof import BaseEDHSPC
import torch
from SPCSim.utils.plot_utils import plot_transient, plot_edh
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')

# Simulating results for distance = 0.1*dmax
PixLdr = PerPixelLoader(min_dist = 0.4,
                        tmax = 100,
                        sig_bkg_list = [
                            [0.5,5]],
                        device = "cpu")

# Generate the per pixel data
data = PixLdr.get_data()


# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(Nr = PixLdr.Nr, Nc = PixLdr.Nc, N_tbins = 100, tmax = PixLdr.tmax, FWHM = 3.5)


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
N_edhbins = 16
N_pulses = 10000

spc1 = BaseEDHSPC(Nr, 
                  Nc,
                  N_pulses,
                  device,
                  N_tbins,
                  N_edhbins)

dead_time_ns = 75
spc1.set_dead_time_bins(dead_time_ns)
#spc1.set_free_running_mode()

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

ymax = 1.0/N_edhbins

plot_edh(oedh_data[ROW,COL,:],
         ax,
         ymax = ymax)

plot_transient(ax, norm_ewh, plt_type = '-k', label="%d-bin EWH"%N_tbins)

plot_edh(gtedh_data[ROW,COL,:], ax,
         tr = norm_phi_bar,
        #  crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax, # uncoment this line to zoom into peak
         ymax = ymax, ls='--')
plt.legend()

fig.savefig("Temp2.png")

