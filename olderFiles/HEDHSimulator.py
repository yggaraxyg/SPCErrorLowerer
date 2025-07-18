from SPCSimLib.SPCSim.sensors import HEDHBaseClass
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

if (len(sys.argv)!=5):
    print("four arguments please! (laser illumination, background illumination, histogram bins and pulses)")
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


##this prepares for photon shots
# Creating transient generator with laser time period of 100ns, FWHM 1 and with
# laser time period divided into 1000 equal time-bins
tr_gen = TransientGenerator(N_tbins = 1000, tmax = PixLdr.tmax, FWHM = 2)


phi_bar = tr_gen.get_transient(data["gt_dist"],
                               data["albedo"],
                               data["albedo"],
                               data["alpha_sig"],
                               data["alpha_bkg"])


# Setting the dimensions, device, number of EWH bins per pixel
# and number of laser pulses

Nr, Nc, N_tbins = phi_bar.shape
device = PixLdr.device
N_edhbins = int(sys.argv[3])
N_pulses = int(sys.argv[4])

spc1 = HEDHBaseClass(Nr,
              Nc,
              N_pulses,
              device,
              N_tbins,
              N_edhbins,
              step_params={
                  "k":2,
                  "step_vals":[1],
              })

captured_data = spc1.capture(phi_bar)

pedh_data = captured_data["edh"]
gtedh_data = captured_data["gtedh"]
ewh_data = captured_data["ewh"]
edh_list = captured_data["traj"]
edh_list = np.array(edh_list)

ROW, COL = [0,0]

fig, ax = plt.subplots(1,1, figsize=(20,11))
ymax = ((torch.sum(ewh_data[ROW,COL,:])/N_edhbins)).item()
plot_edh(pedh_data[ROW,COL,:].cpu().numpy(),
         ax,
         ymax = ymax)
plot_edh(gtedh_data[ROW,COL,:].cpu().numpy(), ax,
         tr = phi_bar[ROW, COL,:].numpy()*spc1.N_pulses,
        #  crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax,
         ymax = ymax, ls='--')
ax.set_title(r'Final EDH boundaries for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,COL], data["alpha_bkg"][ROW,COL], spc1.N_pulses))
fig.savefig("Photos/HEDH-sig-"+str(sig)+"-bkg-"+str(bkg)+"-bins-"+str(N_edhbins)+"-pulses-"+str(N_pulses)+".png")

fig1, ax1 = plt.subplots(1,1, figsize=(20,11))
plot_edh_traj(ax1, edh_list, gtedh_data[ROW,COL,1:-1], ewh_data[0,0,:].cpu().numpy())
ax1.set_title(r'HEDH CV trajectories for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,COL], data["alpha_bkg"][ROW,COL], spc1.N_pulses))
fig1.savefig("Photos/HEDH-sig-"+str(sig)+"-bkg-"+str(bkg)+"-bins-"+str(N_edhbins)+"-pulses-"+str(N_pulses)+"-trajectory.png")


## as signal and background increase and decrease the HEDH changes like the EDH changes.

##in terms of trajectory:
##as signals increase the width of the arch the trajectory of the histograms move towards the center quicker.
##as the background increases the trajectory of the histograms becomes less sharp and both it and the arch become far more unpredictable.
