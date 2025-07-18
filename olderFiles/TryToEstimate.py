from SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.utils.plot_utils import plot_transient, plot_ewh, plot_edh, plot_edh_traj
import matplotlib.pyplot as plt
from SPCSim.sensors.dtof import RawSPC, BaseEWHSPC, BaseEDHSPC, HEDHBaseClass, PEDHBaseClass, PEDHOptimized
from SPCSim.postproc.ewh_postproc import PostProcEWH
from SPCSim.postproc.edh_postproc import PostProcEDH
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import numpy as np

def getError(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, val):

    #SCENE PROPERTIES!#
    
    num_unique_distance_values = 10 # @param {type:"slider", min:1, max:1000, step:1}
    min_distance_fraction = 0.25 # @param {type:"slider", min:0, max:0.9, step:0.05}
    max_distance_fraction = 1 # @param {type:"slider", min:0.2, max:1, step:0.05}
    signal_bkg_illumination_combinations = [[1,5]] # @param {type:"raw"}
    num_independent_simulation_runs_per_combination = 11 # @param {type:"slider", min:1, max:1000, step:1}
    device = 'cpu' # @param ["cpu", "cuda"]

    #LASER PROPERTIES!#
    laser_time_period_ns = 100.0 # @param {type:"number"}
    num_time_bins = 1000 # @param {type:"integer"}
    laser_FWHM_ns = 2 # @param {type:"number"}

    #SENSOR PROPERTIES!#
    laser_time_period_ns = 100.0 # @param {type:"number"}
    num_time_bins = 1000 # @param {type:"integer"}
    laser_FWHM_ns = 2 # @param {type:"number"}

    #SELECTING EXPERIMENTAL INDEX TO PLOT RESULTS!#

    illumination_condition_index = 0 # @param {type:"integer"}
    distance_value_index = 0 # @param {type:"integer"}
    independent_run_index = 0 # @param {type:"integer"}

    ROW = PixLdr.get_row(sbr_idx = illumination_condition_index,dist_idx = distance_value_index)
    COL = independent_run_index

    ################################################################################
    ########## Setting the 3D scene parameters #####################################
    ################################################################################

    # Simulating results for distance = 0.1*dmax
    PixLdr = PerPixelLoader(
                        num_dists=num_unique_distance_values,
                        min_dist = min_distance_fraction,
                        max_dist = max_distance_fraction,
                        tmax = laser_time_period_ns,
                        sig_bkg_list = signal_bkg_illumination_combinations,
                        num_runs=num_independent_simulation_runs_per_combination,
                        device = device)

    # Generate the per pixel data
    data = PixLdr.get_data()

    
    ################################################################################
    ### Generating the transient for set scene conditions and laser parameters #####
    ################################################################################

    # Creating transient generator with laser time period of 100ns, FWHM 1 and with
    # laser time period divided into 1000 equal time-bins
    tr_gen = TransientGenerator(Nr = PixLdr.Nr,
                                Nc = PixLdr.Nc,
                                N_tbins = num_time_bins,
                                tmax = PixLdr.tmax,
                                FWHM = laser_FWHM_ns)


    # Using the get function to generate the transient
    # for a given distance, albedo, intensity, and illumination condition
    phi_bar = tr_gen.get_transient(data["gt_dist"],
                                   data["albedo"],
                                   data["albedo"],
                                   data["alpha_sig"],
                                   data["alpha_bkg"])

    Nr, Nc, N_tbins = phi_bar.shape
    device = PixLdr.device


    ################################################################################
    ########## Initialize object for selected sensor class #########################
    ################################################################################

    
    SensorClass = sensor_id_dict[sensor_id]

    if sensor_id == "RawSPC":
        spc1 = SensorClass(
            Nr,
            Nc,
            num_laser_pulses,
            device,
            num_time_bins,
            num_output_timestamps)
    else:
        spc1 = SensorClass(
            Nr,
            Nc,
            num_laser_pulses,
            device,
            num_time_bins,
            num_histogram_bins)


    ################################################################################
    ########## Capture the dToF data for given exposure time #######################
    ################################################################################
    
    
    captured_data = spc1.capture(phi_bar)
    
    fig, ax1 = plt.subplots(1,1,figsize=(8,4))
    
    
    ################################################################################
    ########## Plot the data captured by the sensor data ###########################
    ##########     &    ############################################################
    ######## Reconstr 3D distance estimates using post-processing pipeline #########
    ################################################################################

    
    if sensor_id == "RawSPC":
        raw_data = captured_data["time_stamps"]
        ewh_data = captured_data["ewh"]
    
        phi_bar1 = phi_bar[ROW, COL, :].cpu().numpy()
        ts = raw_data[ROW, COL, :].cpu().numpy().flatten()
        
        xaxis = torch.arange(0.5,1+N_tbins).to(torch.float)
        hist,_ = torch.histogram(raw_data[ROW,COL,:], xaxis)
        hist2 = ewh_data[ROW,COL,:]
        plot_transient(ax1, hist2.cpu().numpy(), plt_type = '-b', label="Captured EW histogram")
        plot_transient(ax1, hist.cpu().numpy(), plt_type = '--r', label="Timestamps histogram")
        plot_transient(ax1, phi_bar[ROW,COL,:].cpu().numpy()*spc1.N_output_ts/np.mean(np.sum(phi_bar.cpu().numpy(), axis=-1)), plt_type = '-g', label="True Transient")
        ax1.set_xlabel('Bins')
        ax1.set_ylabel('Frequency')
        ax1.set_title(r'Histogram of raw data for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f'%(data["alpha_sig"][ROW, COL], data["alpha_bkg"][ROW, COL]))
        ax1.legend()
    elif sensor_id == "BaseEWHSPC":

        ewh_data = captured_data["ewh"].cpu().numpy()
        phi_bar = phi_bar.cpu().numpy()

        ewh_bins_axis = torch.linspace(0,N_tbins-N_tbins//num_histogram_bins,num_histogram_bins)
        
        plot_ewh(ax1, ewh_bins_axis, ewh_data[ROW, COL,:], label = "EWH histogram", color = 'w')
        plot_transient(ax1, phi_bar[ROW, COL,:]*spc1.N_pulses, plt_type = '-r', label="True Transient")
        ax1.set_xlabel("Time (a.u.)")
        ax1.set_ylabel("Photon counts")
        ax1.set_title(r'%d-bin Equi-depth histogram for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f'%(num_histogram_bins, data["alpha_sig"][ROW, COL], data["alpha_bkg"][ROW, COL]))
        plt.legend()

        postproc_ewh = PostProcEWH(
            Nr,
            Nc,
            num_time_bins,
            laser_time_period_ns,
            device
        )

        dist_idx, pred_dist = postproc_ewh.ewh2depth_t(captured_data["ewh"])
        fig2, ax2 = plt.subplots(1,2, figsize=(8,8))
        im = ax2[0].imshow(data["gt_dist"].cpu().numpy(), cmap = 'jet')
        ax2[0].axis('off')
        ax2[0].set_title("True Dist")
        divider = make_axes_locatable(ax2[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(length=0)
        cbar.set_label('Distance (m)', rotation=270, labelpad=15)
        im = ax2[1].imshow(pred_dist, cmap = 'jet')
        ax2[1].axis('off')
        ax2[1].set_title("Pred. Dist")
        divider = make_axes_locatable(ax2[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(length=0)
        cbar.set_label('Distance (m)', rotation=270, labelpad=15)
        
        fig2.suptitle(r'Distance estimates for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,RUN], data["alpha_bkg"][ROW,RUN], spc1.N_pulses))
        fig2.savefig("DistanceOutput.png")
    elif sensor_id == "BaseEDHSPC":
        oedh_data = captured_data["oedh"].cpu().numpy()
        gtedh_data = captured_data["gtedh"].cpu().numpy()
        ewh_data = captured_data["ewh"].cpu().numpy()
        phi_bar = phi_bar.cpu().numpy()
        
        ymax = ((np.sum(ewh_data[ROW,COL,:])/num_histogram_bins)).item()
        
        plot_edh(oedh_data[ROW,COL,:],
                 ax1,
                 ymax = ymax)

        plot_edh(gtedh_data[ROW,COL,:], ax1,
        tr = phi_bar[ROW, COL,:]*spc1.N_pulses,
        #  crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax, # uncoment this line to zoom into peak
        ymax = ymax, ls='--')
        ax1.set_title(r'Final %d-bin Oracle EDH boundaries for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(num_histogram_bins,
                                                                                                                     data["alpha_sig"][ROW,COL],
                                                                                                                     data["alpha_bkg"][ROW,COL],
                                                                                                                     spc1.N_pulses))


    elif sensor_id in ["HEDHBaseClass","PEDHBaseclass", "PEDHOptimized"]:
        pedh_data = captured_data["edh"].cpu().numpy()
        gtedh_data = captured_data["gtedh"].cpu().numpy()
        ewh_data = captured_data["ewh"].cpu().numpy()
        edh_list = captured_data["traj"]
        phi_bar = phi_bar.cpu().numpy()
        edh_list = np.array(edh_list)
        
        ymax = ((np.sum(ewh_data[ROW,COL,:])/num_histogram_bins)).item()
        plot_edh(pedh_data[ROW,COL,:],
                 ax1,
                 ymax = ymax)
        plot_edh(gtedh_data[ROW,COL,:], ax1,
                 tr = phi_bar[ROW, COL,:]*spc1.N_pulses,
                 #  crop_window= tr_gen.FWHM*1.5*tr_gen.N_tbins*1.0/tr_gen.tmax,
                 ymax = ymax, ls='--')
        ax1.set_title(r'Final EDH boundaries for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,COL], data["alpha_bkg"][ROW,COL], spc1.N_pulses))
        # fig.savefig("Temp.png")
        
        fig_, ax_ = plt.subplots(1,1, figsize=(8,4))
        plot_edh_traj(ax_, edh_list, gtedh_data[ROW,COL,1:-1], ewh_data[ROW,COL,:])
        ax1.set_title(r'EDH CV trajectories for $\Phi_{sig}$ = %.2f, $\Phi_{bkg}$ = %.2f, %d pulses'%(data["alpha_sig"][ROW,COL], data["alpha_bkg"][ROW,COL], spc1.N_pulses))
        plt.plot()
        
    else:
        print("Incorrect Sensor choice")
    
    return 0

def findBestInGroup(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, minVal, maxVal, sections, avgpoints, MaxLoops, Loops):
    #ONE MOMENT!
    #this is a rescursive function meant to figure out what value between Max and Min gives the minimum error. it does this by divinig the area between max and min into evenly-spaced sections, finding the error of the middle point in each section, finding the point with the lowest error, taking that point's lowest-error neighbor (neighbor meaning point with value already determined one section before or after one point), and performing the sampe operation with those two points as max and min (determined by whichever is lower and higher respectively) then recursing until desired number of recursions are performed. once desired number is perfromed, we take the midpoint of the area beteween max and min and call it sufficient.
    print("Recursion Number: "+loops+" Min: "+minVal+" Max: "+maxVal)
    if(maxLoops<=Loops):
        ErAt = getError(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, ((min+max)/2));
        return [((min+max)/2), ErAt]
    newMin=minVal
    newMax=maxVal
    ErrorOfSection  = [0 for _ in range(sections)]
    ValOfSection = [0 for _ in range(sections)]
    mindex=0
    for i in range(sections):
        ValOfSection[i]= (((maxVal-minVal)/sections)*(i+0.5))+minVal
        avg = getError(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, ValOfSection[i])
        for j in range(avgpoints):
            avg = avg + getError(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, (ValOfSection[i]+((j+1)*((maxVal-minVal)/(sections*2*avgpoints)))))
            avg = avg + getError(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, (ValOfSection[i]-((j+1)*((maxVal-minVal)/(sections*2*avgpoints)))))
        avg=avg/((2*avgpoints)+1)
        ErrorOfSection[i] = avg
        if(ErrorOfSection[i]<ErrorOfSection[mindex]):
            mindex=i
    
    if(mindex==0):
        newMin=ValOfSection[0]
        newMax=ValOfSection[1]
    elif(mindex==sections-1):
        newMin=ValOfSection[sections-2]
        newMax=ValOfSection[sections-1]
    else:
        if(ErrorOfSection[mindex-1] < ErrorOfSection [mindex+1]):
            newMin=ValOfSection[mindex-1]
            newMax=ValOfSection[mindex]
        else:
            newMin=ValOfSection[mindex]
            newMax=ValOfSection[mindex+1]

    return findBestInGroup(LUMEN, DIST_NUM, DIST_MIN, DIST_MAX, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, newMin, newMax, sections, avgpoints, MaxLoops, Loops+1);
    
def main():
    LUMEN_MIN = 0
    LUMEN_MAX = 10000
    LUMEN_INT = 1

    DIST_NUM_MIN = 0
    DIST_NUM_MAX = 10
    DIST_NUM_INT = 1
    
    DIST_MIN_MIN = 0
    DIST_MIN_MAX = 1
    DIST_MIN_INT = 0.01
    
    DIST_MAX_MIN = 0
    DIST_MAX_MAX = 1
    DIST_MAX_INT = 0.01
    
    NUMBER_OF_RUNS_MIN = 1
    NUMBER_OF_RUNS_MAX = 100
    NUMBER_OF_RUNS_INT = 1
    
    LASER_REPETITION_PERIOD_MIN = 1
    LASER_REPETITION_PERIOD_MAX = 10000
    LASER_REPETITION_PERIOD_INT = 1
    
    LASER_TIME_PERIOD_BINS_MIN = 1
    LASER_TIME_PERIOD_BINS_MAX = 10000
    LASER_TIME_PERIOD_BINS_INT = 1
    
    LASER_FWHM_MIN = 1
    LASER_FWHM_MAX = 10
    LASER_FWHM_INT = 1
    
    LUMEN_BOX = int((LUMEN_MAX-LUMEN_MIN)/LUMEN_INT)
    DIST_NUM_BOX = int((DIST_NUM_MAX-DIST_NUM_MIN)/DIST_NUM_INT)
    DIST_MIN_BOX = int((DIST_MIN_MAX-DIST_MIN_MIN)/DIST_MIN_INT)
    DIST_MAX_BOX = int((DIST_MAX_MAX-DIST_MAX_MIN)/DIST_MAX_INT)
    NUMBER_OF_RUNS_BOX = int((NUMBER_OF_RUNS_MAX-NUMBER_OF_RUNS_MIN)/NUMBER_OF_RUNS_INT)
    LASER_REPETITION_PERIOD_BOX = int((LASER_REPETITION_PERIOD_MAX-LASER_REPETITION_PERIOD_MIN)/LASER_REPETITION_PERIOD_INT)
    LASER_TIME_PERIOD_BINS_BOX = int((LASER_TIME_PERIOD_BINS_MAX-LASER_TIME_PERIOD_BINS_MIN)/LASER_TIME_PERIOD_BINS_INT)
    LASER_FWHM_BOX = int((LASER_FWHM_MAX-LASER_FWHM_MIN)/LASER_FWHM_INT)

    optimalBlockage = [[[[[[[[0 for _ in range(LUMEN_BOX)] for _ in range(DIST_NUM_BOX)] for _ in range(DIST_MIN_BOX)] for _ in range(DIST_MAX_BOX)] for _ in range(NUMBER_OF_RUNS_BOX)] for _ in range(LASER_REPETITION_PERIOD_BOX)] for _ in range(LASER_TIME_PERIOF_BINS_BOX)] for _ in range(LASER_FWHM_BOX)]
    errorAtOptimal = [[[[[[[[0 for _ in range(LUMEN_BOX)] for _ in range(DIST_NUM_BOX)] for _ in range(DIST_MIN_BOX)] for _ in range(DIST_MAX_BOX)] for _ in range(NUMBER_OF_RUNS_BOX)] for _ in range(LASER_REPETITION_PERIOD_BOX)] for _ in range(LASER_TIME_PERIOF_BINS_BOX)] for _ in range(LASER_FWHM_BOX)]

    
    for LUMEN in range(LUMEN_BOX):
        for DIST_NUM in range(DIST_NUM_BOX):
            for DIST_MIN in range(DIST_MIN_BOX):
                for DIST_MAX in range(DIST_MAX_BOX):
                    for NUMBER_OF_RUNS in range(NUMBER_OF_RUNS_BOX):
                        for LASER_REPETITION_PERIOD in range(LASER_REPETITION_PERIOD_BOX):
                            for LASER_TIME_PERIOD_BINS in range(LASER_TIME_PERIOD_BINS_BOX):
                                for LASER_FWHM in range(LASER_FWHM_BOX):
                                    optimal = findBestInGroup((LUMEN_MIN+(LUMEN_INT*LUMEN)),(DIST_NUM_MIN+(DIST_NUM_INT*DIST_NUM)),(DIST_MIN_MIN+(DIST_MIN_INT*DIST_MIN)),(DIST_MAX_MIN+(DIST_MAX_INT*DIST_MAX)),(NUMBER_OF_RUNS_MIN+(NUMBER_OF_RUNS_INT*NUMBER_OF_RUNS)),(LASER_REPETITION_PERIOD_MIN+(LASER_REPETITION_PERIOD_INT*LASER_REPETITION_PERIOD)),(LASER_TIME_PERIOD_BINS_MIN+(LASER_TIME_PERIOD_BINS_INT*LASER_TIME_PERIOD_BINS)),(LASER_FWHM_MIN+(LASER_FWHM_INT*LASER_FWHM)), 0, 100, 10, 5, 6, 0)
                                    optimalBlockage[LUMEN][DIST_NUM][DIST_MIN][DIST_MAX][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM] = optimal[0]
                                    errorAtOptimal[LUMEN][DIST_NUM][DIST_MIN][DIST_MAX][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM] = optimal[1]
                                    print("LUMEN: "+(LUMEN_MIN+(LUMEN_INT*LUMEN))+" DIST NUM: "+(DIST_NUM_MIN+(DIST_NUM_INT*DIST_NUM))+" DIST MIN: "+(DIST_MIN_MIN+(DIST_MIN_INT*DIST_MIN))+" DIST MAX: "+(DIST_MAX_MIN+(DIST_MAX_INT*DIST_MAX))+" NUMBER OF RUNS: "+(NUMBER_OF_RUNS_MIN+(NUMBER_OF_RUNS_INT*NUMBER_OF_RUNS))+" LASER REPETITION PERIOD: "+(LASER_REPETITION_PERIOD_MIN+(LASER_REPETITION_PERIOD_INT*LASER_REPETITION_PERIOD))+" LASER TIME PERIOD BINS: "+(LASER_TIME_PERIOD_BINS_MIN+(LASER_TIME_PERIOD_BINS_INT*LASER_TIME_PERIOD_BINS))+" LASER FWHM: "+(LASER_FWHM_MIN+(LASER_FWHM_INT*LASER_FWHM))+" Optimal Blockage: "+optimalBlockage[LUMEN][DIST_NUM][DIST_MIN][DIST_MAX][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM]+" Error at Optimal: "+errorAtOptimal[LUMEN][DIST_NUM][DIST_MIN][DIST_MAX][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM]+".")
    print("That's all folks!")


print(getError(0, 0, 0, 0, 0, 0, 0, 0, 0));
