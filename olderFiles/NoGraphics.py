from SPCSimLib.SPCSim.data_loaders.perpixel_loaders import PerPixelLoader
from SPCSimLib.SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSimLib.SPCSim.sensors.dtof import RawSPC, BaseEWHSPC, BaseEDHSPC, HEDHBaseClass, PEDHBaseClass, PEDHOptimized
from SPCSimLib.SPCSim.postproc.ewh_postproc import PostProcEWH
from SPCSimLib.SPCSim.postproc.edh_postproc import PostProcEDH
import numpy as np

import matplotlib
from matplotlib import pyplot
#matplotlib.use('Agg')
matplotlib.use('TkAgg')

def getError(DIST_NUM, DIST_MIN, DIST_MAX, LASER_PHOTON, BACKGROUND_PHOTON, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, NUM_LASER_PULSES, NUM_HISTOGRAM_BINS, val):

    #SCENE PROPERTIES!#

    #print("Laser Photon: "+str("{:.5f}".format(LASER_PHOTON))+" Background Photon: "+str("{:.5f}".format(BACKGROUND_PHOTON))+" Attenuation: "+str("{:.5f}".format(1-val))+" Adjusted Laser Photon: "+str("{:.5f}".format(LASER_PHOTON*(1-val)))+" Adjusted Background Photon: "+str("{:.5f}".format(BACKGROUND_PHOTON*(1-val))))
    num_unique_distance_values = DIST_NUM # @param {type:"slider", min:1, max:1000, step:1}
    min_distance_fraction = DIST_MIN # @param {type:"slider", min:0, max:0.9, step:0.05}
    max_distance_fraction = DIST_MAX # @param {type:"slider", min:0.2, max:1, step:0.05}
    signal_bkg_illumination_combinations = [[LASER_PHOTON*(1-val),BACKGROUND_PHOTON*(1-val)]] # @param {type:"raw"}
    num_independent_simulation_runs_per_combination = NUMBER_OF_RUNS # @param {type:"slider", min:1, max:1000, step:1}
    device = 'cpu' # @param ["cpu", "cuda"]

    #LASER PROPERTIES!#
    laser_time_period_ns = LASER_REPETITION_PERIOD # @param {type:"number"}
    num_time_bins = LASER_TIME_PERIOD_BINS # @param {type:"integer"}
    laser_FWHM_ns = LASWER_FWHM # @param {type:"number"}

    #SENSOR PROPERTIES!#
    sensor_id = "BaseEDHSPC" # @param ["RawSPC", "BaseEWHSPC", "BaseEDHSPC", "HEDHBaseClass", "PEDHBaseClass", "PEDHOptimized"]
    num_laser_pulses = NUM_LASER_PULSES # @param {type:"number"}
    num_histogram_bins = NUM_HISTOGRAM_BINS # @param {type:"number"}
    num_output_timestamps = NUM_HISTOGRAM_BINS
    
    sensor_id_dict = {
        "RawSPC": RawSPC,
        "BaseEWHSPC": BaseEWHSPC,
        "BaseEDHSPC": BaseEDHSPC,
        "HEDHBaseClass": HEDHBaseClass,
        "PEDHBaseClass": PEDHBaseClass,
        "PEDHOptimized": PEDHOptimized
    }

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
    
    SensorClass = sensor_id_dict[sensor_id]

    if(sensor_id!="RawSPC"):
        spc1 = SensorClass(
            Nr,
            Nc,
            num_laser_pulses,
            device,
            num_time_bins,
            num_histogram_bins)
    else:
        spc1 = SensorClass(
            Nr,
            Nc,
            num_laser_pulses,
            device,
            num_time_bins,
            num_output_timestamps)
    
    captured_data = spc1.capture(phi_bar)

    amtOfError=0;
    if sensor_id == "BaseEWHSPC":
        
        postproc_ewh = PostProcEWH(
            Nr,
            Nc,
            num_time_bins,
            laser_time_period_ns,
            device
        )
        
        pred_dist = postproc_ewh.ewh2depth_t(captured_data["ewh"])[1]

        '''
        try:
            print(data["gt_dist"].cpu().numpy())
            print(pred_dist.numpy())
        except:
            print("_0_")
            print(" | ")
            print("/ \\")
            print(" ----------------------------------------")
            print("(If you can see me, something went wrong.)")
            print(" ---------------------------------------- ")
        '''
        
        ErrBox=abs(data["gt_dist"].cpu().numpy()-pred_dist.numpy())

        #print("From this point on, I am the coder now")
        for i in range(len(ErrBox)):
            for j in range(len(ErrBox[i])):
                #print("i: "+str(i)+" j: "+str(j)+" Real: "+str(real[i][j])+" Expected: "+str(expected[i][j])+" Dif: "+str(real[i][j]-expected[i][j]))
                amtOfError+=ErrBox[i][j]

    elif(sensor_id == "BaseEDHSPC"):

        postproc_edh = PostProcEDH(
            Nr,
            Nc,
            num_time_bins,
            laser_time_period_ns,
            device
        )

        #messy_data = postproc_edh.edh2depth_t(captured_data["oedh"])[3].numpy()
        oedh_data = captured_data["oedh"].cpu().numpy()
        gtedh_data = captured_data["gtedh"].cpu().numpy()

        '''
        print("CORNFLAKES!")
        print(oedh_data)
        print("True:")
        print(gtedh_data)
        '''
        
        ErrBox = abs(gtedh_data-oedh_data)
        
        for i in range(len(gtedh_data)):
            for j in range(len(gtedh_data[i])):
                for k in range(len(gtedh_data[i][j])):
                    #print("i: "+str(i)+" j: "+str(j)+" k: "+str(k)+" Real: "+str(gtedh_data[i][j][k])+" Expected: "+str(oedh_data[i][j][k])+" Dif: "+str(gtedh_data[i][j][k]-oedh_data[i][j][k]))
                    amtOfError+=ErrBox[i][j][k]
        #print("Length: "+str(len(gtedh_data))+" "+str(len(gtedh_data[0]))+" "+str(len(gtedh_data[0][0])))
        #'''
    elif sensor_id == "RawSPC":
        raw_data = captured_data["time_stamps"]
        ewh_data = captured_data["ewh"]
        print(raw_data)
        print(ewh_data)

        return -1
    else:
        print("Incorrect Sensor choice")
    
    return amtOfError

def findBestInGroup(DIST_NUM, DIST_MIN, DIST_MAX, LASER_PHOTON, BACKGROUND_PHOTON, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, NUM_LASER_PULSES, NUM_HISTOGRAM_BINS, precisionDigits):
    mindex = 0
    minVal = 1
    curval = 0
    for i in range(10^precisionDigits):
        curVal =getError(DIST_NUM, DIST_MIN, DIST_MAX, LASER_PHOTON, BACKGROUND_PHOTON, NUMBER_OF_RUNS, LASER_REPETITION_PERIOD, LASER_TIME_PERIOD_BINS, LASWER_FWHM, NUM_LASER_PULSES, NUM_HISTOGRAM_BINS, (float(i)/(10^precisionDigits)))
        if(curVal<minVal):
            minVal=curVal
            mindex=i
    return [mindex , minval];

def main():
    
    DIST_NUM_MIN = 1
    DIST_NUM_MAX = 1000
    DIST_NUM_INT = 1
    
    DIST_MIN_MIN = 0
    DIST_MIN_MAX = 0.9
    DIST_MIN_INT = 0.05
    
    DIST_MAX_MIN = 0.2
    DIST_MAX_MAX = 1
    DIST_MAX_INT = 0.05

    LASER_PHOTON_MIN = 1;
    LASER_PHOTON_MAX = 100;
    LASER_PHOTON_INT = 1;

    BACKGROUND_PHOTON_MIN = 1;
    BACKGROUND_PHOTON_MAX = 100;
    BACKGROUND_PHOTON_INT = 1;
    
    NUMBER_OF_RUNS_MIN = 1
    NUMBER_OF_RUNS_MAX = 1000
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

    NUM_LASER_PULSES_MIN=100
    NUM_LASER_PULSES_MAX=10000
    NUM_LASER_PULSES_INT=100

    NUM_HISTOGRAM_BINS_MIN=1
    NUM_HISTOGRAM_BINS_MAX=100
    NUM_HISTOGRAM_BINS_INT=1;
    
    DIST_NUM_BOX = int((DIST_NUM_MAX-DIST_NUM_MIN)/DIST_NUM_INT)
    DIST_MIN_BOX = int((DIST_MIN_MAX-DIST_MIN_MIN)/DIST_MIN_INT)
    DIST_MAX_BOX = int((DIST_MAX_MAX-DIST_MAX_MIN)/DIST_MAX_INT)
    LASER_PHOTON_BOX = int((LASER_PHOTON_MAX-LASER_PHOTON_MIN)/LASER_PHOTON_INT)
    BACKGROUND_PHOTON_BOX = int((BACKGROUND_PHOTON_MAX-BACKGROUND_PHOTON_MIN)/BACKGROUND_PHOTON_INT)
    NUMBER_OF_RUNS_BOX = int((NUMBER_OF_RUNS_MAX-NUMBER_OF_RUNS_MIN)/NUMBER_OF_RUNS_INT)
    LASER_REPETITION_PERIOD_BOX = int((LASER_REPETITION_PERIOD_MAX-LASER_REPETITION_PERIOD_MIN)/LASER_REPETITION_PERIOD_INT)
    LASER_TIME_PERIOD_BINS_BOX = int((LASER_TIME_PERIOD_BINS_MAX-LASER_TIME_PERIOD_BINS_MIN)/LASER_TIME_PERIOD_BINS_INT)
    LASER_FWHM_BOX = int((LASER_FWHM_MAX-LASER_FWHM_MIN)/LASER_FWHM_INT)
    NUM_LASER_PULSES_BOX = int((NUM_LASER_PULSES_MAX-NUM_LASER_PULSES_MIN)/NUM_LASER_PULSES_INT)
    NUM_HISTOGRAM_BINS_BOX = int((NUM_HISTOGRAM_BINS_MAX-NUM_HISTOGRAM_BINS_MIN)/NUM_HISTOGRAM_BINS_INT)
    
    optimalBlockage = [[[[[[[[[[[0 for _ in range(DIST_NUM_BOX)] for _ in range(DIST_MIN_BOX)] for _ in range(DIST_MAX_BOX)] for _ in range(LASER_PHOTON_BOX)] for _ in range(BACKGROUND_PHOTON_BOX)] for _ in range(NUMBER_OF_RUNS_BOX)] for _ in range(LASER_REPETITION_PERIOD_BOX)] for _ in range(LASER_TIME_PERIOF_BINS_BOX)] for _ in range(LASER_FWHM_BOX)] for _ in range(NUM_LASER_PULSES_BOX)] for _ in range(NUM_HISTOGRAM_BINS_BOX)]
    errorAtOptimal  = [[[[[[[[[[[0 for _ in range(DIST_NUM_BOX)] for _ in range(DIST_MIN_BOX)] for _ in range(DIST_MAX_BOX)] for _ in range(LASER_PHOTON_BOX)] for _ in range(BACKGROUND_PHOTON_BOX)] for _ in range(NUMBER_OF_RUNS_BOX)] for _ in range(LASER_REPETITION_PERIOD_BOX)] for _ in range(LASER_TIME_PERIOF_BINS_BOX)] for _ in range(LASER_FWHM_BOX)] for _ in range(NUM_LASER_PULSES_BOX)] for _ in range(NUM_HISTOGRAM_BINS_BOX)]

    
    for DIST_NUM in range(DIST_NUM_BOX):
        for DIST_MIN in range(DIST_MIN_BOX):
            for DIST_MAX in range(DIST_MAX_BOX):
                for LASER_PHOTON in range(LASER_PHOTON_BOX):
                    for BACKGROUND_PHOTON in range(BACKGROUND_PHOTON_BOX):
                        for NUMBER_OF_RUNS in range(NUMBER_OF_RUNS_BOX):
                            for LASER_REPETITION_PERIOD in range(LASER_REPETITION_PERIOD_BOX):
                                for LASER_TIME_PERIOD_BINS in range(LASER_TIME_PERIOD_BINS_BOX):
                                    for LASER_FWHM in range(LASER_FWHM_BOX):
                                        for NUM_LASER_PULSES in range(NUM_LASER_PULSES_BOX):
                                            for NUM_HISTOGRAM_BINS in range(NUM_HISTOGRAM_BINS):
                                                optimal = findBestInGroup((DIST_NUM_MIN+(DIST_NUM_INT*DIST_NUM)),(DIST_MIN_MIN+(DIST_MIN_INT*DIST_MIN)),(DIST_MAX_MIN+(DIST_MAX_INT*DIST_MAX)),(LASER_PHOTON_MIN+(LASER_PHOTON_INT*LASER_PHOTON)),(BACKGROUND_PHOTON_MIN+(BACKGROUND_PHOTON_INT*BACKGROUND_PHOTON)),(NUMBER_OF_RUNS_MIN+(NUMBER_OF_RUNS_INT*NUMBER_OF_RUNS)),(LASER_REPETITION_PERIOD_MIN+(LASER_REPETITION_PERIOD_INT*LASER_REPETITION_PERIOD)),(LASER_TIME_PERIOD_BINS_MIN+(LASER_TIME_PERIOD_BINS_INT*LASER_TIME_PERIOD_BINS)),(LASER_FWHM_MIN+(LASER_FWHM_INT*LASER_FWHM)),(NUM_LASER_PULSES_MIN+(NUM_LASER_PULSES_INT*NUM_LASER_PULSES)),(NUM_HISTOGRAM_BINS_MIN+(NUM_HISTOGRAM_BINS_INT*NUM_HISTOGRAM_BINS)), 0, 100, 10, 5, 6, 0)
                                                optimalBlockage[DIST_NUM][DIST_MIN][DIST_MAX][LASER_PHOTON][BACKGROUND_PHOTON][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM][NUM_LASER_PULSES][NUM_HISTOGRAM_BINS] = optimal[0]
                                                errorAtOptimal[DIST_NUM][DIST_MIN][DIST_MAX][LASER_PHOTON][BACKGROUND_PHOTON][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM][NUM_LASER_PULSES][NUM_HISTOGRAM_BINS] = optimal[1]
                                                print("DIST NUM: "+(DIST_NUM_MIN+(DIST_NUM_INT*DIST_NUM))+" DIST MIN: "+(DIST_MIN_MIN+(DIST_MIN_INT*DIST_MIN))+" DIST MAX: "+(DIST_MAX_MIN+(DIST_MAX_INT*DIST_MAX))+" LASER PHOTON: "+(LASER_PHOTON_MIN+(LASER_PHOTON_INT*LASER_PHOTON))+" BACKGROUND PHOTON: "+(BACKGROUND_PHOTON_MIN+(BACKGROUND_PHOTON_INT*BACKGROUND_PHOTON))+" NUMBER OF RUNS: "+(NUMBER_OF_RUNS_MIN+(NUMBER_OF_RUNS_INT*NUMBER_OF_RUNS))+" LASER REPETITION PERIOD: "+(LASER_REPETITION_PERIOD_MIN+(LASER_REPETITION_PERIOD_INT*LASER_REPETITION_PERIOD))+" LASER TIME PERIOD BINS: "+(LASER_TIME_PERIOD_BINS_MIN+(LASER_TIME_PERIOD_BINS_INT*LASER_TIME_PERIOD_BINS))+" LASER FWHM: "+(LASER_FWHM_MIN+(LASER_FWHM_INT*LASER_FWHM))+" NUM LASER PULSES: "+(NUM_LASER_PULSES_MIN+(NUM_LASER_PULSES_INT*NUM_LASER_PULSES))+" NUM HISTOGRAM BINS: "+(NUM_HISTOGRAM_BINS_MIN+(NUM_HISTOGRAM_BINS_INT*NUM_HISTOGRAM_BINS))+" Optimal Blockage: "+optimalBlockage[DIST_NUM][DIST_MIN][DIST_MAX][LASER_PHOTON][BACKGROUND_PHOTON][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM][NUM_LASER_PULSES][NUM_HISTOGRAM_BINS]+" Error at Optimal: "+errorAtOptimal[DIST_NUM][DIST_MIN][DIST_MAX][LASER_PHOTON][BACKGROUND_PHOTON][NUMBER_OF_RUNS][LASER_REPETITION_PERIOD][LASER_TIME_PERIOD_BINS][LASER_FWHM][NUM_LASER_PULSES][NUM_HISTOGRAM_BINS]+".")
    print("That's all folks!")

'''
data = [0 for _ in range(1000)]
error =[0 for _ in range(1000)]
mindex=0;
for i in range(1000):
    data[i] =((i+9000)/100)
    error[i]=getError(10, 0.25, 1, 1, 5, 11, 100.0, 1000, 2, 3000, 8, (i+9000)/10000)
    if(error[i]<error[mindex]):
        mindex=i
    print("\r"+str("{:.5f}".format(data[i])+"%"),end="")
    #print(str("{:.5f}".format(data[i]))+"% filtration error value: "+str("{:.5f}".format(error[i]))));
    
print("Minimum: "+str("{:.5f}".format(data[mindex]))+"% attenuation, with an error of " + str("{:.5f}".format(error[mindex])))
#'''
'''
data = [0 for _ in range(10000)]
error =[0 for _ in range(10000)]
mindex=0;
print('|', end="")
for i in range(10000):
    data[i] =(i/100)
    error[i]=getError(10, 0.25, 1, 1, 5, 11, 100.0, 1000, 2, 3000, 8, i/10000)
    if(error[i]<error[mindex]):
        mindex=i
    print("\r"+str("{:.5f}".format(data[i])+"%"),end="")
                      
print("Minimum: "+str("{:.5f}".format(data[mindex]))+"% attenuation, with an error of " + str("{:.5f}".format(error[mindex])))
#'''
"""
data = [0 for _ in range(10000)]
error =[0 for _ in range(10000)]
mindex=0;
print('|')
for i in range(10000):
    data[i] =(i/100)
    error[i]=getError(9, 0.1, 0.9, 2, 3, 15, 110.0, 1200, 3, 3200, 100, i/10000)
    if(error[i]<error[mindex]):
        mindex=i
    print("\r"+str("{:.5f}".format(data[i])+"%"),end="")
                      
print("Minimum: "+str("{:.5f}".format(data[mindex]))+"% attenuation, with an error of " + str("{:.5f}".format(error[mindex])))
#"""
"""
data = [0 for _ in range(1000)]
error =[0 for _ in range(1000)]
mindex=0;
print('|')
for i in range(1000):
    data[i] =((i+9000)/100)
    error[i]=getError(9, 0.1, 0.9, 2, 3, 15, 110.0, 1200, 3, 3200, 100, (i+9000)/10000)
    if(error[i]<error[mindex]):
        mindex=i
    print("\r"+str("{:.5f}".format(data[i])+"%"),end="")
                      
print("Minimum: "+str("{:.5f}".format(data[mindex]))+"% attenuation, with an error of " + str("{:.5f}".format(error[mindex])))
#"""

#print(getError(10, 0.25, 1, 1, 5, 11, 100.0, 1000, 2, 3000, 8, 0))

#'''
data = [0 for _ in range(10000)]
error =[0 for _ in range(10000)]
mindex=0;
for i in range(10000):
    data[i] =(i/100)
    error[i]=getError(10, 0.25, 1, 5, 1, 11, 100.0, 1000, 2, 5000, 8, i/10000)
    if(error[i]<error[mindex]):
        mindex=i
    print("\r"+str("{:.5f}".format(data[i])+"%")+" "+str("{:.5f}".format(error[i])+"."),end="")
    #print(str("{:.5f}".format(data[i]))+"% filtration error value: "+str("{:.5f}".format(error[i]))));
    
print("\nMinimum: "+str("{:.5f}".format(data[mindex]))+"% attenuation, with an error of " + str("{:.5f}".format(error[mindex])))

pyplot.plot(data,error)
pyplot.show()

     
#'''
