'''this script is used to run the hardware circuit fault diagnosis case studies. 
It loads the healthy data, processes it, and then calls the find_faulty_parameter_hardware function 
to estimate the faulty parameter. The true faulty parameter is then printed for comparison.'''


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.fault_diagnosis import find_faulty_parameter_hardware
import pandas as pd
import numpy as np

def main():
    ###################################################################################
    # %% Initialization, data loading and data processing
    # %% 
    row_num = 1
    n_FFT = 512
    Adj = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 0]])
    freq_uLim = 100e3
    freq_lLim = 1000
    nperseg, noverlap, win = 4096, 2048, 'hann'
    f_samp = 1e6

    # %% 
    time_series_path = "data/time_series_data/hardware_circuit/healthy"
    # Load all data using list comprehension and stack into columns
    data_Normal = pd.read_excel(f'{time_series_path}/Healthy_Hardware_Data_1MS_s.xlsx').iloc[:, 1::2].values
    # Scaling factors/gains used in data acquisition system. We need to compensate 
    # for scaling of the instrumentation amplifiers and other filters in the measurement circuit.
    # The filters have unity gain between 15Hz and 100kHz, therefore we need to compensate for the
    # instrumentation amplifier's gain. The gains of the instrumentation amplifiers listed on data sets
    # may vary within tolerance because of resistance tolerance of the R_G. We need to calibrate the gain 
    # and fine tune it in order to get proper scaling. Therefore, we choose the multiplier in the following
    # format (total datasheet gain)*(calibration factor due to component tolerance)
    data_scale_fac_1 = (991*1.99)*(1.15)
    data_scale_fac_2 = (991*1.99)*(1.15)
    data_scale_fac_3 = (991*1.99)*(1)
    data_scale_fac_4 = (496*1.99)*(0.9)
    data_Normal[:,0] = data_Normal[:,0]*(1/data_scale_fac_1)
    data_Normal[:,1] = data_Normal[:,1]*(1/data_scale_fac_2)
    data_Normal[:,2] = data_Normal[:,2]*(1/data_scale_fac_3)
    data_Normal[:,3] = data_Normal[:,3]*(1/data_scale_fac_4)

    estimated_faulty_param = find_faulty_parameter_hardware(row_num, data_Normal, Adj, n_FFT, f_samp, nperseg, noverlap, win, freq_lLim, freq_uLim)
    print(f"Estimated faulty parameter: {estimated_faulty_param}")

    Tag_path = f"data/time_series_data/hardware_circuit/faulty/Faulty_Params_Tag.csv"
    Tag_file = pd.read_csv(Tag_path)
    print(f"True faulty parameter: {Tag_file.loc[row_num-1,'Component ID']} with value {Tag_file.loc[row_num-1,'Value']}")





# Entry point of the script
if __name__ == "__main__":
    main() 