import numpy as np
import control as ct
import matlab.engine
import os
import time

def get_params(device_name,dcOpData):
    '''
    Safely retrieves the transconductance (Gm) and drain-source resistance (r) for a given device.
    
    Parameters:
    ----------
    device_name (str): The name of the device.
    dcOpData (pd.DataFrame): The DataFrame containing DC operating point data.
    
    Returns:
    -------
    tuple: A tuple containing the transconductance (gm) and drain-source resistance (r).
    '''

    try:
        # loc[] accesses the row by the index label (Device name)
        gm = dcOpData.loc[device_name, 'Gm']
        gds = dcOpData.loc[device_name, 'Gds']
        
        # Calculate resistance, handling division by zero
        r = 1.0 / gds if gds != 0 else np.inf
        
        return gm, r
    except KeyError:
        print(f"Warning: Device '{device_name}' not found in the data.")
        return 0.0, np.inf # Return safe defaults or handle as needed



def get_simple_math_TF_biopot_sensor_python(Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co):
    ''' 
    This function takes the circuit parameters along with the OpAmp open loop gains
    and computes the SIMPLIFIED transfer functions of different stages of teh Biopotential Sensor 
    topology in Guerrero et. al. 2016 IEEE TCAS-I
    
    Parameters:
    ----------
    Aa: Open loop gain of the OpAmp in the input stage a
    Ab: Open loop gain of the OpAmp in the buffer stage b
    Ac: Open loop gain of the OpAmp in the input stage c
    Ao: Open loop gain of the OpAmp in the output stage o
    Ra: Resistance value for the input stage a.
    Ca: Capacitance value for the input stage a.
    Rc: Resistance value for the input stage c.
    Cc: Capacitance value for the input stage c.
    Ro: Resistance value for the output stage.
    Co: Capacitance value for the output stage.
    
    Returns:
    -------
    A tuple containing the transfer functions H_oa, H_oc, H_od, and H_db.
    '''
    
    # Define 's' the laplace variable
    s = ct.TransferFunction.s
    
    Za = Ra + 1/(s*Ca)
    Zc = Rc + 1/(s*Cc)
    Zo = 1/(1/Ro + s*Co)
    As = Ao/(1/Za + 1/Zc + 1/Zo)
    
    H_oa = ct.minreal(-(Aa*As)/((1+Aa)*(1+As/Zo)*Za), verbose=False)
    H_oc = ct.minreal(-(Ac*As)/((1+Ac)*(1+As/Zo)*Zc), verbose=False)
    H_od = ct.minreal(Ao/(1+As/Zo), verbose=False)
    H_db = Ab/(1+Ab) + 0*s
    
    return H_oa, H_oc, H_od, H_db



def get_math_TF_biopot_sensor_MATLAB(circuit_params):
    '''
    Compute the transfer functions for the biopotential sensor using MATLAB.
    
    Parameters:
    ----------
    circuit_params: A dictionary containing the circuit parameters:
        - Ra, Ca: Resistance and capacitance for the input stage a
        - Rc, Cc: Resistance and capacitance for the input stage c
        - Ro, Co: Resistance and capacitance for the output stage
        
    Returns:
    -------
    H_oa, H_oc, H_od, H_db: The transfer functions for the different stages of the biopotential sensor.
    '''

    print("Starting MATLAB Engine...")
    eng = matlab.engine.start_matlab()
    
    # Add src directory to MATLAB path to ensure it finds the function
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, 'src')
    eng.addpath(src_dir)

    # 1. Define Parameters (matching MATLAB script)
    Ra_val = circuit_params['Ra']
    Ca_val = circuit_params['Ca']
    Rc_val = circuit_params['Rc']
    Cc_val = circuit_params['Cc']
    Ro_val = circuit_params['Ro']
    Co_val = circuit_params['Co']

    print("Calling MATLAB function 'get_bioPot_TFs'...")
    matlab_call_time = time.time()
    # Call the MATLAB Function
    # nargout=8 tells python to expect 8 return values
    # We pass inputs as floats to ensure MATLAB interprets them correctly
    outputs = eng.get_bioPot_TFs(float(Ra_val), float(Ca_val), 
                                 float(Rc_val), float(Cc_val), 
                                 float(Ro_val), float(Co_val), 
                                 nargout=8)
    matlab_return_time = time.time()
    print(f"Returened matlab outputs. Time taken = {matlab_return_time-matlab_call_time}")
    
    # Unpack the results
    # MATLAB returns these as 2D arrays. We flatten them for Python.
    nmd_oa = np.array(outputs[0]).flatten()
    dnd_oa = np.array(outputs[1]).flatten()
    nmd_oc = np.array(outputs[2]).flatten()
    dnd_oc = np.array(outputs[3]).flatten()
    nmd_od = np.array(outputs[4]).flatten()
    dnd_od = np.array(outputs[5]).flatten()
    nmd_db = np.array(outputs[6]).flatten()
    dnd_db = np.array(outputs[7]).flatten()

    eng.quit() # Close MATLAB engine to free memory

    # 3. Create Transfer Functions in Python
    H_oa = ct.tf(nmd_oa, dnd_oa)
    H_oc = ct.tf(nmd_oc, dnd_oc)
    H_od = ct.tf(nmd_od, dnd_od)
    H_db = ct.tf(nmd_db, dnd_db)
    
    return H_oa, H_oc, H_od, H_db



def get_math_TF_three_STIA_low_freq(circuit_params):
    '''
    Compute the transfer functions for the three-stage transimpedance amplifier.
    
    Parameters:
    ----------
    circuit_params: A dictionary containing the circuit parameters:
        - R_d2, L_d2, R_d3, L_d3: Resistor and inductor values for the two stages
        - gm2, gm3: Transconductance values for the two stages
        - rds2, rds3: Drain-source resistance values for the two stages
        - C_L: Load capacitance at the output

    Returns:
    -------
    H_ba: Transfer function from Va to Vb
    H_cb: Transfer function from Vb to Vc
    '''

    R_d2 = circuit_params['R_d2']
    L_d2 = circuit_params['L_d2']
    R_d3 = circuit_params['R_d3']
    L_d3 = circuit_params['L_d3']
    gm2 = circuit_params['gm2']
    gm3 = circuit_params['gm3']
    rds2 = circuit_params['rds2']
    rds3 = circuit_params['rds3']
    C_L = circuit_params['C_L']

    s = ct.TransferFunction.s

    H_ba = - ((s*L_d2*gm2*rds2 + R_d2*gm2*rds2) / (s*L_d2 + rds2 + R_d2))

    H_cb = - ((s*L_d3*gm3*rds3 + R_d3*gm3*rds3) / (L_d3*C_L*rds3*s**2 + s*(C_L*R_d3*rds3 + L_d3) + rds3 + R_d3))

    return H_ba, H_cb



def get_math_TF_8_stagger_tunned_amplifier(circuit_params):
    '''
    Compute the transfer functions for the 8-stage stagger-tuned amplifier.

    Parameters:
    ----------
    circuit_params: A dictionary containing the circuit parameters for each stage:
        - R_vg{stage}: Gate resistance for stage {stage}
        - C_{stage}: Capacitance for stage {stage}
        - L_{stage}: Inductance for stage {stage}
        - gm{stage}: Transconductance for stage {stage}
        - rds{stage}: Drain-source resistance for stage {stage}

    Returns:
    -------
    A tuple containing the transfer functions for each stage: (H_21, H_32, H_43, H_54, H_65, H_76, H_87)
    '''

    H = {}
    for stage in range(2,9):
        Rvg_k = circuit_params[f'R_vg{stage}']
        C_k = circuit_params[f'C_{stage}']
        L_k = circuit_params[f'L_{stage}']
        gm_k = circuit_params[f'gm{stage}']
        rds_k = circuit_params[f'rds{stage}']
        C_k1 = circuit_params[f'C_{stage-1}']
        Rvgk_1 = circuit_params[f'R_vg{stage-1}']

        s = ct.TransferFunction.s

        h_k = - (gm_k*rds_k*L_k*C_k*Rvg_k*s**2 + s*L_k*gm_k*rds_k) / (L_k*C_k*(rds_k + Rvg_k)*s**2 + s*(L_k + C_k*rds_k*Rvg_k) + rds_k)
        z_k = (s*C_k1*Rvgk_1) / (s*C_k1*Rvgk_1 + 1)
        H[f'stage_{stage}'] = h_k * z_k

    H_21 = H['stage_2']
    H_32 = H['stage_3']
    H_43 = H['stage_4']
    H_54 = H['stage_5']
    H_65 = H['stage_6']
    H_76 = H['stage_7']
    H_87 = H['stage_8']

    return H_21, H_32, H_43, H_54, H_65, H_76, H_87



def get_math_TF_5_tran_OTA_python(circuit_params):
    '''
    Compute the transfer functions for the 5-transistor operational transconductance amplifier (OTA).

    Parameters:
    ----------
    circuit_params: A dictionary containing the circuit parameters for each transistor
        - gm_{transistor}: Transconductance for transistor {transistor}
        - rds_{transistor}: Drain-source resistance for transistor {transistor}
        - C_L: Load capacitance at the output

    Returns:
    -------
    A tuple containing the transfer functions for the OTA: (H_41, H_42, H_43)
    '''

    gm_1 = circuit_params['gm_1']
    rds_1 = circuit_params['rds_1']
    gm_2 = circuit_params['gm_2']
    rds_2 = circuit_params['rds_2']
    gm_3 = circuit_params['gm_3']
    rds_3 = circuit_params['rds_3']
    gm_4 = circuit_params['gm_4']
    rds_4 = circuit_params['rds_4']
    gm_5 = circuit_params['gm_5']
    rds_5 = circuit_params['rds_5']
    C_L = circuit_params['C_L']

    s = ct.TransferFunction.s

    a_1 = (1 + rds_5/((1 + (rds_1/rds_3) + gm_3*rds_1)*rds_3) - (gm_1*rds_5)/(1 + (rds_1/rds_3) + gm_3*rds_1) + (gm_3*rds_5)/(1 + (rds_1/rds_3) + gm_3*rds_1) + (rds_5/rds_2) + gm_1*rds_5 + gm_2*rds_5)
    b_1 = ((gm_4*rds_1)/(s*C_L*(1 + (rds_1/rds_3) + gm_3*rds_1)*rds_3) - gm_4/(s*C_L) + gm_2/(s*C_L) + 1/(s*C_L*rds_2) - (gm_1*gm_4*rds_1)/(s*C_L*(1 + (rds_1/rds_3) + gm_3*rds_1)) + (gm_3*gm_4*rds_1)/(s*C_L*(1 + (rds_1/rds_3) + gm_3*rds_1)))
    a_2 = 1 + 1/(s*C_L*rds_2) + 1/(s*C_L*rds_4) - (rds_5*b_1)/(rds_2*a_1)

    H_41 = (gm_1*gm_4*rds_1)/(s*C_L*(1 + (rds_1/rds_3) + gm_3*rds_1)*a_2) + (1 - 1/(1 + (rds_1/rds_3) + gm_3*rds_1))*((gm_1*rds_5*b_1)/(a_1*a_2))
    H_42 = (gm_2*rds_5*b_1)/(a_1*a_2) - gm_2/(s*C_L*a_2)
    H_43 = - (rds_5*b_1*gm_5)/(a_1*a_2)

    return H_41, H_42, H_43



def get_math_TF_Hardware_Circuit_python(circuit_params):
    '''
    Compute the transfer functions for the hardware circuit.
    
    Parameters:
    ----------
    circuit_params: A dictionary containing the circuit parameters:
        - R1 to R25: Resistance values for the circuit components
        - C1 to C8: Capacitance values for the circuit components
        - gm2, gm4, gm5: Transconductance values for the transistors
        - rds2, rds4, rds5: Drain-source resistance values for the transistors
    
    Returns:
    -------
    A tuple containing the transfer functions H_31, H_32, H_43
    '''
    
    R1 = circuit_params['R1']
    R2 = circuit_params['R2']
    R3 = circuit_params['R3']
    R4 = circuit_params['R4']
    R5 = circuit_params['R5']
    R6 = circuit_params['R6']
    R7 = circuit_params['R7']
    R8 = circuit_params['R8']
    R9 = circuit_params['R9']
    R10 = circuit_params['R10']
    R11 = circuit_params['R11']
    R12 = circuit_params['R12']
    R13 = circuit_params['R13']
    R14 = circuit_params['R14']
    R15 = circuit_params['R15']
    R16 = circuit_params['R16']
    R17 = circuit_params['R17']
    R18 = circuit_params['R18']
    R19 = circuit_params['R19']
    R20 = circuit_params['R20']
    R21 = circuit_params['R21']
    R22 = circuit_params['R22']
    R23 = circuit_params['R23']
    R24 = circuit_params['R24']
    R25 = circuit_params['R25']
    C1 = circuit_params['C1']
    C2 = circuit_params['C2']
    C3 = circuit_params['C3']
    C4 = circuit_params['C4']
    C5 = circuit_params['C5']
    C6 = circuit_params['C6']
    C7 = circuit_params['C7']
    C8 = circuit_params['C8']
    gm2 = circuit_params['gm2']
    gm4 = circuit_params['gm4']
    gm5 = circuit_params['gm5']
    rds2 = circuit_params['rds2']
    rds4 = circuit_params['rds4']
    rds5 = circuit_params['rds5']

    s = ct.TransferFunction.s

    calZ_3 = ((s*C2*R6*(R1+R6)*R5*R7 + s*C2*R1*R2*R5*R7 + R1*R5*R7 + R2*R5*R7 + s*C2*R1*R2*R6*R7 + R1*R6*R7 + R2*R6*R7)/(s*C2*R6*(R1+R6)*R5 
            + s*C2*R1*R2*R5 + R1*R5 + R2*R5 + s*C2*R1*R2*R6 + R1*R6 + R2*R6 + R1*R7 + R2*R7 + s*C2*R6*(R1+R6)*R7 + s*C2*R1*R2*R7))
    calZ_s2 = R8/(s*C3*R8 + 1)
    calZ_s4 = R18/(s*C6*R18 + 1)
    M_alpha2 = 1 + gm2*calZ_s2 + calZ_s2/rds2
    M_alpha4 = 1 + gm4*calZ_s4 + calZ_s4/rds4
    T2 = calZ_3/(rds2*M_alpha2)
    T4 = calZ_3/(rds4*M_alpha4)
    calC_2 = - (calZ_3*gm2)/M_alpha2
    calC_4 = - (calZ_3*gm4)/M_alpha4
    Z_2 = (s*C4*R9*R10*R12)/(s*C4*(R9*R10*R11 + R9*R10*R12 + R10*R11*R12 + R9*R11*R12) + R9*R11 + R9*R12 + R10*R11 + R10*R12)
    Z_4 = (s*C7*R17*R19*R21)/(s*C7*(R17*R19*R20 + R17*R19*R21 + R19*R20*R21 + R17*R20*R21) + R17*R20 + R17*R21 + R19*R20 + R19*R21)
    H_2 = calC_2/(1 + T2 + T4)
    H_4 = calC_4/(1 + T2 + T4)

    H_31 = H_2*Z_2
    H_32 = H_4*Z_4
    H_43 = (-((s*R3*rds5*gm5*C1*R4 + R3*rds5*gm5)/(s*C1*R4*rds5 + rds5 +gm5*rds5*R4 + R4 + s*C1*R4*R3 + R3)*
              ((s*C2*R1*R2*R6)/(s*C2*(R1*R2*R5 + R1*R2*R6 + R2*R5*R6 + R1*R5*R6) + R1*R5 + R1*R6 + R2*R5 + R2*R6))))
    
    return H_31, H_32, H_43



def get_math_TF_three_MSTA_TIA(circuit_params):
    '''
    Compute the transfer functions for the three-stage multi-stage transimpedance amplifier (MSTA TIA).
    
    Parameters:
    ----------
    circuit_params: A dictionary containing the circuit parameters:
        - Rd2: Drain resistance for stage 2
        - Rd3: Drain resistance for stage 3
        - RF: Feedback resistance
        - L2: Inductance for stage 2
        - L3: Inductance for stage 3
        - CL: Load capacitance
        - Cin: Input capacitance
        - gm2: Transconductance for stage 2
        - rds2: Drain-source resistance for stage 2
        - gm3: Transconductance for stage 3
        - rds3: Drain-source resistance for stage 3

    Returns:
    -------
    H_ba: Transfer function from Va to Vb
    H_cb: Transfer function from Vb to Vc
    '''


    Rd2 = circuit_params['Rd2']
    Rd3 = circuit_params['Rd3']
    RF = circuit_params['RF']
    L2 = circuit_params['L2']
    L3 = circuit_params['L3']
    CL = circuit_params['CL']
    Cin = circuit_params['Cin']
    gm1 = circuit_params['gm1']
    rds1 = circuit_params['rds1']
    gm2 = circuit_params['gm2']
    rds2 = circuit_params['rds2']
    gm3 = circuit_params['gm3']
    rds3 = circuit_params['rds3']
    RM4 = circuit_params['RM4']

    s = ct.TransferFunction.s

    calZ_o3 = ((Rd3 + s*L3)*(RF + 1/(s*Cin)))/((RF + 1/(s*Cin)) + (Rd3 + s*L3) + s*CL*(Rd3 + s*L3)*(RF + 1/(s*Cin)))
    H_ba = -(s*L2*gm2*rds2 + gm2*rds2*Rd2)/(s*L2 + Rd2 + rds2)
    H_cb = -(gm3*rds3*calZ_o3)/(rds3 + calZ_o3)
    H_ac = -((RM4*gm1*rds1)/(rds1 + RM4))*(1/(s*Cin*RF + 1))

    return H_ba, H_cb, H_ac



def get_math_TF_LDO_python(circuit_params):
    '''
    Compute the transfer functions for the LDO.

    Parameters:
    ----------
    circuit_params: A dictionary containing the circuit parameters for each transistor
        - gm{transistor}: Transconductance for transistor {transistor}
        - rds{transistor}: Drain-source resistance for transistor {transistor}
        - Ra, Rb, Rt: Resistor values for the current mirror loads and biasing
        - R1, R2: Resistor values for the output voltage divider
        - Rc, Cc: Compensation resistor and capacitor values

    Returns:
    -------
    A tuple containing the transfer functions for the two-stage op-amp: (H_QX, H_QA, H_QY, H_BX, H_BA, H_BY, H_YQ, H_YB)
    '''

    Ra = circuit_params['Ra']
    Rb = circuit_params['Rb']
    Rt = circuit_params['Rt']
    gm1 = circuit_params['gm1']
    gm2 = circuit_params['gm2']
    gm3 = circuit_params['gm3']
    gm4 = circuit_params['gm4']
    gm5 = circuit_params['gm5']
    gm6 = circuit_params['gm6']
    gm7 = circuit_params['gm7']
    gm8 = circuit_params['gm8']
    gm9 = circuit_params['gm9']
    gm10 = circuit_params['gm10']
    gm11 = circuit_params['gm11']
    rds1 = circuit_params['rds1']
    rds2 = circuit_params['rds2']
    rds3 = circuit_params['rds3']
    rds4 = circuit_params['rds4']
    rds5 = circuit_params['rds5']
    rds6 = circuit_params['rds6']
    rds7 = circuit_params['rds7']
    rds8 = circuit_params['rds8']
    rds9 = circuit_params['rds9']
    rds10 = circuit_params['rds10']
    rds11 = circuit_params['rds11']
    R1 = circuit_params['R1']
    R2 = circuit_params['R2']
    Rc = circuit_params['Rc']
    Cc = circuit_params['Cc']

    s = ct.TransferFunction.s

    a1 = 1 + rds3/rds1 + rds3/Ra
    a2 = 1 + rds4/rds2 + rds4/Rb
    a3 = 1 + gm1*rds5 + gm2*rds5 + rds5/rds1 + rds5/rds2
    b1 = (gm2*rds4)/a2 + rds4/(a2*rds2)
    a4 = 1 - (b1*rds5)/(a3*rds2)
    b2 = rds3/(a1*rds1) + (gm1*rds3)/a1
    a5 = 1 - (rds5*b2)/(a3*rds1)
    b3 = (rds5*b2)/(a3*rds2*a5)
    a6 = 1 - (rds5*b1*b3)/(a3*rds1*a4)
    b5 = Rb/(Ra + Rb)
    b6 = Ra/(Ra +Rb)
    a7 = (1 - b5*(-(gm3*rds3)/(a5*a1*a6) + (rds3)/(a1*Ra*a5*a6) - (gm4*rds4*b3)/(a4*a2*a6) + (b3*rds4)/(a4*a2*Rb*a6)))
    b4 = b5*(-(gm4*rds4)/(a4*a2) + (rds4)/(a4*a2*Rb)) + (rds5*b1)/(a3*rds1*a4)
    a8 = 1 - (b6*(-(gm4*rds4)/(a4*a2) + rds4/(a4*a2*Rb)) + (-(gm3*rds3*b4)/(a5*a1*a6) + (rds3*b4)/(a1*Ra*a5*a6) - 
              (gm4*rds4*b3*b4)/(a4*a2*a6) + (b3*rds4*b4)/(a4*a2*Rb*a6))*(b6/a7))
    a9 = 1 + gm9*rds9 + rds9/rds10
    a10 = -(gm10*rds9)/a9
    a11 = 1 + (Rc + 1/(s*Cc))*(1/rds7 + 1/rds8)
    a12 = 1 - (R1+R2)*((1/a11)*(1/(Rc + 1/(s*Cc)) - gm11) - (1/rds11 + 1/(Rc + 1/(s*Cc))))
    H_YB = - ((gm7*R2)/(a11*a12))*(Rc + 1/(s*Cc))*(1/(Rc + 1/(s*Cc)) - gm11)
    H_YQ = - ((gm8*R2)/(a11*a12))*(Rc + 1/(s*Cc))*(1/(Rc + 1/(s*Cc)) - gm11)
    H_QX = ((gm1*rds5*b1)/(a4*a3*a8) - (gm1*rds3*b4)/(a1*a5*a6*a7*a8) + (gm1*rds5*b2*b4)/(a5*a3*a6*a7*a8) + (gm1*rds5*b1*b3*b4)/(a4*a3*a6*a7*a8)) + 0*s
    H_QA = (- (gm5*rds5*b1)/(a4*a3*a8) - (gm5*rds5*b2*b4)/(a5*a3*a6*a7*a8) - (gm5*rds5*b1*b3*b4)/(a4*a3*a6*a7*a8)) + 0*s
    H_QY = ((gm2*rds5*b1)/(a4*a3*a8) - (gm2*rds4)/(a4*a2*a8) + (gm2*rds5*b2*b4)/(a5*a3*a6*a7*a8) - (gm2*rds4*b3*b4)/(a4*a2*a6*a7*a8) + (gm2*rds5*b1*b3*b4)/(a4*a3*a6*a7*a8)) + 0*s
    H_BQ = ((-(gm3*rds3)/(a5*a1*a6) + (rds3)/(a1*Ra*a5*a6) - (gm4*rds4*b3)/(a4*a2*a6) + (b3*rds4)/(a4*a2*Rb*a6))*(a10*b6/a7)) + 0*s
    H_BX = (- (gm1*rds3*a10)/(a1*a5*a6*a7) + (gm1*rds5*b2*a10)/(a5*a3*a6*a7) + (gm1*rds5*b1*b3*a10)/(a4*a3*a6*a7) + H_BQ*H_QX) + 0*s
    H_BA = (- (gm5*rds5*b2*a10)/(a5*a3*a6*a7) - (gm5*rds5*b1*b3*a10)/(a4*a3*a6*a7) + H_BQ*H_QA) + 0*s
    H_BY = (- (gm2*rds4*b3*a10)/(a4*a2*a6*a7) + (gm2*rds5*b1*b3*a10)/(a4*a3*a6*a7) + (gm2*rds5*b2*a10)/(a5*a3*a6*a7) + H_BQ*H_QY) + 0*s
    
    return H_QX, H_QA, H_QY, H_BX, H_BA, H_BY, H_YQ, H_YB