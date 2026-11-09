function [num_41, den_41, num_42, den_42, num_43, den_43] = get_five_tran_OTA_TFs()

    % Read DC Operating point data from cadence file
    dcOpData = readtable("data/op_and_freq_resp_data/five_tran_ota/FiveTranOTA45nm_dc_data.csv");
    
    %
    M1 = find(strcmp(dcOpData.Device,'M1')==1);
    M2 = find(strcmp(dcOpData.Device,'M2')==1);
    M3 = find(strcmp(dcOpData.Device,'M3')==1);
    M4 = find(strcmp(dcOpData.Device,'M4')==1);
    M5 = find(strcmp(dcOpData.Device,'M5')==1);
    
    %%
    syms rds_1 rds_2 rds_3 rds_4 rds_5
    syms gm_1 gm_2 gm_3 gm_4 gm_5
    syms C_L
    syms s
    
    a_1 = (1 + rds_5/((1 + rds_1/rds_3 + gm_3*rds_1)*rds_3) - (gm_1*rds_5)/(1 + rds_1/rds_3 + gm_3*rds_1) + (gm_3*rds_5)/(1 + rds_1/rds_3 + gm_3*rds_1) + rds_5/rds_2 + gm_1*rds_5 + gm_2*rds_5);
    b_1 = ((gm_4*rds_1)/((s*C_L*(1 + rds_1/rds_3 + gm_3*rds_1)*rds_3)) - gm_4/(s*C_L) + gm_2/(s*C_L) + 1/(s*C_L*rds_2) - (gm_1*gm_4*rds_1)/((s*C_L*(1 + rds_1/rds_3 + gm_3*rds_1))) + (gm_3*gm_4*rds_1)/((s*C_L*(1 + rds_1/rds_3 + gm_3*rds_1))));
    a_2 = 1 + 1/(s*C_L*rds_2) + 1/(s*C_L*rds_4) - (b_1*rds_5)/(rds_2*a_1);

    H_41_sym = (gm_1*gm_4*rds_1)/(s*C_L*(1 + rds_1/rds_3 + gm_3*rds_1)*a_2) + (1 - 1/(1 + rds_1/rds_3 + gm_3*rds_1))*((gm_1*rds_5*b_1)/(a_1*a_2));
    H_42_sym = (gm_2*rds_5*b_1)/(a_1*a_2) - gm_2/(s*C_L*a_2);
    H_43_sym = - (rds_5*b_1*gm_5)/(a_1*a_2);
    
    
    gm_1_val = dcOpData.Gm(M1);
    gm_2_val = dcOpData.Gm(M2);
    gm_3_val = dcOpData.Gm(M3);
    gm_4_val = dcOpData.Gm(M4);
    gm_5_val = dcOpData.Gm(M5);
    rds_1_val = 1/dcOpData.Gds(M1);
    rds_2_val = 1/dcOpData.Gds(M2);
    rds_3_val = 1/dcOpData.Gds(M3);
    rds_4_val = 1/dcOpData.Gds(M4);
    rds_5_val = 1/dcOpData.Gds(M5);
    
    C_L_val = 10e-12;
    
    sym_vars = [gm_1, gm_2, gm_3, gm_4, gm_5, rds_1, rds_2, rds_3, rds_4, rds_5, C_L];
    val_vars = [gm_1_val, gm_2_val, gm_3_val, gm_4_val, gm_5_val, rds_1_val, rds_2_val, rds_3_val, rds_4_val, rds_5_val, C_L_val];
    
    %
    H_41 = vpa(subs((H_41_sym), sym_vars, val_vars));
    [nm_41, dn_41] = numden(H_41);
    nmd_41 = sym2poly(nm_41);
    dnd_41 = sym2poly(dn_41);
    H_41_s = minreal(tf(nmd_41,dnd_41));

    %
    H_42 = vpa(subs((H_42_sym), sym_vars, val_vars));
    [nm_42, dn_42] = numden(H_42);
    nmd_42 = sym2poly(nm_42);
    dnd_42 = sym2poly(dn_42);
    H_42_s = minreal(tf(nmd_42,dnd_42));

    %
    H_43 = vpa(subs((H_43_sym), sym_vars, val_vars));
    [nm_43, dn_43] = numden(H_43);
    nmd_43 = sym2poly(nm_43);
    dnd_43 = sym2poly(dn_43);
    H_43_s = minreal(tf(nmd_43,dnd_43));
    


    [num_41, den_41] = tfdata(H_41_s);
    [num_42, den_42] = tfdata(H_42_s);
    [num_43, den_43] = tfdata(H_43_s);
    
end
    
    

