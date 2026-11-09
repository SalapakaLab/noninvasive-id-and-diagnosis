function [num_oa, den_oa, num_oc, den_oc, num_od, den_od, num_db, den_db] = get_bioPot_TFs(Ra_val, Ca_val, Rc_val, Cc_val, Ro_val, Co_val)

    % Read DC Operating point data from cadence file
    dcOpData = readtable("data/op_and_freq_resp_data/bio_pot_sensor/bio_pot_sensor_dc_data.csv");
    
    %
    M1 = find(strcmp(dcOpData.Device,'M34')==1);
    M2 = find(strcmp(dcOpData.Device,'M35')==1);
    M3 = find(strcmp(dcOpData.Device,'M37')==1);
    M4 = find(strcmp(dcOpData.Device,'M38')==1);
    M5 = find(strcmp(dcOpData.Device,'M33')==1);
    M7 = find(strcmp(dcOpData.Device,'M31')==1);
    M8 = find(strcmp(dcOpData.Device,'M36')==1);
    M9 = find(strcmp(dcOpData.Device,'M45')==1);
    M10 = find(strcmp(dcOpData.Device,'M44')==1);
    M11 = find(strcmp(dcOpData.Device,'M47')==1);
    M12 = find(strcmp(dcOpData.Device,'M48')==1);
    M13 = find(strcmp(dcOpData.Device,'M42')==1);
    M15 = find(strcmp(dcOpData.Device,'M43')==1);
    M16 = find(strcmp(dcOpData.Device,'M46')==1);
    M17 = find(strcmp(dcOpData.Device,'M53')==1);
    M18 = find(strcmp(dcOpData.Device,'M52')==1);
    M19 = find(strcmp(dcOpData.Device,'M55')==1);
    M20 = find(strcmp(dcOpData.Device,'M56')==1);
    M21 = find(strcmp(dcOpData.Device,'M50')==1);
    M23 = find(strcmp(dcOpData.Device,'M51')==1);
    M24 = find(strcmp(dcOpData.Device,'M54')==1);
    M25 = find(strcmp(dcOpData.Device,'M61')==1);
    M26 = find(strcmp(dcOpData.Device,'M60')==1);
    M27 = find(strcmp(dcOpData.Device,'M63')==1);
    M28 = find(strcmp(dcOpData.Device,'M64')==1);
    M29 = find(strcmp(dcOpData.Device,'M58')==1);
    M31 = find(strcmp(dcOpData.Device,'M59')==1);
    M32 = find(strcmp(dcOpData.Device,'M62')==1);
    
    %%
    syms r1 r2 r3 r4 r5 r7 r8
    syms gm1 gm2 gm3 gm4 gm8
    syms R1b R2b C1b C2b
    syms s
    
    Z1 = R1b + 1/(s*C1b);
    Z2 = 1/(1/r8 + 1/r7 + s*C2b);
    a1 = 1 + r5/((1+r1/r3+gm3*r1)*r3) - (gm1*r5)/(1+r1/r3+gm3*r1) + (gm3*r5)/(1+r1/r3+gm3*r1) + r5/r2 + gm1*r5 + gm2*r5;
    b1 = (gm3*gm4*r1*r4)/(1+r1/r3+gm3*r1) - (gm1*gm4*r1*r4)/(1+r1/r3+gm3*r1) + (gm4*r1*r4)/((1+r1/r3+gm3*r1)*r3) + r4/r2 + gm2*r4 - gm4*r4;
    a2 = 1 + r4/r2 - (b1*r5)/(r2*a1) + r4/Z1;
    b2 = (Z2*(1/Z1 - gm8))/(1 + Z2/Z1 - ((r4*Z2)/(Z1*a2))*(1/Z1 - gm8));
    a3 = 1 - (1-1/(1+r1/r3+gm3*r1))*((gm1*r5*b1*b2)/(a1*a2)) - (gm1*gm4*r1*r4*b2)/((1+r1/r3+gm3*r1)*a2);
    
    
    syms r9 r10 r11 r12 r13 r15 r16
    syms gm9 gm10 gm11 gm12 gm16
    syms R1A C1A C2A
    
    Z1A = R1A + 1/(s*C1A);
    Z2A = 1/(1/r15 + 1/r16 + s*C2A);
    a1A = 1 + r13/((1+r10/r11+gm11*r10)*r11) - (gm10*r13)/(1+r10/r11+gm11*r10) + (gm11*r13)/(1+r10/r11+gm11*r10) + r13/r9 + gm10*r13 + gm9*r13;
    b1A = (gm11*gm12*r10*r12)/(1+r10/r11+gm11*r10) - (gm10*gm12*r10*r12)/(1+r10/r11+gm11*r10) + (gm12*r10*r12)/((1+r10/r11+gm11*r10)*r11) + r12/r9 + gm9*r12 - gm12*r12;
    a2A = 1 + r12/r9 - (b1A*r13)/(r9*a1A) + r12/Z1A;
    
    
    syms r17 r18 r19 r20 r21 r23 r24
    syms gm17 gm18 gm19 gm20 gm24
    syms R1C C1C C2C
    
    Z1C = R1C + 1/(s*C1C);
    Z2C = 1/(1/r23 + 1/r24 + s*C2C);
    a1C = 1 + r21/((1+r18/r19+gm19*r18)*r19) - (gm18*r21)/(1+r18/r19+gm19*r18) + (gm19*r21)/(1+r18/r19+gm19*r18) + r21/r17 + gm18*r21 + gm17*r21;
    b1C = (gm19*gm20*r18*r20)/(1+r18/r19+gm19*r18) - (gm18*gm20*r18*r20)/(1+r18/r19+gm19*r18) + (gm20*r18*r20)/((1+r18/r19+gm19*r18)*r19) + r20/r17 + gm17*r20 - gm20*r20;
    a2C = 1 + r20/r17 - (b1C*r21)/(r17*a1C) + r20/Z1C;
    
    
    syms r25 r26 r27 r28 r29 r31 r32
    syms gm25 gm26 gm27 gm28 gm32
    syms R1P C1P C2P
    
    Z1P = R1P + 1/(s*C1P);
    Z2P = 1/(1/r31 + 1/r32 + s*C2P);
    a1P = 1 + r29/((1+r26/r27+gm27*r26)*r27) - (gm26*r29)/(1+r26/r27+gm27*r26) + (gm27*r29)/(1+r26/r27+gm27*r26) + r29/r25 + gm26*r29 + gm25*r29;
    b1P = (gm27*gm28*r26*r28)/(1+r26/r27+gm27*r26) - (gm26*gm28*r26*r28)/(1+r26/r27+gm27*r26) + (gm28*r26*r28)/((1+r26/r27+gm27*r26)*r27) + r28/r25 + gm25*r28 - gm28*r28;
    a2P = 1 + r28/r25 - (b1P*r29)/(r25*a1P) + r28/Z1P;
    
    syms Ra Ca Rc Cc Ro Co
    Za = Ra + 1/(s*Ca);
    Zc = Rc + 1/(s*Cc);
    Zo = 1/(1/Ro + s*Co);
    k1 = (Z2P/Z1P - Z2P*gm32)/(1 + Z2P/Z1P + Z2P/Zo);
    k2 = (Z2A/Z1A - Z2A*gm16)/(1 + Z2A/Z1A + Z2A/Za);
    Q1 = 1 - (1 - 1/(1+r10/r11+gm11*r10))*((gm10*r13*b1A*k2)/(a1A*a2A)) - (gm10*gm12*r10*r12*k2)/((1+r10/r11+gm11*r10)*a2A) - (r12*k2)/(Z1A*a2A);
    k3 = (Z2C/Z1C - Z2C*gm24)/(1 + Z2C/Z1C + Z2C/Zc);
    Q2 = 1 - (1 - 1/(1+r18/r19+gm19*r18))*((gm18*r21*b1C*k3)/(a1C*a2C)) - (gm18*gm20*r18*r20*k3)/((1+r18/r19+gm19*r18)*a2C) - (r20*k3)/(Z1C*a2C);
    k4 = 1/Za + 1/Zc + 1/Zo - Z2A/(Q1*Za*Za*(1 + Z2A/Z1A + Z2A/Za)) - Z2C/(Q2*Zc*Zc*(1 + Z2C/Z1C + Z2C/Zc));
    Q3 = (1 - 1/(1+r26/r27+gm27*r26))*((gm26*r29*b1P*k1)/(a1P*a2P)) + (gm26*gm28*r26*r28*k1)/((1+r26/r27+gm27*r26)*a2P) + Z2P/((1+Z2P/Z1P+Z2P/Zo)*Zo);
    Q4 = 1 - Q3/(Zo*k4) - (r28*k1)/(Z1P*a2P);
    
    H_oa_sym = (b1A*gm9*r13*k2*Q3)/(Za*a1A*a2A*Q1*k4*Q4) - (gm9*r12*k2*Q3)/(Za*a2A*Q1*k4*Q4);
    H_oc_sym = (b1C*gm17*r21*k3*Q3)/(Zc*a1C*a2C*Q2*k4*Q4) - (gm17*r20*k3*Q3)/(Zc*a2C*Q2*k4*Q4);
    H_od_sym = (b1P*gm25*r29*k1)/(a1P*a2P*Q4) - (gm25*r28*k1)/(a2P*Q4);
    H_db_sym = (b2/a3)*((b1*gm2*r5)/(a1*a2) - (gm2*r4)/a2);
    
    gm1_val = dcOpData.Gm(M1);
    gm2_val = dcOpData.Gm(M2);
    gm3_val = dcOpData.Gm(M3);
    gm4_val = dcOpData.Gm(M4);
    gm8_val = dcOpData.Gm(M8);
    r1_val = 1/dcOpData.Gds(M1);
    r2_val = 1/dcOpData.Gds(M2);
    r3_val = 1/dcOpData.Gds(M3);
    r4_val = 1/dcOpData.Gds(M4);
    r5_val = 1/dcOpData.Gds(M5);
    r7_val = 1/dcOpData.Gds(M7);
    r8_val = 1/dcOpData.Gds(M8);
    
    gm9_val = double(dcOpData.Gm(M9));
    gm10_val = double(dcOpData.Gm(M10));
    gm11_val = double(dcOpData.Gm(M11));
    gm12_val = double(dcOpData.Gm(M12));
    gm16_val = double(dcOpData.Gm(M16));
    r9_val = double(1/dcOpData.Gds(M9));
    r10_val = double(1/dcOpData.Gds(M10));
    r11_val = double(1/dcOpData.Gds(M11));
    r12_val = double(1/dcOpData.Gds(M12));
    r13_val = double(1/dcOpData.Gds(M13));
    r15_val = double(1/dcOpData.Gds(M15));
    r16_val = double(1/dcOpData.Gds(M16));
    
    gm17_val = double(dcOpData.Gm(M17));
    gm18_val = double(dcOpData.Gm(M18));
    gm19_val = double(dcOpData.Gm(M19));
    gm20_val = double(dcOpData.Gm(M20));
    gm24_val = double(dcOpData.Gm(M24));
    r17_val = double(1/dcOpData.Gds(M17));
    r18_val = double(1/dcOpData.Gds(M18));
    r19_val = double(1/dcOpData.Gds(M19));
    r20_val = double(1/dcOpData.Gds(M20));
    r21_val = double(1/dcOpData.Gds(M21));
    r23_val = double(1/dcOpData.Gds(M23));
    r24_val = double(1/dcOpData.Gds(M24));
    
    gm25_val = dcOpData.Gm(M25);
    gm26_val = dcOpData.Gm(M26);
    gm27_val = dcOpData.Gm(M27);
    gm28_val = dcOpData.Gm(M28);
    gm32_val = dcOpData.Gm(M32);
    r25_val = 1/dcOpData.Gds(M25);
    r26_val = 1/dcOpData.Gds(M26);
    r27_val = 1/dcOpData.Gds(M27);
    r28_val = 1/dcOpData.Gds(M28);
    r29_val = 1/dcOpData.Gds(M29);
    r31_val = 1/dcOpData.Gds(M31);
    r32_val = 1/dcOpData.Gds(M32);
    
    R1A_val = 2.3e3;
    C1A_val = 3.5e-12;
    C2A_val = 4e-12;
    R1C_val = 2.3e3;
    C1C_val = 3.5e-12;
    C2C_val = 4e-12;
    R1P_val = 2.3e3;
    C1P_val = 3.5e-12;
    C2P_val = 4e-12;
    
    R1b_val = 2.3e3;
    R2b_val = 220e3;
    C1b_val = 3.5e-12;
    C2b_val = 4e-12;
    
    sym_vars = [gm1, gm2, gm3, gm4, gm8, r1, r2, r3, r4, r5, r7, r8, R1b, R2b, C1b, C2b];
    val_vars = [gm1_val, gm2_val, gm3_val, gm4_val, gm8_val, r1_val, r2_val, r3_val, r4_val, r5_val, r7_val, r8_val, R1b_val, R2b_val, C1b_val, C2b_val];
    
    %
    H_db = vpa(subs((H_db_sym), sym_vars, val_vars));
    [nm_db, dn_db] = numden(H_db);
    nmd_db = sym2poly(nm_db);
    dnd_db = sym2poly(dn_db);
    H_db_s = minreal(tf(nmd_db,dnd_db));
    
    sym_vars2 = [gm9, gm10, gm11, gm12, gm16, gm17, gm18, gm19, gm20, gm24, gm25, gm26, gm27, gm28, gm32, r9, r10, r11, r12, r13, r15, r16, r17, r18, r19, r20, r21, r23, r24, r25, r26, r27, r28, r29, r31, r32, R1A, C1A, C2A, R1C, C1C, C2C, R1P, C1P, C2P, Ra, Ca, Rc, Cc, Ro, Co];
    val_vars2 = [gm9_val, gm10_val, gm11_val, gm12_val, gm16_val, gm17_val, gm18_val, gm19_val, gm20_val, gm24_val, gm25_val, gm26_val, gm27_val, gm28_val, gm32_val, r9_val, r10_val, r11_val, r12_val, r13_val, r15_val, r16_val, r17_val, r18_val, r19_val, r20_val, r21_val, r23_val, r24_val, r25_val, r26_val, r27_val, r28_val, r29_val, r31_val, r32_val, R1A_val, C1A_val, C2A_val, R1C_val, C1C_val, C2C_val, R1P_val, C1P_val, C2P_val, Ra_val, Ca_val, Rc_val, Cc_val, Ro_val, Co_val];
    
    %
    H_oa = vpa(subs(H_oa_sym, sym_vars2, val_vars2));
    [nm_oa, dn_oa] = numden(H_oa);
    nmd_oa = sym2poly(nm_oa);
    dnd_oa = sym2poly(dn_oa);
    H_oa_s = minreal(tf(nmd_oa,dnd_oa));
    
    %
    H_oc = vpa(subs((H_oc_sym), sym_vars2, val_vars2));
    [nm_oc, dn_oc] = numden(H_oc);
    nmd_oc = sym2poly(nm_oc);
    dnd_oc = sym2poly(dn_oc);
    H_oc_s = minreal(tf(nmd_oc,dnd_oc));
    
    %
    H_od = vpa(subs((H_od_sym), sym_vars2, val_vars2));
    [nm_od, dn_od] = numden(H_od);
    nmd_od = sym2poly(nm_od);
    dnd_od = sym2poly(dn_od);
    H_od_s = minreal(tf(nmd_od,dnd_od));


    [num_oa, den_oa] = tfdata(H_oa_s);
    [num_oc, den_oc] = tfdata(H_oc_s);
    [num_od, den_od] = tfdata(H_od_s);
    [num_db, den_db] = tfdata(H_db_s);
    
end
    
    

