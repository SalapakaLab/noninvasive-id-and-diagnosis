import numpy as np
from itertools import combinations
from src.cyclic_utilities import indepTest
from src.cyclic_utilities import compute_undir_graph_degrees




def stepA(psi,data,freqChoice,threshold):
    '''
    Perform step A of CCD alorithm.
    Inputs:  (1) psi:  An adjacency matix (p-adjacency) of size (nNodes*nNodes) representing an initialization of the PAG.
                    This could be a complete undirected graph initialization or 
                    the moral graph (obtaind by loacality property of Wiener filters).
            (2) data: Timeseries data for nNodes to perform intependence test on.
            (3) freqChoice: Frequencies over which the average magnitudes are to be calculated 
                            for the Wiener filters in the conditional independence checking step.
            (4) threshold: Threshold on the magnitudes of Wiener filters used for conditional independence test.
    Outputs: (1) A PAG with p-adjacencies of the equivalence class
            (2) A dictionary of the Sepsets i.e. d-eparation/m-separation sets
            (3) A dictionary of all the edge endpoints in the output PAG initialized as o-o
    '''
    nNodes = psi.shape[0]
    Sepset = {}
    n = 0
    flag = True
    while(flag == True):
        for X in range(nNodes):
            for Y in range(nNodes):
                if X != Y:
                    if psi[X,Y]==1 or psi[Y,X]==1:
                        nAdjacencies_psi_X = np.sum(psi[X,:])
                        if (nAdjacencies_psi_X-1) >= n:
                            Adjacencies_psi_X = np.nonzero(psi[X,:])[0]
                            indxY = np.where(Adjacencies_psi_X==Y)
                            D = np.delete(Adjacencies_psi_X,indxY)
                            # iterate over the combinations of size=n of the vertices in D
                            combination_set=[i for i in combinations(D,n)]
                            for S in combination_set:
                                testResult = indepTest(X,Y,S,data,freqChoice,threshold)
                                if testResult == True:
                                    psi[X,Y] = 0
                                    psi[Y,X] = 0
                                    Sepset[X,Y] = S
                                    Sepset[Y,X] = S
                                    break
        n = n + 1
        # Check the size of adjacencies of p-adjacent nodes in psi
        flag = False
        for X in range(nNodes):
            for Y in range(nNodes):
                if X != Y:
                    if psi[X,Y]==1 or psi[Y,X]==1:
                        nAdjacencies_psi_X = np.sum(psi[X,:])
                        # if atleast one node has adjacency >= n then continue the while loop
                        if (nAdjacencies_psi_X-1) >= n:
                            flag = True
                            break
            if flag == True:
                break    
        # Annotate the edge endpoints of the ouput graph of step A as o-o.
        if flag == False:
            EdgeEndPoints = {}
            for X in range(nNodes):
                for Y in range(nNodes):
                    if X != Y:
                        if psi[X,Y]==1 or psi[Y,X]==1:
                            EdgeEndPoints[X,Y] = 'o-o'
                            EdgeEndPoints[Y,X] = 'o-o'
                        
                        
    return psi,Sepset,EdgeEndPoints
                
    
                    

def stepB(psi,Sepset,EdgeEndPoints):
    '''
    Perform step B of CCD alorithm.
    Inputs:  (1) psi:  Output of step A. This is an adjacency (p-adjacency) matix of a PAG.
            (2) Sepset: The d-separation/m-separation set output of step A.
            (3) EdgeEndpoints: Orientation of the edge endpoints of the output PAG of step A
    Outputs: (1) A dictionary representing the underlines found in step B
            (2) A dictionary with some updated EdgeEndPoints found in step B
    '''
    nNodes = psi.shape[0]
    Underlines = {}
    for A in range(nNodes):
        for B in range(nNodes):
            for C in range(nNodes):
                if A != B and B != C and A != C:
                    if psi[A,B] == 1 and psi[B,C] == 1 and psi[A,C] == 0:
                        if B not in Sepset[A,C]:
                            EdgeEndPoints[A,B] = "-->"
                            EdgeEndPoints[B,A] = "<--"
                            EdgeEndPoints[C,B] = "-->"
                            EdgeEndPoints[B,C] = "<--"
                        else:
                            Underlines[str(A)+'*-*'+str(B)+'*-*'+str(C)] = 'Solid'
                        
    return Underlines,EdgeEndPoints




def stepC(psi,EdgeEndPoints,Sepset,data,freqChoice,threshold):
    '''
    Perform step C of CCD alorithm.
    Inputs:  (1) psi:  Output of step A. This is an adjacency (p-adjacency) matix of a PAG.
            (2) Sepset: The d-separation/m-separation set output of step A.
            (3) EdgeEndpoints: Orientation of the edge endpoints of the output PAG of step B.
            (4) data: Dataset to perform indepencence test if required.
            (5) freqChoice: Frequencies over which the average magnitudes are to be calculated 
                            for the Wiener filters in the conditional independence checking step.
            (6) threshold: Threshold on the magnitudes of Wiener filters used for conditional independence test.
    Output: (1) A dictionary with some updated EdgeEndPoints found in step C
    '''
    nNodes = psi.shape[0]
    for A in range(nNodes):
        for X in range(nNodes):
            for Y in range(nNodes):
                if A != X and X != Y and A != Y:
                    if psi[A,X] == 0 and psi[A,Y] == 0 and psi[X,Y] == 1 and X not in Sepset[A,Y]:
                        if set(Sepset[A,Y]).issubset(Sepset[A,X]) and len(Sepset[A,Y])<len(Sepset[A,X]):
                            EdgeEndPoints[X,Y] = "<--"
                            EdgeEndPoints[Y,X] = "-->"
                        elif not set(Sepset[A,X]).issubset(Sepset[A,Y]):
                            testResult = indepTest(A,X,Sepset[A,Y],data,freqChoice,threshold)
                            if testResult == False:
                                EdgeEndPoints[X,Y] = "<--"
                                EdgeEndPoints[Y,X] = "-->"
                            
    return EdgeEndPoints




def stepD(psi,Sepset,EdgeEndPoints,Underlines,data,freqChoice,threshold):
    '''
    Perform step D of CCD alorithm.
    Inputs:  (1) psi:  Output of step A. This is an adjacency (p-adjacency) matix of a PAG.
            (2) Sepset: The d-separation/m-separation set output of step A.
            (3) EdgeEndpoints: Orientation of the edge endpoints of the output PAG of step C.
            (4) data: Dataset to perform indepencence test if required
            (5) Underlines: A dictionary representing the underlines found in step B.
            (6) freqChoice: Frequencies over which the average magnitudes are to be calculated 
                            for the Wiener filters in the conditional independence checking step.
            (7) threshold: Threshold on the magnitudes of Wiener filters used for conditional independence test.
    Outputs: (1) A dictionary with some updated underlines found in step D
            (2) A dictionary containing the SupSepsets
    '''
    nNodes = psi.shape[0]
    vertices = range(nNodes)
    # Find the Local(psi,V) sets
    Local = {}
    for V in range(nNodes):
        Local[V] = tuple(np.nonzero(psi[V,:])[0])
        for X in vertices:
            if psi[X,V] == 0 and X != V:
                for Y in vertices:
                    if Y != X and Y != V and psi[X,Y] == 1 and psi[Y,V] == 1:
                        if EdgeEndPoints[X,Y] == "-->" and EdgeEndPoints[Y,V] == "<--":
                            Local[V] = Local[V] + (X,)
    m = 1
    SupSepset = {}
    flag = True
    while(flag == True):
        for A in vertices:
            for B in vertices:
                for C in vertices:
                    if A != B and B != C and A != C:
                        if psi[A,B] == 1 and psi[B,C] == 1 and psi[A,C] == 0:
                            if EdgeEndPoints[A,B] == "-->" and EdgeEndPoints[B,C] == "<--":
                                LocalSetA = set(Local[A])
                                sepsetAC = set(Sepset[A,C])
                                BC = {B,C}
                                sepsetACUnionBC = sepsetAC.union(BC)
                                D = LocalSetA.difference(sepsetACUnionBC)
                                if len(D)>= m:
                                    combination_set = [i for i in combinations(D,m)]
                                    for T in combination_set:
                                        Tset = set(T)
                                        sepsetACUnionB = sepsetAC.union({B})
                                        S = Tset.union(sepsetACUnionB)
                                        testResult = indepTest(A,C,tuple(S),data,freqChoice,threshold)
                                        if testResult == True:
                                            Underlines[str(A)+'-->'+str(B)+'<--'+str(C)] = 'Dotted'
                                            SupSepset[str(A)+str(B)+str(C)] = S
                                            break
        m = m + 1
        flag =  False
        for A in vertices:
            for B in vertices:
                for C in vertices:
                    if A != B and B != C and A != C:
                        if psi[A,B] == 1 and psi[B,C] == 1 and psi[A,C] == 0:
                            if EdgeEndPoints[A,B] == "-->" and EdgeEndPoints[B,C] == "<--":
                                LocalSetA = set(Local[A])
                                sepsetAC = set(Sepset[A,C])
                                BC = {B,C}
                                sepsetACUnionBC = sepsetAC.union(BC)
                                D = LocalSetA.difference(sepsetACUnionBC)
                                if len(D)>= m:
                                    flag = True
                                    break
                if flag == True:
                    break
            if flag == True:
                break
            
    return Underlines, SupSepset




def stepE(psi,EdgeEndPoints,Underlines,SupSepset):
    '''
    Perform step E of CCD alorithm.
    Inputs:  (1) psi:  Output of step A. This is an adjacency (p-adjacency) matix of a PAG.
            (2) SupSepset: The d-separation/m-separation set output of step D.
            (3) EdgeEndpoints: Orientation of the edge endpoints of the output PAG of step C.
            (4) Underlines: A dictionary representing the underlines found in step B and D
    Output:  (1) A dictionary with some updated edge end points found in step E
    '''
    nNodes = psi.shape[0]
    vertices = range(nNodes)
    for A in vertices:
        for B in vertices:
            for C in vertices:
                for D in vertices:
                    if A != B and B != C and C != D and D != A:
                        if psi[A,B] == 1 and psi[B,C] == 1:
                            if EdgeEndPoints[A,B] == "-->" and EdgeEndPoints[B,C] == "<--" and (str(A)+'-->'+str(B)+'<--'+str(C)) in Underlines:
                                if psi[A,D] == 1 and psi[D,C] == 1:
                                    if Underlines[str(A)+'-->'+str(B)+'<--'+str(C)] == "Dotted" and EdgeEndPoints[A,D] == "-->" and EdgeEndPoints[D,C] == "<--":
                                        if psi[B,D] == 1:
                                            if D not in SupSepset[str(A)+str(B)+str(C)]:
                                                EdgeEndPoints[B,D] = "-->"
                                                EdgeEndPoints[D,B] = "<--"
                                            else:
                                                if EdgeEndPoints[B,D] == "--o":
                                                    EdgeEndPoints[B,D] = "---"
                                                    EdgeEndPoints[D,B] = "---"
                                                elif EdgeEndPoints[B,D] == "o-o":
                                                    EdgeEndPoints[B,D] = "o--"
                                                    EdgeEndPoints[D,B] = "--o"
                                                elif EdgeEndPoints[B,D] == "<-o":
                                                    EdgeEndPoints[B,D] = "<--"
                                                    EdgeEndPoints[D,B] = "-->"
                                        
    return EdgeEndPoints





def stepF(psi,EdgeEndPoints,Underlines,SupSepset,data,freqChoice,threshold):
    '''
    Perform step F of CCD alorithm.
    Inputs: (1) psi:  Output of step A. This is an adjacency (p-adjacency) matix of a PAG.
            (2) SupSepset: The d-separation/m-separation set output of step D.
            (3) EdgeEndpoints: Orientation of the edge endpoints of the output PAG of step E.
            (4) data: Dataset to perform indepencence test if required.
            (5) Underlines: A dictionary representing the underlines found in step B and D.
            (6) freqChoice: Frequencies over which the average magnitudes are to be calculated 
                            for the Wiener filters in the conditional independence checking step.
            (7) threshold: Threshold on the magnitudes of Wiener filters used for conditional independence test.
    Output: (1) A dictionary with some updated edge end points found in step F
'''
    nNodes = psi.shape[0]
    vertices = range(nNodes)
    for A in vertices:
        for B in vertices:
            for C in vertices:
                for D in vertices:
                    if A != B and B != C and C != D and D != A:
                        if psi[A,B] == 1 and psi[B,C] == 1:
                            if EdgeEndPoints[A,B] == "-->" and EdgeEndPoints[B,C] == "<--" and (str(A)+'-->'+str(B)+'<--'+str(C)) in Underlines:
                                if Underlines[str(A)+'-->'+str(B)+'<--'+str(C)] == "Dotted":
                                    if (not (psi[A,D] == 1 and psi[C,D] == 1)) and psi[B,D] == 1:
                                        SupSepsetABC = SupSepset[str(A)+str(B)+str(C)]
                                        SupSepSetABCunionD = SupSepsetABC.union({D})
                                        testResult = indepTest(A, C, tuple(SupSepSetABCunionD), data, freqChoice, threshold)
                                        if testResult == False:
                                            EdgeEndPoints[B,D] = "-->"
                                            EdgeEndPoints[D,B] = "<--"
    return EdgeEndPoints




def algCCD(nNodes,data,freqChoice,threshold):
    '''
    Perform the CCD alorithm.
    Inputs: (1) nNodes: Number of nodes in generative structure.
            (2) data: FFT of time-series data.
            (3) freqChoice: Frequencies over which the average magnitudes are to be calculated 
                            for the Wiener filters in the conditional independence checking step.
            (4) threshold: Threshold on the magnitudes of Wiener filters used for conditional independence test.
    Outputs:(1) psi: Adjacency matrix of PAG representing the p-adjacencies.
            (2) EdgeEndPointsF: Edge orientations in the PAG
            (3) UnderlinesD: Underlines in the PAG
    '''
    # Start with complete graph
    completGraphAdj = np.ones((nNodes,nNodes)) - np.identity(nNodes)
    
    # Step-A
    psiA,SepsetA,EdgeEndPointsA = stepA(completGraphAdj, data, freqChoice, threshold)
    
    # Step-B
    UnderlinesB,EdgeEndPointsB = stepB(psiA, SepsetA, EdgeEndPointsA)
    
    # Step-C
    EdgeEndPointsC = stepC(psiA, EdgeEndPointsB, SepsetA, data, freqChoice, threshold)
    
    # Step-D
    UnderlinesD,SupSepsetD = stepD(psiA, SepsetA, EdgeEndPointsC, UnderlinesB, data, freqChoice, threshold)
    
    # Step-E
    EdgeEndPointsE = stepE(psiA, EdgeEndPointsC, UnderlinesD, SupSepsetD)
    
    # Step-F
    EdgeEndPointsF = stepF(psiA, EdgeEndPointsE, UnderlinesD, SupSepsetD, data, freqChoice, threshold)
    
    return psiA, EdgeEndPointsF, UnderlinesD





def find_tau_CCD_avg_deg(fftData,freqChoice,tauSearchLow,tauSearchHigh,tauSearchRes,avgDegLow,avgDegHigh,min_deg_lim,max_deg_lim,nSamples):
    '''
    Find threshold for PAG reconstruction
    Inputs: (1) fftData: FFT of the data to be used for skeleton rconstruction.
            (2) freq_choice: The choice of frequency over which the average of phase needs to be calculated.
            (3) tauSearchLow: Lower limit of threshold in the search space
            (4) tauSearchHigh: Upper limit of threshold in the search space
            (5) tauSearchRes: Reslution to be used for threshold searching
            (6) avgDegLow: Lower limit of average degree of PAG skeleton
            (7) avgDegHigh: Upper limit of average degree of PAG skeleton
            (8) min_deg_lim: Lower limit of degree of PAG skeleton
            (9) max_deg_lim: Upper limit of degree of PAG skeleton
            (10) nSamples: Number of samples in time-series dataset
    Output: 
            (1) tau_CCD: Smallest threshold that reconstructs a skeleton of PAG that satisfies the conditions on the degrees 
    '''
    nNodes = fftData.shape[1]
    
    for tau_CCD in np.arange(tauSearchLow, tauSearchHigh, tauSearchRes):
        psi,EdgeEndPoints,Underlines = algCCD(nNodes,fftData,freqChoice,tau_CCD)
        min_deg, max_deg, avg_deg = compute_undir_graph_degrees(psi)
        if avg_deg >= avgDegLow and  avg_deg <= avgDegHigh and max_deg <= max_deg_lim and min_deg >= min_deg_lim:
            # print(f"CCD threshold {nSamples} samples is {tau_CCD}")
            break
    if tau_CCD == None:
        print(f"Could not find tauCCD for {nSamples} samples")
    return tau_CCD