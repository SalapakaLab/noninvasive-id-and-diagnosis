import numpy as np
from itertools import combinations
from src.utils import data_FFT

def Wiener_Proj(X,Z):
    '''
    Computes the projection of Z on to the column space of X using the Wiener filter coefficients.
    The projection is given by W*X, where W is the Wiener filter coefficient matrix given by 
    W=(X^*X)^{-1}X^*Z, where X^* is the conjugate transpose of X.
    
    Parameters:
    ----------
    X: nSamples X m matrix of the data of the nodes in the conditioning set
    Z: nSamples X 1 vector of the data of the node to be projected
    
    Returns:
    -------
    W: m X 1 vector of the Wiener filter coefficients
    '''

    # Project X on to Z, Z=a0x_0+a1x1+...am xm, 
    # where m is the number of columns of X
    X_star = X.conj().T
    XX=np.matmul(X_star, X)
    if np.isscalar(XX):
        return np.matmul(np.divide( X_star,XX), Z)
    else:
        return np.matmul(np.matmul(np.linalg.inv(np.matmul(X_star, X)), X_star), Z)



def compute_partial_Wiener_coeffs_avg(data,i,j,z=[],freq_choice=[]):
    '''
    Computes the partial Wiener coefficients of node i on node j conditioned on the set z, 
    averaged over the frequencies in freq_choice.

    Parameters:
    ----------
    data: nSamples X nNodes X n_FFT
    i: index of the node for which we want to compute the partial Wiener coefficients
    j: index of the node on which we want to project
    z: list of indices of the nodes in the conditioning set
    freq_choice: list of frequency indices to consider for the projection coefficient

    Returns:
    -------
    W: n_FFT X len(z) matrix of the partial Wiener coefficients of node i on node j conditioned on the set z,
      for each frequency in freq_choice
    W_mag_avg: average of the partial Wiener coefficients over the frequencies in freq_choice
    '''

    c=np.append(j,np.int32(z))
    indexes = np.unique(c, return_index=True)[1]
    c=[c[index] for index in sorted(indexes)]
    n1=len(c)
    n_FFT=data.shape[2]
    W=np.zeros([n_FFT,n1],dtype='complex')
    W_mag_avg = np.zeros([n1],dtype=float)
    for freq_index in freq_choice:
        fft_data=data[:,:,freq_index]
        X=fft_data[:,i]
        X_bar=fft_data[:,c]
        W[freq_index,:]=Wiener_Proj(X_bar,X)
        W_mag_avg = W_mag_avg + np.abs(W[freq_index,:])
    W_mag_avg = np.divide(W_mag_avg,len(freq_choice))
    return W,W_mag_avg



def Run_PC_Wiener_test(data,threshd,freq_choice,q, verbose=False):
    '''
    Takes in the FFT of the data and returns the separation sets for each pair of nodes.

    Parameters:
    ----------
    data: nSamples X nNodes X n_FFT
    threshd: threshold for the projection coefficient to determine the separation set
    freq_choice: list of frequency indices to consider for the projection coefficient
    q: maximum size of the separation set to consider

    Returns:
    -------
    C: dictionary with keys as tuples of node indices and values as the separation set for that pair of nodes
    '''
    nNodes=data.shape[1]
    C={}
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if ind1>ind2:
                if verbose:
                    print('Chosen node pair: [',ind1+1,',',ind2+1,']')
                flag=False
                # the complement set through which we condition
                D=[i for i in range(nNodes) if i != ind1 and i !=ind2]
                if verbose:
                    print("D ",np.array(D)+1)
                # iterate over the combinations of various size in the increasing order
                for ind_cond in range(nNodes-1):
                    if flag==True:
                        break
                    combination_set=[i for i in combinations(D,ind_cond)]
                    # print("combination"+str(ind_cond)+" =",combination_set)
                    for c in combination_set:
                        if verbose:
                            print("\n combination set: ",np.array(c)+1)
                        _,W_mag_i_jZ=compute_partial_Wiener_coeffs_avg(data,ind1,ind2,c,freq_choice)
                        if verbose:
                            print("Projection coefficient of X"+str(ind1+1)+" on "+str(np.append(ind2,c)+1)+"=",W_mag_i_jZ)
                        if W_mag_i_jZ[0]<threshd:
                            if verbose:
                                print("Separation set of ( X"+str(ind1+1)+", X"+str(ind2+1),") is : ",np.array(c)+1)
                            C[ind1,ind2]=c
                            flag=True
                            break
                        if np.shape(c)[0]==q:
                            if verbose:
                                print("No separation set for ( X"+str(ind1+1)+", X"+str(ind2+1),")")
                            C[ind1,ind2]=None
                            flag=True
                            break
            
    return C



def estimate_essential_graph(Top_est,C):
    '''
    Estimates essenial graph of underlying network using the estimated skeleton and conditional independence sets
    
    Parameters:
    ----------
    Top_est: Adjacency matrix of estimated skeleton
    C: Dictionary containing conditional independence/d-separation sets
    
    Returns:
    -------
    essential_graph: Estimated essential graph
    '''

    nNodes = Top_est.shape[0]
    #% Collider Identification
    collider_graph = np.copy(Top_est)
    collider_set = np.ones((nNodes,1))*(-1)
    collider_index = 0
    for i in range(nNodes):
        for j in range(nNodes):
            for k in range(nNodes):
                if i>j and j!=k and i!=k and Top_est[i,k]==1 and Top_est[j,k]==1 and Top_est[i,j]!=1 and (C[(i,j)]!=None) and (k not in C[(i,j)]):
                    collider_graph[i,k] = 0
                    collider_graph[j,k] = 0
                    collider_set[collider_index,0] = k
                    collider_index = collider_index + 1

    # % Orienting other edges if possible
    essential_graph = np.copy(collider_graph)
    for i in range(nNodes):
        for j in range(nNodes):
            if i>j and collider_graph[i,j]==1 and collider_graph[j,i]==1:
                if i in collider_set:
                    essential_graph[i,j] = 0
                if j in collider_set:
                    essential_graph[j,i] = 0

    
    return essential_graph



def compute_undir_graph_degrees(Adj):
    '''
    Computes the minimum, maximum and average degree of an undirected graph given its adjacency matrix.
    
    Parameters:
    ----------
    Adj: nNodes X nNodes adjacency matrix of the undirected graph

    Returns:
    -------
    min_deg: minimum degree of the graph
    max_deg: maximum degree of the graph
    avg_deg: average degree of the graph
    '''
    nNodes = Adj.shape[0]
    node_degs = []
    for node in range(nNodes):
        node_degs.append(sum(Adj[:,node]))
    max_deg = max(node_degs)
    min_deg = min(node_degs)
    avg_deg = sum(node_degs)/nNodes
    
    return min_deg, max_deg, avg_deg



def tune_WPC_threshold(data,nFFT_list,nperseg,thresh_range,freq_choice,avg_deg_lim,max_deg_lim,min_deg_lim,nSamplesTune,q):
    '''
    Tunes the threshold for W-PC test at different nFFT values based on the average degree of the estimated skeleton graph.
    The threshold is chosen such that the average degree of the estimated skeleton graph is within the specified limits.
    
    Parameters:
    ----------
    data: nSamples X nNodes X n_FFT
    nFFT_list: list of nFFT values to tune the threshold for
    thresh_range: range of thresholds to search
    freq_choice: list of frequency indices to consider for the projection coefficient
    avg_deg_lim: limit for average degree
    max_deg_lim: limit for maximum degree
    min_deg_lim: limit for minimum degree
    nSamplesTune: number of samples to use for tuning
    q: number of conditioning variables
    
    Returns:
    -------
    thresh_freq: list of tuned thresholds for each nFFT value
    '''

    nNodes=data.shape[1]
    thresh = np.full(len(nFFT_list), 0.01) # Default value of threshold
    thresh_vec = np.arange(thresh_range[0],thresh_range[1],thresh_range[2])
    for nFFT_idx in range(len(nFFT_list)):
        for ind in range((len(thresh_vec))):
            fft_data = data_FFT(data[:nSamplesTune,:], nperseg, nFFT_list[nFFT_idx])  
            C_FFT = Run_PC_Wiener_test(fft_data,thresh_vec[ind],freq_choice,q)
            Top_est_half = np.zeros([nNodes,nNodes])
            for c in C_FFT:
                if C_FFT[c] == None: Top_est_half[c]=1
            Top_est_FFT = (Top_est_half + Top_est_half.T)
            min_deg, max_deg, avg_deg = compute_undir_graph_degrees(Top_est_FFT)
            if avg_deg >= avg_deg_lim[0] and  avg_deg <= avg_deg_lim[1] and max_deg <= max_deg_lim and min_deg >= min_deg_lim:
                thresh[nFFT_idx] = thresh_vec[ind]
                break
        print(f"Chosen W-PC threshold at nFFT {nFFT_list[nFFT_idx]} is {thresh[nFFT_idx]}") 
    
    return thresh