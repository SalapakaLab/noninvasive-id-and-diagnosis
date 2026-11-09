import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def Wiener_Proj(X,Z):
    '''
    Calculates the Wiener filter coefficients for projection of X on to Z, Z=a0x_0+a1x1+...am xm, where m is the number of columns of X
    Inputs:  (1) X: A node.
            (2) Z: A set of nodes to project on.
    Outputs: (1) Linear lest squares estimate of the Wiener filters.
    '''

    X_star = X.conj().T
    XX=np.matmul(X_star, X)
    if np.isscalar(XX):
        return np.matmul(np.divide( X_star,XX), Z)
    else:
        return np.matmul(np.matmul(np.linalg.inv(np.matmul(X_star, X)), X_star), Z)    


    

def compute_Wiener_coeffs(nNodes,data):
    '''
    Calculates the Wiener filter coefficients for a complete projection
    Inputs:  (1) nNodes: Number of nodes in graph.
            (2) data: FFT data to be used for projection.
    Outputs: (1) W: Array of dimension (nNodes X nSamples X nNodes) containing calculated Wiener projections at each frequency.
            (2) W_mag: Magnitude of the Wiener filters
    '''
    nSamples=data.shape[2]
    W=np.zeros([nNodes,nSamples,nNodes],dtype='complex')
    W_mag=np.zeros([nNodes,nNodes])
    for proj_ind in range(nNodes):
        W1=np.zeros([nSamples,nNodes],dtype='complex')
        for freq_index in range(nSamples):
            fft_data=data[:,:,freq_index]
            Z=fft_data[:,proj_ind]
            X_bar=np.delete(fft_data,proj_ind,1)
            W1[freq_index,:]=np.insert(Wiener_Proj(X_bar,Z),proj_ind,0)
        W[proj_ind,:,:]=W1
        W_mag[proj_ind,:]=np.amax(abs(W1),axis=0)
    return W,W_mag





def compute_partial_Wiener_coeffs_avg(data,i,j,z=[],freq_choice=[]):
    '''
    Compute partial wiener coefficients for partial projections.
    Inputs:  (1) data: Data to be used for the independence test
            (2) i: First node.
            (3) j: Second node.
            (4) z: Set of nodes to be conditioned on. 
            (5) freq_choice: Frequencies over which the average magnitudes are to be calculated.    
    Outputs: (1) W: Array containing calculated Wiener projections at each frequency.
            (2) W_mag_avg: Magnitude of the Wiener filters averaged over the cosen frequencies
    '''
    # Returns W and W_mag
    # W: nNodes X n_FFT X nNodes
    # W_mag: returns h-infinity norm over frequencies
    c=np.append(j,np.int32(z))
    # c=np.unique(c)
    indexes = np.unique(c, return_index=True)[1]
    c=[c[index] for index in sorted(indexes)]
    n1=len(c)
    n_FFT=data.shape[2]
    W=np.zeros([n_FFT,n1],dtype='complex')
    # W_mag=np.zeros([n1])
    W_mag_avg = np.zeros([n1],dtype=float)
    for freq_index in freq_choice:
        fft_data=data[:,:,freq_index]
        # print(fft_data.shape)
        # print(freq_index)
        X=fft_data[:,i]
        X_bar=fft_data[:,c]
        # print(Z.shape,X_bar.shape)
        # print(X_bar[:3,:])
        W[freq_index,:]=Wiener_Proj(X_bar,X)
        W_mag_avg = W_mag_avg + np.abs(W[freq_index,:])
    W_mag_avg = np.divide(W_mag_avg,len(freq_choice))
    # W_mag=np.amax(abs(W),axis=0)
    return W,W_mag_avg




def indepTest(X,Y,S,data,freqChoice,threshold):
    '''
    Perform conditional independence test (wiener separation test) on given data.
    Inputs:  (1) X: First node.
            (2) Y: Second node.
            (3) Z: Set of nodes to be conditioned on.
            (4) data: Data to be used for the independence test.
            (5) freqChoice: Frequencies over which the average magnitudes are to be calculated 
                            for the Wiener filters in the conditional independence checking step.
            (6) threshold: Threshold on the magnitudes of Wiener filters used for conditional independence test.
    Outputs: (1) result: True if X and Y are Wiener separated given Z, False otherwise.
    '''
    W,W_mag_avg = compute_partial_Wiener_coeffs_avg(data,X,Y,S,freqChoice)
    if W_mag_avg[0] < threshold:
        result = True
    else:
        result = False
    return result



def compute_undir_graph_degrees(Adj):
    '''
    Find the minimum, maximum, and average degree of a undirected graph from its adjacency matrix
    Inputs: (1) Adj: Adjacency matrix of the graph for which the degrees need to be calculated
    Output: (1) min_deg: Minimum degree of the given graph
            (2) max_deg: Maximum degree of the given graph
            (3) avg_deg: Average degree of the given graph
    '''
    nNodes = Adj.shape[0]
    node_degs = []
    for node in range(nNodes):
        node_degs.append(sum(Adj[:,node]))
    max_deg = max(node_degs)
    min_deg = min(node_degs)
    avg_deg = sum(node_degs)/nNodes
    
    return min_deg, max_deg, avg_deg