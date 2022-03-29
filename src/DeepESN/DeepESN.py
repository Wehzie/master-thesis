import numpy as np
import numpy.matlib as npm
import scipy as sc
import random
import sys

DEBUG = False

class DeepESN():
    
    '''
    Deep Echo State Network (DeepESN) class:
    this class implement the DeepESN model suitable for 
    time-series prediction and sequence classification.

    Reference paper for DeepESN model:
    C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A
    Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99
    
    Reference paper for the design of DeepESN model in multivariate time-series prediction tasks:
    C. Gallicchio, A. Micheli, L. Pedrelli, "Design of deep echo state networks",
    Neural Networks, 2018, vol. 108, pp. 33-47 
    
    ----

    This file is a part of the DeepESN Python Library (DeepESNpy)

    Luca Pedrelli
    luca.pedrelli@di.unipi.it
    lucapedrelli@gmail.com

    Department of Computer Science - University of Pisa (Italy)
    Computational Intelligence & Machine Learning (CIML) Group
    http://www.di.unipi.it/groups/ciml/

    ----
    '''
    
    def __init__(self, Nu,Nr,Nl, configs, verbose=0):
        # initialize the DeepESN model
        
        if verbose:
            sys.stdout.write('init DeepESN...')
            sys.stdout.flush()
        
        rhos = np.array(configs.rhos) # spectral radius (maximum absolute eigenvalue)
        lis = np.array(configs.lis) # leaky rate
        iss = np.array(configs.iss) # input scale
        IPconf = configs.IPconf # configuration for Deep Intrinsic Plasticity
        reservoirConf = configs.reservoirConf # reservoir configurations
        
        if len(rhos.shape) == 0:
            rhos = npm.repmat(rhos, 1,Nl)[0]

        if len(lis.shape) == 0:
            lis = npm.repmat(lis, 1,Nl)[0]

        if len(iss.shape) == 0:
            iss = npm.repmat(iss, 1,Nl)[0]
            
        self.W = {} # recurrent weights
        self.Win = {} # recurrent weights
        self.Gain = {} # activation function gain
        self.Bias = {} # activation function bias

        self.Nu = Nu # number of inputs
        self.Nr = Nr # number of units per layer
        self.Nl = Nl # number of layers
        self.rhos = rhos.tolist() # list of spectral radius
        self.lis = lis # list of leaky rate
        self.iss = iss # list of input scale

        self.IPconf = IPconf   
        
        self.readout = configs.readout
               
        # sparse recurrent weights init
        if reservoirConf.connectivity < 1:
            for layer in range(Nl):
                self.W[layer] = np.zeros((Nr,Nr))
                for row in range(Nr):
                    number_row_elements = round(reservoirConf.connectivity * Nr)
                    row_elements = random.sample(range(Nr), number_row_elements)
                    self.W[layer][row,row_elements] = np.random.uniform(-1,+1, size = (1,number_row_elements))
                    
        # full-connected recurrent weights init      
        else:
            for layer in range(Nl):
                self.W[layer] = np.random.uniform(-1,+1, size = (Nr,Nr))
        
        # layers init
        for layer in range(Nl):

            target_li = lis[layer]
            target_rho = rhos[layer]
            input_scale = iss[layer]

            # TODO: search over distributions, introduce more zero weights
            if layer==0:
                # mapping between input dim and layer dim
                self.Win[layer] = np.random.uniform(-input_scale, input_scale, size=(Nr,Nu+1))
            else:
                # mapping between layer dim and layer dim
                self.Win[layer] = np.random.uniform(-input_scale, input_scale, size=(Nr,Nr+1))

            Ws = (1-target_li) * np.eye(self.W[layer].shape[0], self.W[layer].shape[1]) + target_li * self.W[layer]
            eig_value,eig_vector = np.linalg.eig(Ws)  
            actual_rho = np.max(np.absolute(eig_value))

            Ws = (Ws *target_rho)/actual_rho
            self.W[layer] = (target_li**-1) * (Ws - (1.-target_li) * np.eye(self.W[layer].shape[0], self.W[layer].shape[1]))
            
            self.Gain[layer] = np.ones((Nr,1))
            self.Bias[layer] = np.zeros((Nr,1)) 
         
        if verbose:
            print('done.')
            sys.stdout.flush()
    
    def computeLayerStateDeepIP(self,
        raw_input: np.ndarray,
        layer,
        initialStatesLayer: np.ndarray = None,
    ) -> np.ndarray:  
        """
        This function computes state transitions using intrinsic plasticity optimization.
        Firstly, the raw input is mapped to the reservoir dimension with an input matrix.
        The input may also be coming from a previous layer and already have the correct dimension; still it will be weighted.
        Secondly, the state for the first time step, t=0, is computed.
        Third, the states for all subsequent time steps of the time series are computed.
        """
        
        time_series_len = raw_input.shape[1] # length of a time series, for example one MIDI
        l_states = np.zeros((self.Nr, time_series_len)) # states of a layer across time-series
        
        if initialStatesLayer is None:
            # None when intrinsic plasticity is not used
            # dimension is (nodes_in_layer, 1)
            # each node of the reservoir has zero activation
            initialStatesLayer = np.zeros(l_states[:,0:1].shape)
        
        # DIMENSIONS
        # raw_input.shape == (data_point_dim, time_series_len)
        # self.Win[layer][:,0:-1].shape == (nodes_in_layer, data_point_dim)
        # the above dimension pairs are used as mappings between dimensions
        # np.expand_dims(self.Win[layer][:,-1],1).shape == (nodes_in_layer, 1)

        # the raw_input directly passed to this function is a data point
        # the last element of the weight vector is the bias
        # by [:,0:-1] the last column is excluded
        # by [:,-1] the last column is selected
        weights = self.Win[layer][:,0:-1]
        biases = np.expand_dims(self.Win[layer][:,-1], axis=1)
        # weighted input is result of map between raw input and reservoir dimension via input weight matrix
        w_input = weights.dot(raw_input) + biases
        # w_input.shape == (nodes_in_layer, time_series_len)

        state_net = np.zeros((self.Nr, w_input.shape[1]))
        state_net[:,0:1] = w_input[:,0:1]
        l_states[:,0:1] = self.lis[layer] * np.tanh(np.multiply(self.Gain[layer], state_net[:,0:1]) + self.Bias[layer])

        # compute reservoir states for times, t, in the time-series where t > 0
        for t in range(1, l_states.shape[1]):
            if DEBUG: print(f"computeLayerState: t {t}")
            state_net[:,t:t+1] = self.W[layer].dot(l_states[:,t-1:t]) + w_input[:,t:t+1]
            l_states[:,t:t+1] = (1-self.lis[layer]) * l_states[:,t-1:t] + self.lis[layer] * np.tanh(np.multiply(self.Gain[layer], state_net[:,t:t+1]) + self.Bias[layer])
            
            eta = self.IPconf.eta
            mu = self.IPconf.mu
            sigma2 = self.IPconf.sigma**2
        
            # IP learning rule
            deltaBias = -eta*((-mu/sigma2)+ np.multiply(l_states[:,t:t+1], (2*sigma2+1-(l_states[:,t:t+1]**2)+mu*l_states[:,t:t+1])/sigma2))
            deltaGain = eta / npm.repmat(self.Gain[layer],1,state_net[:,t:t+1].shape[1]) + deltaBias * state_net[:,t:t+1]
            
            # update gain and bias of activation function
            self.Gain[layer] = self.Gain[layer] + deltaGain
            self.Bias[layer] = self.Bias[layer] + deltaBias
                
        return l_states


    def computeLayerState(self,
        raw_input: np.ndarray,
        layer,
        initialStatesLayer: np.ndarray = None,
    ) -> np.ndarray:  
        """
        Compute state transitions for a given layer.
        """
        
        time_series_len = raw_input.shape[1] # length of a time series, for example one MIDI
        l_states = np.zeros((self.Nr, time_series_len)) # states of a layer across time-series
        
        if initialStatesLayer is None:
            # always None
            # maybe this could be useful for saving and loading?
            # dimension is (nodes_in_layer, 1)
            # each node of the reservoir has zero activation
            initialStatesLayer = np.zeros(l_states[:,0:1].shape)
        
        # DIMENSIONS
        # raw_input.shape == (data_point_dim, time_series_len)
        # self.Win[layer][:,0:-1].shape == (nodes_in_layer, data_point_dim)
        # the above dimension pairs are used as mappings between dimensions
        # np.expand_dims(self.Win[layer][:,-1],1).shape == (nodes_in_layer, 1)

        # the raw_input directly passed to this function is a data point
        # the last element of the weight vector is the bias
        # by [:,0:-1] the last column is excluded
        # by [:,-1] the last column is selected
        weights = self.Win[layer][:,0:-1]
        biases = np.expand_dims(self.Win[layer][:,-1], axis=1)
        # weighted input is result of map between raw input and reservoir dimension via input weight matrix
        w_input = weights.dot(raw_input) + biases
        # w_input.shape == (nodes_in_layer, time_series_len)
        
        # w_input[:,0:1] gives all rows at column 0
        # print(len(w_input[:,0:1])) gives 100 which is the number of nodes per reservoir
        # w_input[:,0:1] then refers to the activation or state of each node for t=0
        
        # compute reservoir state for first time step of the time-series
        l_states[:,0:1] = (1-self.lis[layer]) * initialStatesLayer + self.lis[layer] * np.tanh( np.multiply(self.Gain[layer], self.W[layer].dot(initialStatesLayer) + w_input[:,0:1]) + self.Bias[layer])        
 
        # compute reservoir states for times, t, in the time-series where t > 0
        for t in range(1, l_states.shape[1]):
            l_states[:,t:t+1] = (1-self.lis[layer]) * l_states[:,t-1:t] + self.lis[layer] * np.tanh( np.multiply(self.Gain[layer], self.W[layer].dot(l_states[:,t-1:t]) + w_input[:,t:t+1]) + self.Bias[layer])
                
        return l_states

    def computeDeepIntrinsicPlasticity(self, inputs):
        # we incrementally perform the pre-training (deep intrinsic plasticity) over layers
        
        len_inputs = range(len(inputs)) # number of unique time-series
        states = [] # for each time-series store the state of all layers across all time steps
        
        # loop over unique time-series
        for i in len_inputs:
            time_series_len = inputs[i].shape[1] # duration of a time series in discrete steps
            # initialize reservoirs with zero activation for each time step
            states.append(np.zeros((self.Nr*self.Nl, time_series_len)))
        
        # loop over layers (reservoirs)
        for layer in range(self.Nl):
            # maximum number of training epochs
            for epoch in range(self.IPconf.Nepochs):
                print(f"DeepIP epoch {epoch}.")
                print(len(inputs))
                Gain_epoch = self.Gain[layer]
                Bias_epoch = self.Bias[layer]

                # computeLayerStateDeepIP writes to Gain and Bias fields
                if len(inputs) == 1: # only one time-series
                    self.computeLayerStateDeepIP(inputs[0][:,self.IPconf.indexes], layer)
                else:
                    # each index points to a time-series
                    for i in self.IPconf.indexes:
                        self.computeLayerStateDeepIP(inputs[i], layer)
                
                # compute difference between old and new gain and bias vectors
                gain_diff = self.Gain[layer]-Gain_epoch
                bias_diff = self.Bias[layer]-Bias_epoch
                # compute l2-norms of the difference vectors
                # stop optimization when l2-norm below threshold
                if (np.linalg.norm(gain_diff, ord=2) < self.IPconf.threshold
                    and np.linalg.norm(bias_diff, ord=2) < self.IPconf.threshold):
                    break
            
            # regular layer states are computed after intrinsic plasticity optimization
            layer_states = []
            for i in range(len(inputs)): # number of time-series
                layer_states.append(self.computeLayerState(inputs[i], layer))
            
            # write layer_states to the states-variable that tracks all time-series, layers and time steps
            for i in range(len(inputs)):
                states[i][(layer)*self.Nr: (layer+1)*self.Nr,:] = layer_states[i]
            
        return states
    
    # TODO: consider function decorators for verbose printing
    def computeState(self, inputs: list, DeepIP: bool=False, initialStates: None=None, verbose: int=0) -> np.ndarray:
        """
        Iterate over each time-series and compute the state of each reservoir (layer) for each time step.
        
        Args:
            inputs: 
            DeepIP: Whether to use intrinsic plasticity.
            initialStates: Unclear why this needs to be passed. Probably always None.
            verbose: Verbose printing.

        Returns:
            An array called "state" with the state of each reservoir for each time step for each time-series.
        """
        
        if DeepIP:
            if verbose:
                print('compute state with DeepIP...')
                # receives all time-series as input
            states = self.computeDeepIntrinsicPlasticity(inputs)
        else:      
            if verbose:
                print('compute state...')
            states = []

            for i_seq in range(len(inputs)):
                # receives a single time-series as input
                states.append(self.computeGlobalState(inputs[i_seq], initialStates))
                
        if verbose:        
            print('done.')
        
        return states
    
    def computeGlobalState(self, input, initialStates):
        # compute the global state of DeepESN

        midi_duration = input.shape[1] # number of discrete time steps in a midi

        # each time step in the midi corresponds to a state of the RNN
        state = np.zeros((self.Nl*self.Nr, midi_duration))
        # example state has dim (500, 346) where
        # 500 = number of layers * number of nodes per layer
        # 346 = the number of discrete time steps in a time series, for example MIDI
        
        initialStatesLayer = None

        for layer in range(self.Nl):
            if DEBUG: print(f"computeGlobalState: layer {layer}")

            if initialStates is not None:
                # NOTE: I think initialStates is always None?
                # theory: initialStates it NOT None when intrinsic plasticity is computed

                # initalStatesLayer seems to be an int used as an index to a layer
                # OOP not nicely used here
                print(initialStates)
                initialStatesLayer = initialStates[layer*self.Nr : (layer+1)*self.Nr,:]            

            # write to state
            state[(layer)*self.Nr: (layer+1)*self.Nr,:] = self.computeLayerState(input, layer, initialStatesLayer)    
            
            # override input
            # the ndarray slice is to be read as follows
            # [row_x : to row_y, across all columns]
            input = state[(layer)*self.Nr: (layer+1)*self.Nr,:]

        return state
        
    def trainReadout(self,trainStates,trainTargets,lb, verbose=0):
        # train the readout of DeepESN

        trainStates = np.concatenate(trainStates,1)
        trainTargets = np.concatenate(trainTargets,1)
        
        # add bias
        X = np.ones((trainStates.shape[0]+1, trainStates.shape[1]))
        X[:-1,:] = trainStates    
        trainStates = X  
        
        if verbose:
            sys.stdout.write('train readout...')
            sys.stdout.flush()
        
        if self.readout.trainMethod == 'SVD': # SVD, accurate method
            U, s, V = np.linalg.svd(trainStates, full_matrices=False);  
            s = s/(s**2 + lb)
                      
            self.Wout = trainTargets.dot(np.multiply(V.T, np.expand_dims(s,0)).dot(U.T));
            
        else:  # NormalEquation, fast method
            B = trainTargets.dot(trainStates.T)
            A = trainStates.dot(trainStates.T)

            self.Wout = np.linalg.solve((A + np.eye(A.shape[0], A.shape[1]) * lb), B.T).T

        if verbose:
            print('done.')
            sys.stdout.flush()
        
    def computeOutput(self,state):
        # compute a linear combination between the global state and the output weights  
        state = np.concatenate(state,1)
        return self.Wout[:,0:-1].dot(state) + np.expand_dims(self.Wout[:,-1],1) # Wout product + add bias
