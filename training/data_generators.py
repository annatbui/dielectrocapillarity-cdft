import numpy as np
import tensorflow as tf
from numpy.lib.recfunctions import structured_to_unstructured
from tensorflow import keras
import itertools


# Check whether TensorFloat-32 execution is currently enabled
tf.config.experimental.enable_tensor_float_32_execution(False)

class DataGeneratorThreeInput(keras.utils.PyDataset):
    """
    Custom Keras data generator for simulation data with a sliding window.
    To be passed to the fit_generator method of a Keras model.
    Follow template from https://github.com/afshinea/keras-data-generator 
    
    Parameters
    ----------
    simData : dict
        Dictionary containing simulation data.
    
    batch_size : int
        Number of samples per batch.
    
    shuffle : bool
        Whether to shuffle the data at the end of each epoch.
    
    inputKeys : list
        List of keys for input data 1.

    inputKeys : list
        List of keys for input data 1.
        
    outputKeys : list
        List of keys for output data 1.

    windowSigma : float
        Width of the sliding window
    """
    
    def __init__(
            self,
            simData,
            batch_size=32,
            steps_per_execution=1,
            shuffle=True,
            inputKeys1=["rho"],
            inputKeys2=["phi"],
            paramsKeys=[],
            outputKeys=["c1"],
            binKey="xbins",
            window1Sigma=2.0,
            window2Sigma=2.0,
            filt=lambda sim: True,
            **kwargs):
        
        super().__init__(**kwargs)
        
        self.simData = {key: sim for key, sim in simData.items() if filt(sim)}
        print(f"Loaded {len(self.simData)} simulations")
        
        self.inputKeys1 = inputKeys1
        self.inputKeys2 = inputKeys2
        self.paramsKeys = paramsKeys
        self.outputKeys = outputKeys
        self.window1Sigma = window1Sigma
        self.window2Sigma = window2Sigma
        
        firstSimData = list(self.simData.values())[0]
        self.dz = firstSimData["profiles"][binKey][0]*2
        self.simDataBins = len(firstSimData["profiles"][binKey])
        self.window1Bins = int(round(self.window1Sigma/self.dz))
        self.window2Bins = int(round(self.window2Sigma/self.dz))
        
        self.validBins = {}
        self.inputData1Padded = {}
        self.inputData2Padded = {}
        for simId in self.simData.keys():
            profile_shape = self.simData[simId]["profiles"][self.outputKeys[0]].shape
            valid = np.full(profile_shape, True)
            for k in self.outputKeys:
                valid = np.logical_and(valid, ~np.isnan(self.simData[simId]["profiles"][k]))
            self.validBins[simId] = np.flatnonzero(valid)
            self.inputData1Padded[simId] = np.pad(self.simData[simId]["profiles"][self.inputKeys1], self.window1Bins, mode="wrap")
            self.inputData2Padded[simId] = np.pad(self.simData[simId]["profiles"][self.inputKeys2], self.window2Bins, mode="wrap")
        
        self.batch_size = batch_size
        self.steps_per_execution = steps_per_execution
        self.input1Shape = (2*self.window1Bins+1,)
        self.input2Shape = (2*self.window2Bins+1,)
        self.outputShape = (len(self.outputKeys),)
        
        self.shuffle = shuffle
        self.on_epoch_end()
        
        print(f"Initialized DataGenerator from {len(self.simData)} simulations which will yield up to {len(self.indices)} input/output samples in batches of {self.batch_size}")

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.indices) / (self.batch_size * self.steps_per_execution))) * self.steps_per_execution

    def __getitem__(self, index):
        """
        Generates one batch of data.
        """
        ids = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        profiles1 = {key: np.empty((self.batch_size, *self.input1Shape)) for key in self.inputKeys1}
        profiles2 = {key: np.empty((self.batch_size, *self.input2Shape)) for key in self.inputKeys2}
        params = {key: np.empty((self.batch_size, 1)) for key in self.paramsKeys}
        output = {key: np.empty((self.batch_size, *self.outputShape)) for key in self.outputKeys}
        for b, (simId, i) in enumerate(ids):
            for key in self.inputKeys1:
                profiles1[key][b] = self.inputData1Padded[simId][key][i:i+2*self.window1Bins+1]
            for key in self.inputKeys2:
                profiles2[key][b] = self.inputData2Padded[simId][key][i:i+2*self.window2Bins+1]
            for key in self.paramsKeys:
                params[key][b] = self.simData[simId]["params"][key]
            for key in self.outputKeys:
                output[key][b] = self.simData[simId]["profiles"][key][i]
        return (profiles1 | profiles2 | params), output

    def on_epoch_end(self):
        """
        Updates indices after each epoch.
        """
        self.indices = []
        for simId in self.simData.keys():
            self.indices.extend(list(itertools.product([simId], list(self.validBins[simId]))))
        if self.shuffle == True:
            np.random.default_rng().shuffle(self.indices)

    def pregenerate(self):
        print("Pregenerating data from DataGenerator")
        batch_size_backup = self.batch_size
        self.batch_size *= len(self)
        data = self[0]
        self.batch_size = batch_size_backup
        return data
