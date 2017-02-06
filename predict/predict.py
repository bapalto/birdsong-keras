# -*- coding: utf-8 -*-
#
# Birdsong classificatione in noisy environment with convolutional neural nets in Keras
# Copyright (C) 2017 Báint Czeba, Bálint Pál Tóth (toth.b@tmit.bme.hu)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. (c) Balint Czeba, Balint Pal Toth
# 
# Please cite the following paper if this code was useful for your research:
# 
# Bálint Pál Tóth, Bálint Czeba,
# "Convolutional Neural Networks for Large-Scale Bird Song Classification in Noisy Environment", 
# In: Working Notes of Conference and Labs of the Evaluation Forum, Évora, Portugália, 2016, p. 8

from scipy import io
from scipy.io import wavfile
import pandas as pd
import numpy as np
np.random.seed(0)
import time
import pickle
import os
import h5py
import sys, getopt
import datetime
from sklearn.metrics import average_precision_score, accuracy_score
sys.path.append('../preprocess/')
import loadData

# prediction related paths, should be consistent with /preprocess/loadData.py and /train/trainModel.py
PATH_TEST_IN_16KWAVS            = '../birdclef_data/test/wav_16khz'
PATH_TEST_IN_XMLPICKLEFILE      = '../birdclef_data/test/xml_data.pickle'
modelPath			= '../train/model-AlexNet.py'
modelWeightsPath		= '../train/modelWeights/best_val_map_999.hdf5' 
labelBinarizerPath 		= "../birdclef_data/labelBinarizer_top999.pickle"

output_dim			= 999
scalerFilePath = None

if __name__ == "__main__":
    argv=sys.argv[1:]

try:
    opts, args = getopt.getopt(argv,"he:p:m:s:",["nbepochs=","hdf5path=","scalerpath"])
except getopt.GetoptError:
    print 'trainModel.py -p <hdf5 file path>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'trainModel.py -p <hdf5 file path> -m <model.py>'
        sys.exit()
    elif opt in ("-p", "--p"):
        hdf5path = arg
    elif opt in ("-m", "--m"):
        modelPath = arg
    elif opt in ("-s", "--s"):
        scalerFilePath = arg

# function to convert probabilities to classes
def proba_to_class(a):
    classCount	= len(a[0])
    to_return	= np.empty((0,classCount))
    for row in a:
        maxind	= np.argmax(row)
        to_return = np.vstack((to_return,[1 if i==maxind else 0 for i in range(classCount)]))
    return to_return
 
# run preprocessing and prediction on one file
def runModelOnWavByPath(model, path):
    (tempSpecUnfiltered, tempSpecFiltered) = loadData.audioToFilteredSpectrogram(io.wavfile.read(path)[1], expandByOne=True)
    tempList = list()
    tempList.append(tempSpecFiltered)
    X,_,_,_ = loadData.spectrogramListToT4(tempList, N = 5*62) # the N must be consistent with /preprocess/loadData.py 
    result = model.predict(X)
    
    return result

scaler		= None
scaleData	= None
if scalerFilePath is not None:
    scaler = pickle.load(open(scalerFilePath, 'rb'))
    # Can't use the build in scaler.transform because it only supports 2d arrays.
    def scaleData(X):
        return (X-scaler.mean_)/scaler.scale_

# load the meta-data saved into a pickle file
df = pd.read_pickle(PATH_TEST_IN_XMLPICKLEFILE)
df = df.iloc[np.random.permutation(len(df))]
df.reset_index(drop=True, inplace=True)
lb = pickle.load(open(labelBinarizerPath, 'rb'))

# calculate prediction time
startTime = time.time()

# Build model and load saved weights
execfile(modelPath)
model.load_weights(modelWeightsPath)

ap=[]
i=0

# writeing predictions
resultColumns = lb.inverse_transform(np.diag([1 for i in range(999)]))
resultsFileName = "test_2015_{}_{}.csv".format(os.path.split(modelPath)[-1], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

# running test with all the data
for i in range(int(df.shape[0]*0.0),df.shape[0]):
    print ('{}/{}'.format(i, df.shape[0]))
    result = runModelOnWavByPath(model, os.path.join(PATH_TEST_IN_16KWAVS, df.FileName[i]))    

    result_avg = np.mean(result, axis=0)
    result_avg = result_avg/np.sum(result_avg)
    
    singleResultDF = pd.DataFrame([result_avg], columns = resultColumns)
    singleResultDF['MediaId'] = df.MediaId[i]
    with open(resultsFileName, "a") as myfile:
        for row in singleResultDF.iterrows():
            for k,v in row[1].iterkv():
                if (k is not 'MediaId'):
                    resultLine = "{};{};{:.16f}\n".format(row[1].MediaId, k, v)
                    myfile.write(resultLine)
    
elapsed = time.time()-startTime;
print("Execution time: {0} s".format(elapsed))
