#!/bin/bash
mkdir -p birdclef_data
cd birdclef_data
wget -nc http://otmedia.lirmm.fr/LifeCLEF/BirdCLEF2016/BirdCLEF2016TestSet.tar.gz
wget -nc http://otmedia.lirmm.fr/LifeCLEF/BirdCLEF2016/BirdCLEF2016TrainingSet.tar.gz
unp *.tar.gz
cd ..
preprocess/resample_wavs_to_16k.sh birdclef_data/TrainingSet/wav
preprocess/resample_wavs_to_16k.sh birdclef_data/test/wav2015
cd preprocess
python loadData.py
cd ..
cd train
mkdir modelWeights
python trainModel.py

