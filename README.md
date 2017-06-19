# Trainig scripts for deep convolutional neural network based audio classification in Keras

The following scripts were created for the BirdCLEF 2016 competition by Bálint Czeba and Bálint Pál Tóth.

*The LifeCLEF bird identification challenge provides a largescale
testbed for the system-oriented evaluation of bird species identifi-
cation based on audio recordings. One of its main strength is that the
data used for the evaluation is collected through Xeno-Canto, the largest
network of bird sound recordists in the world. This makes the task closer
to the conditions of a real-world application than previous, similar initiatives.
The main novelty of the 2016-th edition of the challenge was the
inclusion of soundscape recordings in addition to the usual xeno-canto
recordings that focus on a single foreground species. This paper reports
the methodology of the conducted evaluation, the overview of the systems
experimented by the 6 participating research groups and a synthetic
analysis of the obtained results.* (More details: http://www.imageclef.org/lifeclef/2016/bird)

With some tweeks (reading meta-data and modifing network structure / how the spectogram is preprocessed) it is possible to apply it to arbitrary audio classification problems.

# Citation

Please cite the following paper if this code was useful for your research:

*Tóth Bálint Pál, Czeba Bálint, "Convolutional Neural Networks for Large-Scale Bird Song Classification in Noisy Environment", In: Working Notes of Conference and Labs of the Evaluation Forum, Évora, Portugália, 2016, p. 8*
Download from here (PDF): http://ceur-ws.org/Vol-1609/16090560.pdf

```
@article{tothczeba,
    author =       "B\'{a}lint P\'{a}l T\'{o}th, B\'{a}lint Czeba",
    title =        "{Convolutional Neural Networks for Large-Scale Bird Song Classification in Noisy Environment}",
    booktitle =    "{Working Notes of Conference and Labs of the Evaluation Forum},
    pages =        "8",
    year =         "2016",
}
```

# Prerequisites
You will need SOX for wave file resampling and Keras deep learning frameworks and some necessary modules. At the time of writeing you can install them in the following way:
```
sudo apt-get install sox
sudo apt-get install python-tk
sudo pip install scipy
sudo pip install matplotlib
sudo pip install sklearn
sudo pip install tensorflow-gpu
sudo pip install keras
```
The code is tested under Python 2.7. with TensorFlow (GPU) 1.0.0a0 and Keras 1.1.1. backend, NVidia Titan X 12GB GPU.

If you use TensorFlow as a backend with Keras 1.x you should set
```
"image_dim_ordering": "th",
```
in ~/.keras/keras.json configuration file.

In Keras 2 "image_dim_ordering" is deprecated. If you use TensorFlow + Keras 2.x, you should change the "image_data_format" setting to "channels_first".

# Directory structure and files 
```
doall.sh                - run this script and it will do everything (you will need plenty of disk space > 100 GB)
preprocess/loadData.py  - responsible for preprocessing the data (wavs and XML meta-data)
preprocess/sample_wavs_to_16k.sh - simple script that resamples wave files to 16 kHz with SOX
preprocess/xmltodict.py - XML processing from https://github.com/martinblech/xmltodict
train/trainModel.py     - after preprocessing this script trains the neural networks
train/model-AlexNet.py  - AlexNet inspired model for audio classification
train/model-BirdClef.py - Another convolutional neural net model for audio classification
train/MAPCallback.py    - Script to calculate MAP scores during training the neural nets
train/generateImages.py - Generate images from the preprocessed spectogram for visualization purposes
train/io_utils_mod.py   - Functions for loading and saving data to HDF5
train/log.py            - Functions for logging purposes
predict/predict.py      - Predict after preprocessing and training is done
```

# Training (and download data and preprocess)

For training you have to simply run
```
./doall.sh
```
Be aware that this will download all the data (>50 GB) from http://otmedia.lirmm.fr/LifeCLEF/BirdCLEF2016/ to 
```
birdclef_data
```
directory, unpack it and resample to 16 kHz and preprocess it into HDF5 files. You need cca. 280 GB of free space for the whole process. If you would like to put the data to somewhere else, please modify the doall.sh, preprocess/loadData.py and train/trainModel.py scripts.

The download process, preprocessing and training takes 4-5 days on an i7 CPU + Titan X GPU.

# Prediction

After the preprocessing and training is do simpy run the following script to make predictions on test data:

```
./predict.sh
```
The prediction results will be written in a .csv file in the predict/ directory.
