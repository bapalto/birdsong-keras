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

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D,  MaxPooling2D 

# Strongly AlexNet based convolutional neural network

model = Sequential()

# convolutional layer
model.add(Convolution2D(input_shape=(1,200,310),
                        nb_filter=48*2,
                        nb_row=16,
                        nb_col=16,
                        border_mode='valid',
                        init='glorot_normal', #glorot_normal lecun_uniform he_uniform
                        activation='relu',
                        subsample=(6, 6)
                        ))
model.add(BatchNormalization())                        
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(nb_filter=128*2,
                        nb_row=3,
                        nb_col=3,
                        border_mode='valid',
                        init='lecun_uniform', #glorot_normal lecun_uniform he_uniform
                        activation='relu'
                        ))  
model.add(BatchNormalization())                        
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(nb_filter=192*2,
                        nb_row=3,
                        nb_col=3,
                        border_mode='same',
                        init='lecun_uniform', #glorot_normal lecun_uniform he_uniform
                        activation='relu'
                        ))  

model.add(Convolution2D(nb_filter=192*2,
                        nb_row=3,
                        nb_col=3,
                        border_mode='same',
                        init='lecun_uniform', #glorot_normal lecun_uniform he_uniform
                        activation='relu'
                        ))  

model.add(Convolution2D(nb_filter=128*2,
                        nb_row=3,
                        nb_col=3,
                        border_mode='same',
                        init='lecun_uniform', #glorot_normal lecun_uniform he_uniform
                        activation='relu'
                        ))  

model.add(BatchNormalization())                        
model.add(MaxPooling2D(pool_size=(2,2)))

# dense layers                  
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim = output_dim, activation='softmax'))
