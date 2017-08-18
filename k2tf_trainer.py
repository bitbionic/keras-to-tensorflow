'''
This script builds and trains a simple Convolutional Nerual Network (CNN)
against a supplied data set. It is used in a tutorial demonstrating
how to build Keras models and run them in native C++ Tensorflow applications.


MIT License

Copyright (c) 2017 bitbionic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from datetime import datetime
import os
import argparse

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import ZeroPadding2D

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator


def buildClassifier( img_shape=128, num_categories=5 ):
    '''
    Builds a very simple CNN outputing num_categories.
    
    Args:
             img_shape (int): The shape of the image to feed the CNN - defaults to 128
        num_categories (int): The number of categories to feed the CNN
 
    Returns:
        keras.models.Model: a simple CNN
    
    '''
    classifier = Sequential()

    # Add our first convolutional layer
    classifier.add( Conv2D( filters=32,
                            kernel_size=(2,2),
                            padding='same',
                            data_format='channels_last',
                            input_shape=(img_shape,img_shape,3),
                            activation = 'relu',
                            name = 'firstConv2D'
                            ) )

    # Pooling
    classifier.add( MaxPooling2D(pool_size=(2,2), name='firstMaxPool') )

    # Add second convolutional layer.
    classifier.add( ZeroPadding2D(padding=(2,2)) )
    classifier.add( Conv2D( filters=16,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'secondConv2D'
                            ) 
                  )
    classifier.add( MaxPooling2D(pool_size=(2,2), name='secondMaxPool') )
    
    # Add second convolutional layer.
    classifier.add( ZeroPadding2D(padding=(2,2)) )
    classifier.add( Conv2D( filters=8,
                            kernel_size=(2,2),
                            activation = 'relu',
                            name = 'thirdc2'
                            ) 
                  )
    classifier.add( MaxPooling2D(pool_size=(2,2), name='thirdpool') )


    # Flattening
    classifier.add( Flatten(name='flat') )

    # Add Fully connected ANN
    classifier.add( Dense( units=256, activation='relu', name='fc256') )
    classifier.add( Dense( units=num_categories, activation = 'softmax', name='finalfc'))

    # Compile the CNN
    #classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier


def trainModel( classifier, trainloc, testloc, img_shape, output_dir='./', batch_size=32, num_epochs=30 ):
    '''
    Trains the supplied model agaist train and test locations specified
    in the args. During the training, each epoch will be evaluated for 
    val_loss and the model will be saved if val_loss is lower than
    previous.
    
    Args:
        classifier (keras.models.Model): the model to be trained.
                         trainloc (str): the location of the training data
                          testloc (str): the location of the test data
                        img_shape (int): the shape of the image to feed the CNN
                       output_dir (str): the directory where output files are saved
                       batch_size (int): the number of samples per gradient update
                       num_epochs (int): the number of epochs to train a model
    
    Returns:
        keras.models.Model: returns the trained CNN
    '''
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2, 
                                       rotation_range=25,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)


    training_set = train_datagen.flow_from_directory(trainloc,
                                                     target_size = (img_shape, img_shape),
                                                     batch_size = batch_size,
                                                     class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(testloc,
                                                target_size = (img_shape, img_shape),
                                                batch_size = batch_size,
                                                class_mode = 'categorical')

    # Saves the model weights after each epoch if the validation loss decreased
    now = datetime.now()
    nowstr = now.strftime('k2tf-%Y%m%d%H%M%S')

    now = os.path.join( output_dir, nowstr)

    # Make the directory
    os.makedirs( now, exist_ok=True )

    # Create our callbacks
    savepath = os.path.join( now, 'e-{epoch:03d}-vl-{val_loss:.3f}-va-{val_acc:.3f}.h5' )
    checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_acc', mode='max', verbose=0, save_best_only=True)
    fout = open( os.path.join(now, 'indices.txt'), 'wt' )
    fout.write( str(training_set.class_indices) + '\n' )

    # train the model on the new data for a few epochs
    classifier.fit_generator(training_set,
                             steps_per_epoch = len(training_set.filenames)//batch_size,
                             epochs = num_epochs,
                             validation_data = test_set,
                             validation_steps = len(test_set.filenames)//batch_size,
                             workers=32, 
                             max_q_size=32,
                             callbacks=[checkpointer]
                             )
    
    return classifier


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Required
    parser.add_argument('--test', dest='test', required=True, help='(REQUIRED) location of the test directory')
    parser.add_argument('--train', dest='train', required=True, help='(REQUIRED) location of the test directory')
    parser.add_argument('--cats', '-c', dest='categories', type=int, required=True, help='(REQUIRED) number of categories for the model to learn')
    # Optional
    parser.add_argument('--output', '-o', dest='output', default='./', required=False, help='location of the output directory (default:./)')
    parser.add_argument('--batch', '-b', dest='batch', default=32, type=int, required=False, help='batch size (default:32)')
    parser.add_argument('--epochs', '-e', dest='epochs', default=30, type=int, required=False, help='number of epochs to run (default:30)')
    parser.add_argument('--shape','-s', dest='shape', default=128, type=int, required=False, help='The shape of the image, single dimension will be applied to height and width (default:128)')
    
    args = parser.parse_args()
    
    classifier = buildClassifier( args.shape, args.categories)
    trainModel( classifier, args.train, args.test, args.shape, args.output, batch_size=args.batch, num_epochs=args.epochs )
    
