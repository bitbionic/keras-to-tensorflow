'''
This script evaluates a simple Convolutional Nerual Network (CNN)
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
import argparse

import numpy as np

from keras.preprocessing import image
from keras.models import load_model

def invertKVPairs( someDictionary ):
    '''
    Inverts the key/value pairs of the supplied dictionary.
    
    Args:
        someDictionary (dict): The dictionary for which you would like the inversion
    Returns:
        Dictionary - the inverse key-value pairing of someDictionary
    '''
    ret = {}
    for k, v in someDictionary.items():
        ret[v] = k

    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m', dest='model', required=True, help='The HDF5 Keras model you wish to run')
    parser.add_argument('--image','-i', dest='image', required=True, help='The image you wish to test')
    parser.add_argument('--shape','-s', type=int, dest='shape', required=True, help='The shape to resize the image for the model')
    parser.add_argument('--labels','-l', dest='labels', required=False, help='The indices.txt file containing the class_indices of the Keras training set')
    args = parser.parse_args()

    model = load_model(args.model)
    
    # These indices are saved on the output of our trainer
    class_indices = { 0:'daisy',  1:'dandelion',  2:'roses', 3:'sunflowers', 4:'tulips' }
    if args.labels:
        with open( args.labels, 'rt' ) as infile:
            label_str = infile.read()
            str2dict = eval(label_str)
            class_indices = invertKVPairs( str2dict )

    test_image = image.load_img(args.image, target_size=(args.shape,args.shape))
    test_image = np.expand_dims( test_image, axis=0 )
    test_image = test_image/255.0
    result = model.predict(test_image)

    for idx,val in enumerate( result[0] ):
        print( '%s : %4.2f percent' % (class_indices[idx], val*100. ) )

