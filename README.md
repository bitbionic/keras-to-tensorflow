# Keras to Tensorflow Tutorial
[Keras](http://keras.io/) is a wonderful high level framework for building machine learning models. It is able to utilize multiple backends such as [Tensorflow](http://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/) to do so. When a keras model is saved via the [.save method](http://keras.io/getting-started/faq/#how-can-i-save-a-keras-model), the canonical save method serializes to an HDF5 format. Tensorflow works with [Protocol Buffers](http://developers.google.com/protocol-buffers/), and therefore loads and saves .pb files. This tutorial demonstrates how to:
  * build a Convolutional Neural Network in Keras
  * save the Keras model as an HDF5 model
  * verify the Keras model
  * convert the HDF5 model to a Protocol Buffer
  * utilize the .pb in a pure Tensorflow app
    * We will utilize Tensorflow's own example code for this

## Assumptions ##
  * You are familiar with Python *(and C++ if you're interested in the C++ portion of this tutorial)*
  * You are familiar with Keras and Tensorflow and already have your dev environment setup
  * Example code is utilizing Python 3.5, if you are using 2.7 you may have to make modifications
