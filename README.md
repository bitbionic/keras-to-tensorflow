# Keras to Tensorflow Tutorial
So you’ve built an awesome machine learning model in Keras and now you want to run it natively thru Tensorflow. This tutorial will show you how.

[Keras](http://keras.io/) is a wonderful high level framework for building machine learning models. It is able to utilize multiple backends such as [Tensorflow](http://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/) to do so. When a keras model is saved via the [.save method](http://keras.io/getting-started/faq/#how-can-i-save-a-keras-model), the canonical save method serializes to an HDF5 format. Tensorflow works with [Protocol Buffers](http://developers.google.com/protocol-buffers/), and therefore loads and saves .pb files. This tutorial demonstrates how to:
  * build a SIMPLE Convolutional Neural Network in Keras for image classification
  * save the Keras model as an HDF5 model
  * verify the Keras model
  * convert the HDF5 model to a Protocol Buffer
  * build a Tensorflow C++ shared library
  * utilize the .pb in a pure Tensorflow app
    * We will utilize Tensorflow's own example code for this

## Assumptions ##
  * You are familiar with Python *(and C++ if you're interested in the C++ portion of this tutorial)*
  * You are familiar with Keras and Tensorflow and already have your dev environment setup
  * Example code is utilizing Python 3.5, if you are using 2.7 you may have to make modifications

The full tutorial can be read here: [http://www.bitbionic.com/2017/08/18/run-your-keras-models-in-c-tensorflow/](http://www.bitbionic.com/2017/08/18/run-your-keras-models-in-c-tensorflow/)
