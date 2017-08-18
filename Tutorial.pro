TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += $$PWD/tensorflow
INCLUDEPATH += $$PWD/tensorflow/bazel-tensorflow/external/eigen_archive
INCLUDEPATH += $$PWD/tensorflow/bazel-tensorflow/external/protobuf/src
INCLUDEPATH += $$PWD/tensorflow/bazel-genfiles

LIBS += -L$$PWD/tensorflow/bazel-bin/tensorflow -ltensorflow_cc

SOURCES += main.cpp
