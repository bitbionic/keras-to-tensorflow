g++ -c -pipe -g -std=gnu++11 -Wall -W -fPIC -I. -I./tensorflow -I./tensorflow/bazel-tensorflow/external/eigen_archive -I./tensorflow/bazel-tensorflow/external/protobuf/src -I./tensorflow/bazel-genfiles -o main.o ./main.cpp
g++  -o Tutorial main.o   -L./tensorflow/bazel-bin/tensorflow -ltensorflow_cc
cp ./tensorflow/bazel-bin/tensorflow/libtensorflow* .
