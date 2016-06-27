# CNN
Keras and Theano source code on image classifying.
The code trains a simple CNN model to classify images into 4 classes.
The dataset includes 2550 training images from 4 classes and 100 evaluation images, all of them are stored in the "img_data.pkl" using [Pickle Library]. The "cnn.py" code just loads images from this file, then processes training part and testing part.

# Installing frameworks and libraries (Ubuntu)
- Open your command line tool (terminal) and type these commands

## Installing Keras and Theano

### Installing [Theano] and its dependencies
- sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
- sudo pip install Theano

### Installing [Keras]
- sudo pip install keras

## Installing [h5py]
- sudo pip install h5py

# How to run
- Open your command line tool (terminal) and type the following command: 
- python cnn.py

[Theano]: https://theano.readthedocs.io/en/rel-0.6rc3/install_ubuntu.html
[Keras]: http://keras.io/
[h5py]: http://docs.h5py.org/en/latest/build.html
[Pickle Library]: https://wiki.python.org/moin/UsingPickle
