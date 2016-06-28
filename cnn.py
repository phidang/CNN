from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import pickle
import numpy as np
import json

############################ GLOBAL VARIABLES ################################
DATA_FILE = 'img_data.pkl'
SAVED_WEIGHTS = "Class_4.h5"
NUM_CLASS = 4
NUM_TRAIN_IMG = 2550
NUM_TEST_IMG = 100
INPUT_CHANNEL = 3
INPUT_WIDTH = 90
INPUT_HEIGHT = 60

############################## LOAD DATA #####################################
f = open(DATA_FILE, 'rb')

training_data = pickle.load(f)
training_inputs = np.reshape(training_data[0], (NUM_TRAIN_IMG, INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))
training_results = np.reshape(training_data[1], (NUM_TRAIN_IMG, NUM_CLASS))

eval_data = pickle.load(f)
X_test = np.reshape(eval_data[0], (NUM_TEST_IMG, INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))
y_test = np.reshape(eval_data[1], (NUM_TEST_IMG, NUM_CLASS))

f.close()
###############################################################################

############################ DESIGN MODEL #####################################
model = Sequential()

# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=(INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT)))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASS))
model.add(Activation('softmax'))
#################################################################################

############################ TRAINING ###########################################
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# starts training
model.fit(training_inputs, training_results, nb_epoch=5, batch_size=32)  
#################################################################################

############################ SAVE MODEL #########################################
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

model.save_weights(SAVED_WEIGHTS) # save weights
#################################################################################

############################ EVALUATION #########################################
score = model.evaluate(X_test, y_test, batch_size=16)

print score
#################################################################################



