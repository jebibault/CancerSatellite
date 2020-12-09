# Load required libraries
import os
from time import time
import numpy as np
import pandas as pd
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D
from keras import optimizers
from keras import applications
from keras import backend as K
from keras.models import model_from_json
from keras import ResNet50 as ResNet
from keras import preprocess_input
from contextlib import redirect_stdout
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Define environment
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"


# Path to the model weights files
weights_path = 'ResNet50.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'


# Open train and test data csv
nb_train_samples = 9248
nb_test_samples = 2304
epochs = 50
batch_size = 32
train = pd.read_csv("CityTrain.csv", dtype=str)
train = train.dropna()
train.image = train.image.astype(str)

test = pd.read_csv("CityTest.csv", dtype=str)
test = test.dropna()
test.image = test.image.astype(str)
print('Train and test dataframes created.')

# Build ResNet50 network
base_model = RestNet(include_top=False, weights='imagenet', input_shape=(600, 600, 3))
print('Model loaded.')

# Define call-backs
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
csv_logger = CSVLogger('Citylog.csv', append=True, separator=',')
reduceLRonplateau = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
earlystopping = EarlyStopping(monitor='mean_squared_error', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
history = History()

# Build regression model to put on top of the convolutional model
top_model = Sequential()
top_model.add(GlobalMaxPooling2D(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='linear'))
top_model.load_weights(top_model_weights_path)

print('Regression model loaded.')

# Add regression model on top in order to successfully do fine-tuning
model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

# Set last 7 layers to trainable
model.trainable = True

set_trainable = False
for layer in model.layers:
    if layer.name == 'top_conv':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Compile the model with an Adam optimizer and a very slow learning rate.
model.compile(loss='mean_squared_error',
    optimizer=optimizers.Adam(lr=0.256, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-2/epochs, amsgrad=False),
    metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])
print('Complete model compiled.')

# Perform data augmentation on train data
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rescale=1. / 255
    )

# Don't perform data augmentation on test data, simply rescale
test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rescale=1. / 255
    )

# Load train data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory='images', 
    x_col='image', 
    y_col='actualprevalence', 
    target_size=(600,600), 
    color_mode='rgb', 
    classes=None, 
    class_mode='raw', 
    batch_size=32, 
    shuffle=True, 
    subset=None, 
    interpolation='nearest', 
    drop_duplicates=False)

# Load test data
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory='images', 
    x_col='image', 
    y_col='actualprevalence', 
    target_size=(600,600), 
    color_mode='rgb', 
    classes=None, 
    class_mode='raw', 
    batch_size=32, 
    shuffle=False,
    subset=None, 
    interpolation='nearest', 
    drop_duplicates=False)

with open('ModelSummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Fine-tune complete model
history = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=test_generator,
    nb_val_samples=nb_test_samples,
    callbacks=[tensorboard, csv_logger, reduceLRonplateau, earlystopping, history])
print('Complete model fine-tuned.')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(weights_path)
print('Fine-tuned model and weights saved.')

# Summarize history for mean squared error
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Model mean squared error')
plt.ylabel('Mean squared error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('../Results/Figures/NonLinearRegression/MeanSquaredErrorCity.eps', format='eps', dpi=1000)

# Summarize history for mean absolute error
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model mean absolute error')
plt.ylabel('Mean absolute error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('../Results/Figures/NonLinearRegression/MeanAbsoluteErrorCity.eps', format='eps', dpi=1000)

# Summarize history for mean squared error
plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('Model mean percentage error')
plt.ylabel('Mean percentage error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('../Results/Figures/NonLinearRegression/MeanPercentageErrorCity.eps', format='eps', dpi=1000)
print("Non linear regression plotted and saved as eps.")

# Save predictions
PredictedPrevalence = model.predict_generator(
	test_generator,
	nb_test_samples // batch_size)
PredictedPrevalence = np.round(PredictedPrevalence,1)
PredictedPrevalence = pd.DataFrame(data=PredictedPrevalence.flatten(), columns = ["predictedprevalence"])
PredictedPrevalence.reset_index()
PredictedPrevalence.to_csv("../Results/PredictedPrevalenceFineTunedResNet50City.csv", header=['id','predictedprevalence'],index=False)