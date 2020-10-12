# Code adapted from https://github.com/ml4a/ml4a-guides/blob/master/notebooks/image-search.ipynb

# Load libraries
import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.decomposition import PCA
import pickle

# Define environment
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model = keras.applications.ResNet50(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)

# Load images
def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

images_path = 'City/images/'
image_extensions = ['.jpg']
max_num_images = 10000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]

tic = time.clock()

# Extract features
features = []
for i, image_path in enumerate(images):
    if i % 500 == 0:
        toc = time.clock()
        elap = toc-tic;
        tic = time.clock()
    img, x = load_image(image_path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

# Keep best 300 features
features = np.array(features)
pca = PCA(n_components=300)
pca.fit(features)
pca_features = pca.transform(features)

pickle.dump([images, pca_features, pca], open('CityResNetFeaturest-SNE.p', 'wb'))