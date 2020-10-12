# Load libraries
import argparse
import csv
import os

from keras import applications
from keras.applications.ResNet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import pandas

width = 224
height = 224
input_shape = (height, width, 3)

# Define environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"


def named_model(name):

    return applications.resnet50.ResNet50(
    	weights='imagenet',
    	include_top=False,  #removes the fully connected layer at the top of the network
    	pooling='max')

parser.add_argument(
	'source',
	default=None)

parser.add_argument(
    'model',
    default='ResNet50',
    nargs="?",
    type=named_model,
)

pargs = parser.parse_args()

source_dir = os.path.dirname(pargs.source)

def get_feature(metadata):
    print('{}'.format(metadata['id']))
    try:
        img_path = os.path.join(source_dir, 'images', metadata['image'])
        if os.path.isfile(img_path):
            print('is file: {}'.format(img_path))
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = pargs.model.predict(x)[0]
                features_arr = np.char.mod('%f', features)

                return {"id": metadata['id'], "features": ','.join(features_arr)}
            except Exception as ex:
                print(ex)
                pass
    except Exception as ex:
        print(ex)
        pass
    return None


def start():
    try:
        data = pandas.read_csv(pargs.source, sep='\t')
        features = map(get_feature, data.T.to_dict().values())
        features = filter(None, features)
        source_filename = os.path.splitext(pargs.source)[0].split(os.sep)[-1]

        with open(os.path.join(source_dir, '{}_features.tsv'.format(source_filename)), 'w') as output:
            w = csv.DictWriter(output, fieldnames=['id', 'features'], delimiter='\t', lineterminator='\n')
            w.writeheader()
            w.writerows(features)

    except EnvironmentError as e:
        print(e)

if __name__ == '__main__':
    start()
