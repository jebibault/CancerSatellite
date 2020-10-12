# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# Load in the ImageIndex csv file
ImageIndex = pd.read_csv("City/City.tsv", sep='\t', header=0)

# Load in the ImageCensus csv file
ImageCensus = pd.read_csv("City/City_fips_index.csv", dtype={'census': str}, header=0)
ImageCensus['census'] = ImageCensus['census'].str[:-4]

# Join ImageIndex and ImageCensus
ImageNameCensus = ImageIndex.set_index('image').join(ImageCensus.set_index('image'))

# Load in the Image Features csv file
ImageFeatures = pd.read_csv("City/City_features.tsv", sep='\t', header=0)

# Split each vector into single features
ImageFeaturesSplit = pd.concat([ImageFeatures['id'], ImageFeatures['features'].str.split(',', expand=True)], axis=1)

# Join ImageNameCensus and ImageFeaturesSplit
ImageCensusFeatures = ImageNameCensus.set_index('id').join(ImageFeaturesSplit.set_index('id'))

# Load Census Cancer Prevalence
CancerPrevalence = pd.read_csv("City/CancerPrevalenceCity.csv", dtype={'census': str},header=0)

# Join ImageCensusFeatures and CancerPrevalence
CancerPrevalenceFeatures = CancerPrevalence.set_index('census').join(ImageCensusFeatures.set_index('census'))
CancerPrevalenceFeatures.to_csv('City/CensusCancerPrevalenceFeaturesCity.csv')
CensusCancerPrevalenceFeatures = pd.read_csv("City/CensusCancerPrevalenceFeaturesCity.csv", dtype={'census': str},header=0)
CensusCancerPrevalenceMeanFeatures = CensusCancerPrevalenceFeatures.groupby('census').mean()

# Save the file
CensusCancerPrevalenceMeanFeatures.to_csv('City/CensusCancerPrevalenceMeanFeaturesCity.csv')