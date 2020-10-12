# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import statsmodels.stats.api as sms

# Load the shapefile
fp = "ShapeFiles/City/City.shp"
map_df = gpd.read_file(fp)

# Check data
map_df.head()

# Preview map
map_df.plot()

# Load in cancer prevalence csv
df = pd.read_csv("CityModelActualPredictedCancerPrevalenceWithIndex.csv", dtype={'census': str},header=0)

#Reformat census ID if necessary to match geodataframe
#df['census'] = df['census'].str[5:] 

df.head()

# Join the geodataframe with the csv
merged=map_df.set_index('geoid10').join(df.set_index('census'))
merged=merged.drop(['notes'], axis=1)
merged = merged.dropna(axis = 0, how ='any')
merged

# Check minimum cancer prevalence for range
df['actualprevalence'].min()

# Check maximum cancer prevalence for range
df['actualprevalence'].max()

ap = df['actualprevalence']
ic95 = sms.DescrStatsW(ap).tconfint_mean()
topic95 = ic95[1]

# Set the range for the choropleth
vmin = 0
vmax = 12

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(
    1,
    figsize=(50, 30))

# Create actual prevalence map
merged.plot(
    column='actualprevalence', 
    cmap='coolwarm', 
    linewidth=0.9,
    edgecolor='0.8',
    norm=plt.Normalize(vmin=vmin, vmax=vmax),
    ax=ax)

# Remove the axis
ax.axis('off')

# Add a title
ax.set_title(
    'City - Actual cancer prevalence',
    fontdict={'fontsize': '60', 'fontweight' : '10'},
    loc='left')

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(
    cmap='coolwarm',
    norm=plt.Normalize(
        vmin=vmin,
        vmax=vmax))

cbar = fig.colorbar(
    sm, 
    fraction=0.046,
    pad=0.04,
    orientation='horizontal')

cbar.ax.tick_params(
    labelsize=40,
    width=1,
    length=10)

# Save figure as png and svg
fig.savefig('CityActualPrevalence.svg')
fig.savefig('CityActualPrevalence.png', dpi=300)
plt.close(fig)

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(
    1,
    figsize=(50, 30))

# Create predicted prevalence map
merged.plot(
    column='predictedprevalence', 
    cmap='coolwarm', 
    linewidth=0.9, 
    edgecolor='0.8',
    norm=plt.Normalize(vmin=vmin, vmax=vmax),
    ax=ax)

# Remove the axis
ax.axis('off')

# Add a title
ax.set_title(
    'City - Predicted cancer prevalence',
    fontdict={'fontsize': '60', 'fontweight' : '10'},
    loc='left')

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(
    cmap='coolwarm',
    norm=plt.Normalize(
        vmin=vmin,
        vmax=vmax))

cbar = fig.colorbar(
    sm, 
    fraction=0.046,
    pad=0.04,
    orientation='horizontal')

cbar.ax.tick_params(
    labelsize=40,
    width=1,
    length=10)

# Save figure as png and svg
fig.savefig('CityPredictedPrevalence.svg')
fig.savefig('CityPredictedPrevalence.png', dpi=300)