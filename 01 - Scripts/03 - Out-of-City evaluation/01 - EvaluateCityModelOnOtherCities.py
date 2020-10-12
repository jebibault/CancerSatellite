# Load libraries
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting ResNetChicagoModel evaluations.")

#Load Chicago's features
DatasetChicago = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesChicago.csv", dtype={'census': str},header=0)
DatasetChicago = DatasetChicago.dropna()
XChicago = DatasetChicago.drop(['cancerprevalence', 'census'], axis = 1)

#Calculate cosine similarity between Chicago and itself
CSChicagoChicago = cosine_similarity(XChicago, XChicago)
CSChicagoChicago_triu = CSChicagoChicago[np.triu_indices(XChicago.shape[0], k=1)]
CSChicagoChicago_triu_mean = np.mean(CSChicagoChicago_triu)
CSChicagoChicago_triu_std = np.std(CSChicagoChicago_triu)

# Calculate L2 Distance between Chicago and itself
L2DChicagoChicago = euclidean_distances(XChicago, XChicago)
L2DChicagoChicago_triu = L2DChicagoChicago[np.triu_indices(XChicago.shape[0], k=1)]
L2DChicagoChicago_triu_mean = np.mean(L2DChicagoChicago_triu)
L2DChicagoChicago_triu_std = np.std(L2DChicagoChicago_triu)

# Save Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "w")
SaveMetricsResults.write("City,R2,MSE,MAE,CSMean,CSStd,L2Mean,L2Std")
SaveMetricsResults.write("\n")
SaveMetricsResults.write("Chicago,1,1,1,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(CSChicagoChicago_triu_mean*1,2), np.round(CSChicagoChicago_triu_std*1,2), np.round(L2DChicagoChicago_triu_mean*1,2), np.round(L2DChicagoChicago_triu_std*1,2)))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for Chicago.")


# Load the model from disk
loaded_model = joblib.load("Models/Chicago_model.sav")


print("Starting Boston.")

# Load Boston Dataset
DatasetBoston = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesBoston.csv", dtype={'census': str},header=0)
DatasetBoston = DatasetBoston.dropna()

# Prepare features and labels with Boston datasets
XBoston = DatasetBoston.drop(['cancerprevalence', 'census'], axis = 1)
yBoston = DatasetBoston['cancerprevalence']
print("Data for Boston loaded.")

yBoston_pred = loaded_model.predict(XBoston)
R2Boston = r2_score(yBoston, yBoston_pred)
MSEBoston = mean_squared_error(yBoston, yBoston_pred)
MAEBoston = mean_absolute_error(yBoston, yBoston_pred)

#Calculate cosine similarity between Chicago and Boston
CSChicagoBoston = cosine_similarity(XChicago, XBoston)
CSChicagoBoston_mean = np.mean(CSChicagoBoston)
CSChicagoBoston_std = np.std(CSChicagoBoston)

# Calculate L2 distance between Chicago and Boston
L2DChicagoBoston = euclidean_distances(XChicago, XBoston)
L2DChicagoBoston_mean = np.mean(L2DChicagoBoston)
L2DChicagoBoston_std = np.std(L2DChicagoBoston)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("Boston,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2Boston*100,2), np.round(MSEBoston*1,2), np.round(MAEBoston*1,2), np.round(CSChicagoBoston_mean*1,2), np.round(CSChicagoBoston_std*1,2), np.round(L2DChicagoBoston_mean*1,2),np.round(L2DChicagoBoston_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for Boston.")

# Save predictions
CVPredictedCancerPrevalence = yBoston_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnBostonPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetBoston[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnBostonActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnBostonPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnBostonActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnBostonActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for Boston.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in Boston (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnBostonRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for Boston.")

print("Starting Dallas.")

# Load Dallas Dataset
DatasetDallas = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesDallas.csv", dtype={'census': str},header=0)
DatasetDallas = DatasetDallas.dropna()

# Prepare features and labels with Dallas datasets
XDallas = DatasetDallas.drop(['cancerprevalence', 'census'], axis = 1)
yDallas = DatasetDallas['cancerprevalence']
print("Data for Dallas loaded.")

yDallas_pred = loaded_model.predict(XDallas)
R2Dallas = r2_score(yDallas, yDallas_pred)
MSEDallas = mean_squared_error(yDallas, yDallas_pred)
MAEDallas = mean_absolute_error(yDallas, yDallas_pred)

#Calculate cosine similarity between Chicago and Dallas
CSChicagoDallas = cosine_similarity(XChicago, XDallas)
CSChicagoDallas_mean = np.mean(CSChicagoDallas)
CSChicagoDallas_std = np.std(CSChicagoDallas)

# Calculate L2 distance between Chicago and Dallas
L2DChicagoDallas = euclidean_distances(XChicago, XDallas)
L2DChicagoDallas_mean = np.mean(L2DChicagoDallas)
L2DChicagoDallas_std = np.std(L2DChicagoDallas)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("Dallas,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2Dallas*100,2), np.round(MSEDallas*1,2), np.round(MAEDallas*1,2), np.round(CSChicagoDallas_mean*1,2), np.round(CSChicagoDallas_std*1,2), np.round(L2DChicagoDallas_mean*1,2),np.round(L2DChicagoDallas_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for Dallas.")

# Save predictions
CVPredictedCancerPrevalence = yDallas_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnDallasPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetDallas[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnDallasActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnDallasPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnDallasActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnDallasActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for Dallas.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in Dallas (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnDallasRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for Dallas.")

print("Starting Houston.")

# Load Houston Dataset
DatasetHouston = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesHouston.csv", dtype={'census': str},header=0)
DatasetHouston = DatasetHouston.dropna()

# Prepare features and labels with Houston datasets
XHouston = DatasetHouston.drop(['cancerprevalence', 'census'], axis = 1)
yHouston = DatasetHouston['cancerprevalence']
print("Data for Houston loaded.")

yHouston_pred = loaded_model.predict(XHouston)
R2Houston = r2_score(yHouston, yHouston_pred)
MSEHouston = mean_squared_error(yHouston, yHouston_pred)
MAEHouston = mean_absolute_error(yHouston, yHouston_pred)

#Calculate cosine similarity between Chicago and Houston
CSChicagoHouston = cosine_similarity(XChicago, XHouston)
CSChicagoHouston_mean = np.mean(CSChicagoHouston)
CSChicagoHouston_std = np.std(CSChicagoHouston)

# Calculate L2 distance between Chicago and Houston
L2DChicagoHouston = euclidean_distances(XChicago, XHouston)
L2DChicagoHouston_mean = np.mean(L2DChicagoHouston)
L2DChicagoHouston_std = np.std(L2DChicagoHouston)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("Houston,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2Houston*100,2), np.round(MSEHouston*1,2), np.round(MAEHouston*1,2), np.round(CSChicagoHouston_mean*1,2), np.round(CSChicagoHouston_std*1,2), np.round(L2DChicagoHouston_mean*1,2),np.round(L2DChicagoHouston_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()

print("Metrics saved for Houston.")

# Save predictions
CVPredictedCancerPrevalence = yHouston_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnHoustonPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetHouston[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnHoustonActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnHoustonPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnHoustonActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnHoustonActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for Houston.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in Houston (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnHoustonRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for Houston.")

print("Starting LosAngeles.")

# Load LosAngeles Dataset
DatasetLosAngeles = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesLosAngeles.csv", dtype={'census': str},header=0)
DatasetLosAngeles = DatasetLosAngeles.dropna()

# Prepare features and labels with LosAngeles datasets
XLosAngeles = DatasetLosAngeles.drop(['cancerprevalence', 'census'], axis = 1)
yLosAngeles = DatasetLosAngeles['cancerprevalence']
print("Data for LosAngeles loaded.")

yLosAngeles_pred = loaded_model.predict(XLosAngeles)
R2LosAngeles = r2_score(yLosAngeles, yLosAngeles_pred)
MSELosAngeles = mean_squared_error(yLosAngeles, yLosAngeles_pred)
MAELosAngeles = mean_absolute_error(yLosAngeles, yLosAngeles_pred)

#Calculate cosine similarity between Chicago and LosAngeles
CSChicagoLosAngeles = cosine_similarity(XChicago, XLosAngeles)
CSChicagoLosAngeles_mean = np.mean(CSChicagoLosAngeles)
CSChicagoLosAngeles_std = np.std(CSChicagoLosAngeles)

# Calculate L2 distance between Chicago and LosAngeles
L2DChicagoLosAngeles = euclidean_distances(XChicago, XLosAngeles)
L2DChicagoLosAngeles_mean = np.mean(L2DChicagoLosAngeles)
L2DChicagoLosAngeles_std = np.std(L2DChicagoLosAngeles)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("LosAngeles,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2LosAngeles*100,2), np.round(MSELosAngeles*1,2), np.round(MAELosAngeles*1,2), np.round(CSChicagoLosAngeles_mean*1,2), np.round(CSChicagoLosAngeles_std*1,2), np.round(L2DChicagoLosAngeles_mean*1,2),np.round(L2DChicagoLosAngeles_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for LosAngeles.")

# Save predictions
CVPredictedCancerPrevalence = yLosAngeles_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnLosAngelesPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetLosAngeles[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnLosAngelesActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnLosAngelesPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnLosAngelesActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnLosAngelesActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for Los Angeles.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in LosAngeles (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnLosAngelesRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for Los Angeles.")

print("Starting NewYork.")

# Load NewYork Dataset
DatasetNewYork = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesNewYork.csv", dtype={'census': str},header=0)
DatasetNewYork = DatasetNewYork.dropna()

# Prepare features and labels with NewYork datasets
XNewYork = DatasetNewYork.drop(['cancerprevalence', 'census'], axis = 1)
yNewYork = DatasetNewYork['cancerprevalence']
print("Data for NewYork loaded.")

yNewYork_pred = loaded_model.predict(XNewYork)
R2NewYork = r2_score(yNewYork, yNewYork_pred)
MSENewYork = mean_squared_error(yNewYork, yNewYork_pred)
MAENewYork = mean_absolute_error(yNewYork, yNewYork_pred)

#Calculate cosine similarity between Chicago and NewYork
CSChicagoNewYork = cosine_similarity(XChicago, XNewYork)
CSChicagoNewYork_mean = np.mean(CSChicagoNewYork)
CSChicagoNewYork_std = np.std(CSChicagoNewYork)

# Calculate L2 distance between Chicago and NewYork
L2DChicagoNewYork = euclidean_distances(XChicago, XNewYork)
L2DChicagoNewYork_mean = np.mean(L2DChicagoNewYork)
L2DChicagoNewYork_std = np.std(L2DChicagoNewYork)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("NewYork,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2NewYork*100,2), np.round(MSENewYork*1,2), np.round(MAENewYork*1,2), np.round(CSChicagoNewYork_mean*1,2), np.round(CSChicagoNewYork_std*1,2), np.round(L2DChicagoNewYork_mean*1,2),np.round(L2DChicagoNewYork_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for NewYork.")

# Save predictions
CVPredictedCancerPrevalence = yNewYork_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnNewYorkPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetNewYork[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnNewYorkActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnNewYorkPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnNewYorkActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnNewYorkActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for New York.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in NewYork (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnNewYorkRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for New York.")

print("Starting Philadelphia.")

# Load Philadelphia Dataset
DatasetPhiladelphia = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesPhiladelphia.csv", dtype={'census': str},header=0)
DatasetPhiladelphia = DatasetPhiladelphia.dropna()

# Prepare features and labels with Philadelphia datasets
XPhiladelphia = DatasetPhiladelphia.drop(['cancerprevalence', 'census'], axis = 1)
yPhiladelphia = DatasetPhiladelphia['cancerprevalence']
print("Data for Philadelphia loaded.")

yPhiladelphia_pred = loaded_model.predict(XPhiladelphia)
R2Philadelphia = r2_score(yPhiladelphia, yPhiladelphia_pred)
MSEPhiladelphia = mean_squared_error(yPhiladelphia, yPhiladelphia_pred)
MAEPhiladelphia = mean_absolute_error(yPhiladelphia, yPhiladelphia_pred)

#Calculate cosine similarity between Chicago and Philadelphia
CSChicagoPhiladelphia = cosine_similarity(XChicago, XPhiladelphia)
CSChicagoPhiladelphia_mean = np.mean(CSChicagoPhiladelphia)
CSChicagoPhiladelphia_std = np.std(CSChicagoPhiladelphia)

# Calculate L2 distance between Chicago and Philadelphia
L2DChicagoPhiladelphia = euclidean_distances(XChicago, XPhiladelphia)
L2DChicagoPhiladelphia_mean = np.mean(L2DChicagoPhiladelphia)
L2DChicagoPhiladelphia_std = np.std(L2DChicagoPhiladelphia)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("Philadelphia,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2Philadelphia*100,2), np.round(MSEPhiladelphia*1,2), np.round(MAEPhiladelphia*1,2), np.round(CSChicagoPhiladelphia_mean*1,2), np.round(CSChicagoPhiladelphia_std*1,2), np.round(L2DChicagoPhiladelphia_mean*1,2),np.round(L2DChicagoPhiladelphia_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for Philadelphia.")

# Save predictions
CVPredictedCancerPrevalence = yPhiladelphia_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnPhiladelphiaPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetPhiladelphia[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnPhiladelphiaActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnPhiladelphiaPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnPhiladelphiaActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnPhiladelphiaActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for Philadelphia.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in Philadelphia (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnPhiladelphiaRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for Philadelphia.")

print("Starting Phoenix.")

# Load Phoenix Dataset
DatasetPhoenix = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesPhoenix.csv", dtype={'census': str},header=0)
DatasetPhoenix = DatasetPhoenix.dropna()

# Prepare features and labels with Phoenix datasets
XPhoenix = DatasetPhoenix.drop(['cancerprevalence', 'census'], axis = 1)
yPhoenix = DatasetPhoenix['cancerprevalence']
print("Data for Phoenix loaded.")


yPhoenix_pred = loaded_model.predict(XPhoenix)
R2Phoenix = r2_score(yPhoenix, yPhoenix_pred)
MSEPhoenix = mean_squared_error(yPhoenix, yPhoenix_pred)
MAEPhoenix = mean_absolute_error(yPhoenix, yPhoenix_pred)

#Calculate cosine similarity between Chicago and Phoenix
CSChicagoPhoenix = cosine_similarity(XChicago, XPhoenix)
CSChicagoPhoenix_mean = np.mean(CSChicagoPhoenix)
CSChicagoPhoenix_std = np.std(CSChicagoPhoenix)

# Calculate L2 distance between Chicago and Phoenix
L2DChicagoPhoenix = euclidean_distances(XChicago, XPhoenix)
L2DChicagoPhoenix_mean = np.mean(L2DChicagoPhoenix)
L2DChicagoPhoenix_std = np.std(L2DChicagoPhoenix)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("Phoenix,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2Phoenix*100,2), np.round(MSEPhoenix*1,2), np.round(MAEPhoenix*1,2), np.round(CSChicagoPhoenix_mean*1,2), np.round(CSChicagoPhoenix_std*1,2), np.round(L2DChicagoPhoenix_mean*1,2),np.round(L2DChicagoPhoenix_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for Phoenix.")

# Save predictions
CVPredictedCancerPrevalence = yPhoenix_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnPhoenixPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetPhoenix[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnPhoenixActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnPhoenixPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnPhoenixActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnPhoenixActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for Phoenix.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in Phoenix (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnPhoenixRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for Phoenix.")

print("Starting SanAntonio.")

# Load SanAntonio Dataset
DatasetSanAntonio = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesSanAntonio.csv", dtype={'census': str},header=0)
DatasetSanAntonio = DatasetSanAntonio.dropna()

# Prepare features and labels with SanAntonio datasets
XSanAntonio = DatasetSanAntonio.drop(['cancerprevalence', 'census'], axis = 1)
ySanAntonio = DatasetSanAntonio['cancerprevalence']
print("Data for SanAntonio loaded.")

ySanAntonio_pred = loaded_model.predict(XSanAntonio)
R2SanAntonio = r2_score(ySanAntonio, ySanAntonio_pred)
MSESanAntonio = mean_squared_error(ySanAntonio, ySanAntonio_pred)
MAESanAntonio = mean_absolute_error(ySanAntonio, ySanAntonio_pred)

#Calculate cosine similarity between Chicago and SanAntonio
CSChicagoSanAntonio = cosine_similarity(XChicago, XSanAntonio)
CSChicagoSanAntonio_mean = np.mean(CSChicagoSanAntonio)
CSChicagoSanAntonio_std = np.std(CSChicagoSanAntonio)

# Calculate L2 distance between Chicago and SanAntonio
L2DChicagoSanAntonio = euclidean_distances(XChicago, XSanAntonio)
L2DChicagoSanAntonio_mean = np.mean(L2DChicagoSanAntonio)
L2DChicagoSanAntonio_std = np.std(L2DChicagoSanAntonio)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("SanAntonio,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2SanAntonio*100,2), np.round(MSESanAntonio*1,2), np.round(MAESanAntonio*1,2), np.round(CSChicagoSanAntonio_mean*1,2), np.round(CSChicagoSanAntonio_std*1,2), np.round(L2DChicagoSanAntonio_mean*1,2),np.round(L2DChicagoSanAntonio_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for SanAntonio.")

# Save predictions
CVPredictedCancerPrevalence = ySanAntonio_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanAntonioPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetSanAntonio[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanAntonioActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnSanAntonioPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnSanAntonioActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanAntonioActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for San Antonio.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in SanAntonio (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnSanAntonioRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for San Antonio.")

print("Starting SanDiego.")

# Load SanDiego Dataset
DatasetSanDiego = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesSanDiego.csv", dtype={'census': str},header=0)
DatasetSanDiego = DatasetSanDiego.dropna()

# Prepare features and labels with SanDiego datasets
XSanDiego = DatasetSanDiego.drop(['cancerprevalence', 'census'], axis = 1)
ySanDiego = DatasetSanDiego['cancerprevalence']
print("Data for SanDiego loaded.")

ySanDiego_pred = loaded_model.predict(XSanDiego)
R2SanDiego = r2_score(ySanDiego, ySanDiego_pred)
MSESanDiego = mean_absolute_error(ySanDiego, ySanDiego_pred)
MAESanDiego = mean_absolute_error(ySanDiego, ySanDiego_pred)

#Calculate cosine similarity between Chicago and SanDiego
CSChicagoSanDiego = cosine_similarity(XChicago, XSanDiego)
CSChicagoSanDiego_mean = np.mean(CSChicagoSanDiego)
CSChicagoSanDiego_std = np.std(CSChicagoSanDiego)

# Calculate L2 distance between Chicago and SanDiego
L2DChicagoSanDiego = euclidean_distances(XChicago, XSanDiego)
L2DChicagoSanDiego_mean = np.mean(L2DChicagoSanDiego)
L2DChicagoSanDiego_std = np.std(L2DChicagoSanDiego)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("SanDiego,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2SanDiego*100,2), np.round(MSESanDiego*1,2), np.round(MAESanDiego*1,2), np.round(CSChicagoSanDiego_mean*1,2), np.round(CSChicagoSanDiego_std*1,2), np.round(L2DChicagoSanDiego_mean*1,2),np.round(L2DChicagoSanDiego_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for SanDiego.")

# Save predictions
CVPredictedCancerPrevalence = ySanDiego_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanDiegoPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetSanDiego[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanDiegoActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnSanDiegoPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnSanDiegoActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanDiegoActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for San Diego.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in SanDiego (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnSanDiegoRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for San Diego.")

print("Starting SanFrancisco.")

# Load SanFrancisco Dataset
DatasetSanFrancisco = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesSanFrancisco.csv", dtype={'census': str},header=0)
DatasetSanFrancisco = DatasetSanFrancisco.dropna()

# Prepare features and labels with SanFrancisco datasets
XSanFrancisco = DatasetSanFrancisco.drop(['cancerprevalence', 'census'], axis = 1)
ySanFrancisco = DatasetSanFrancisco['cancerprevalence']
print("Data for SanFrancisco loaded.")

ySanFrancisco_pred = loaded_model.predict(XSanFrancisco)
R2SanFrancisco = r2_score(ySanFrancisco, ySanFrancisco_pred)
MSESanFrancisco = mean_squared_error(ySanFrancisco, ySanFrancisco_pred)
MAESanFrancisco = mean_absolute_error(ySanFrancisco, ySanFrancisco_pred)

#Calculate cosine similarity between Chicago and SanFrancisco
CSChicagoSanFrancisco = cosine_similarity(XChicago, XSanFrancisco)
CSChicagoSanFrancisco_mean = np.mean(CSChicagoSanFrancisco)
CSChicagoSanFrancisco_std = np.std(CSChicagoSanFrancisco)

# Calculate L2 distance between Chicago and SanFrancisco
L2DChicagoSanFrancisco = euclidean_distances(XChicago, XSanFrancisco)
L2DChicagoSanFrancisco_mean = np.mean(L2DChicagoSanFrancisco)
L2DChicagoSanFrancisco_std = np.std(L2DChicagoSanFrancisco)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("SanFrancisco,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2SanFrancisco*100,2), np.round(MSESanFrancisco*1,2), np.round(MAESanFrancisco*1,2), np.round(CSChicagoSanFrancisco_mean*1,2), np.round(CSChicagoSanFrancisco_std*1,2), np.round(L2DChicagoSanFrancisco_mean*1,2),np.round(L2DChicagoSanFrancisco_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for SanFrancisco.")

# Save predictions
CVPredictedCancerPrevalence = ySanFrancisco_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanFranciscoPredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetSanFrancisco[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanFranciscoActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnSanFranciscoPredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnSanFranciscoActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanFranciscoActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for SanFrancisco.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in SanFrancisco (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnSanFranciscoRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for SanFrancisco.")

print("Starting SanJose.")

# Load SanJose Dataset
DatasetSanJose = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesSanJose.csv", dtype={'census': str},header=0)
DatasetSanJose = DatasetSanJose.dropna()

# Prepare features and labels with SanJose datasets
XSanJose = DatasetSanJose.drop(['cancerprevalence', 'census'], axis = 1)
ySanJose = DatasetSanJose['cancerprevalence']
print("Data for SanJose loaded.")

ySanJose_pred = loaded_model.predict(XSanJose)
R2SanJose = r2_score(ySanJose, ySanJose_pred)
MSESanJose = mean_squared_error(ySanJose, ySanJose_pred)
MAESanJose = mean_absolute_error(ySanJose, ySanJose_pred)

#Calculate cosine similarity between Chicago and SanJose
CSChicagoSanJose = cosine_similarity(XChicago, XSanJose)
CSChicagoSanJose_mean = np.mean(CSChicagoSanJose)
CSChicagoSanJose_std = np.std(CSChicagoSanJose)

# Calculate L2 distance between Chicago and SanJose
L2DChicagoSanJose = euclidean_distances(XChicago, XSanJose)
L2DChicagoSanJose_mean = np.mean(L2DChicagoSanJose)
L2DChicagoSanJose_std = np.std(L2DChicagoSanJose)

# Save R2, Mean Square Error, Cosine Similarity and L2 Distance
SaveMetricsResults = open("Metrics/ChicagoModelMetricsOnOtherCities.csv", "a")
SaveMetricsResults.write("SanJose,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (np.round(R2SanJose*100,2), np.round(MSESanJose*1,2), np.round(MAESanJose*1,2), np.round(CSChicagoSanJose_mean*1,2), np.round(CSChicagoSanJose_std*1,2), np.round(L2DChicagoSanJose_mean*1,2),np.round(L2DChicagoSanJose_std*1,2),))
SaveMetricsResults.write("\n")
SaveMetricsResults.close()
print("Metrics saved for SanJose.")

# Save predictions
CVPredictedCancerPrevalence = ySanJose_pred
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanJosePredictedCancerPrevalence.csv", header=['id','predictedprevalence'],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = DatasetSanJose[['census','cancerprevalence']]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanJoseActualCancerPrevalenceWithIndex.csv", header=['id','census','actualprevalence'],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnSanJosePredictedCancerPrevalence.csv", dtype={'census': str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("Prevalence/ChicagoModelOnSanJoseActualCancerPrevalenceWithIndex.csv", dtype={'census': str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index('id').join(PredictedCancerPrevalence.set_index('id'))
CensusActualPredictedCancerPrevalence.to_csv("Prevalence/ChicagoModelOnSanJoseActualPredictedCancerPrevalenceWithIndex.csv", header=['census','actualprevalence','predictedprevalence'],index=False)
print("Actual and predicted cancer prevalences saved as a csv for San Jose.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x='actualprevalence', y='predictedprevalence', data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel='Actual cancer prevalence', ylabel='Predicted cancer prevalence in SanJose (ResNetChicago)')
FigureCancerPrevalence.fig.set_size_inches(10,10)
FigureCancerPrevalence.savefig("Figures/ChicagoModelOnSanJoseRegressionPlot.png", format='png', dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png for San Jose.")

print("ResNetChicagoModel finished.")
print()
print()
print()