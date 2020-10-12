# Load libraries
import numpy as np
import math
import pandas as pd
import joblib
import sklearn
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, BinnedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_regression


# Load Dataset
Dataset = pd.read_csv("Features/CensusCancerPrevalenceMeanFeaturesCity.csv", dtype={"census": str},header=0)
Dataset = Dataset.dropna()

# Load, split and scale train and test datasets
X = Dataset.drop(["cancerprevalence", "census"], axis = 1)
y = Dataset["cancerprevalence"]
print("Data loaded.")

# Build the model
param_grid = [{
"kbest__k": range(1, 200),
"ElasticNet__l1_ratio": [.05, .15, .95, .99, 1],
"ElasticNet__alpha": np.logspace(-5, 0.5, 60),
}]

print("ElasticNet GridSearchCV starts.")

enet = ElasticNet()
cv = BinnedStratifiedKFold(
	n_splits=5,
	shuffle=True,
	random_state=42)
pipe = Pipeline(
	steps=[
	("scale", StandardScaler()),
	("kbest", SelectKBest(f_regression, 154)),
	("ElasticNet", enet)
	])
grid = GridSearchCV(
	estimator=pipe, 
	param_grid = param_grid,
	cv=cv,
	n_jobs=-1,
	verbose=50)
grid.fit(X,y)

OptimisedElasticNet = grid.best_estimator_

# Save the best parameters
print("Here are the best parameters:")
print (grid.best_params_)
SaveBestParameters = open("Parameters/CityModelParameters.txt", "w")
SaveBestParameters.write("Best parameters are:")
SaveBestParameters.write("\n")
SaveBestParameters.write(str(grid.best_params_))
SaveBestParameters.close()

# Save the model
filename = "Models/ResNetCity_model.sav"
joblib.dump(OptimisedElasticNet, filename)

# Calculate R Squared, Mean Squared Error and Mean Absolute Error across the 5-fold cross-validation
scores = cross_validate(OptimisedElasticNet, X, y, scoring=("r2", "neg_mean_squared_error", "neg_mean_absolute_error"), cv=cv, verbose=0)
R25CV = scores["test_r2"]
MSE5CV = scores["test_neg_mean_squared_error"]
MAE5CV = scores["test_neg_mean_absolute_error"]

# Save R Squared, Mean Squared Error and Mean Absolute Error across the 5-fold cross-validation
SaveMetricsResults = open("Metrics/CityModelMetrics.txt", "w")
SaveMetricsResults.write("R Squared across the 5-fold cross-validation is %0.2f (+/- %0.2f)" % (np.round(R25CV.mean()*100,2), np.round(2.571*R25CV.std()/math.sqrt(5)*100,2)))
SaveMetricsResults.write("\n")
SaveMetricsResults.write("Mean Squared Error across the 5-fold cross-validation is %0.2f (+/- %0.2f)" % (np.round(MSE5CV.mean()*-1,2), np.round(2.571*MSE5CV.std()/math.sqrt(5)*-1,2)))
SaveMetricsResults.write("\n")
SaveMetricsResults.write("Mean Absolute Error across the 5-fold cross-validation is %0.2f (+/- %0.2f)" % (np.round(MAE5CV.mean()*-1,2), np.round(2.571*MAE5CV.std()/math.sqrt(5)*-1,2)))
SaveMetricsResults.close()
print("R2 across the 5-fold cross-validation is %0.2f (+/- %0.2f)" % (np.round(R25CV.mean()*100,2), np.round(2.571*R25CV.std()/math.sqrt(5)*100,2)))
print("Mean Square Error across the 5-fold cross-validation is %0.2f (+/- %0.2f)" % (np.round(MSE5CV.mean()*-1,2), np.round(2.571*MSE5CV.std()/math.sqrt(5)*-1,2)))
print("Mean Absolute Error across the 5-fold cross-validation is %0.2f (+/- %0.2f)" % (np.round(MAE5CV.mean()*-1,2), np.round(2.571*MAE5CV.std()/math.sqrt(5)*-1,2)))
print("R2, MSE and MAE saved as a txt.")

# Save predictions across the 5-fold cross-validation
CVPredictedCancerPrevalence = cross_val_predict(OptimisedElasticNet, X, y, cv=cv, verbose=0)
CVPredictedCancerPrevalence = np.round(CVPredictedCancerPrevalence,1)
PredictedCancerPrevalence = pd.DataFrame(CVPredictedCancerPrevalence)
PredictedCancerPrevalence.reset_index(level=0, inplace=True)
PredictedCancerPrevalence.to_csv("City/CityModelPredictedCancerPrevalence.csv", header=["id","predictedprevalence"],index=False)

# Save census list with id and actual cancer prevalence
CensusActualCancerPrevalence = Dataset[["census","cancerprevalence"]]
CensusActualCancerPrevalence = np.round(CensusActualCancerPrevalence,1)
CensusActualCancerPrevalence = pd.DataFrame(CensusActualCancerPrevalence)
CensusActualCancerPrevalence.reset_index(level=0, inplace=True)
CensusActualCancerPrevalence
CensusActualCancerPrevalence.to_csv("City/CityModelActualCancerPrevalenceWithIndex.csv", header=["id","census","actualprevalence"],index=False)

# Merge actual and predicted cancer prevalence
PredictedCancerPrevalence = pd.read_csv("City/CityModelPredictedCancerPrevalence.csv", dtype={"census": str}, header=0)
CensusActualCancerPrevalence = pd.read_csv("City/CityModelActualCancerPrevalenceWithIndex.csv", dtype={"census": str}, header=0)
CensusActualPredictedCancerPrevalence = CensusActualCancerPrevalence.set_index("id").join(PredictedCancerPrevalence.set_index("id"))
CensusActualPredictedCancerPrevalence.to_csv("City/CityModelActualPredictedCancerPrevalenceWithIndex.csv", header=["census","actualprevalence","predictedprevalence"],index=False)
print("Actual and predicted cancer prevalences saved as a csv.")

# Plot actual vs predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.lmplot(x="actualprevalence", y="predictedprevalence", data=CensusActualPredictedCancerPrevalence)
FigureCancerPrevalence.set(xlim=(0,12), ylim=(0,12), xlabel="Actual cancer prevalence", ylabel="Predicted cancer prevalence (ResNet)")
FigureCancerPrevalence.fig.set_size_inches(10,10)
#sns.despine(offset=10, trim=False);
FigureCancerPrevalence.savefig("Figures/CityModelRegressionPlot.png", format="png", dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png.")