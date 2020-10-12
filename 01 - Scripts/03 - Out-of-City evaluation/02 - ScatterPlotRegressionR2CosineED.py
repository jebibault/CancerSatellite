# Load libraries
import seaborn as sns
import pandas as pd
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

#Load csv
df = pd.read_csv("CityModelMetricsOnOtherCities2.csv")

x=df["r2"] 
y=df["Mean cosine similarity"]

r2=stats.pearsonr(x, y)

#Plot regression with the distribution of the actual and the predicted cancer prevalence
sns.set_style("ticks")
r2Corr = sns.jointplot(
    x="r2", 
    y="Mean cosine similarity", 
    height=10, 
    data=df, 
    kind="reg", 
    xlim=(-100,100), 
    ratio=7,
)

print(r2)

r2Corr.set_axis_labels('r2 score', 'Cosine similarity', fontsize=13)

r2Corr.savefig("r2CorrCS.png", format="png", dpi=1000)

x=df["r2"] 
y=df["Mean euclidean distance"]

r2=stats.pearsonr(x, y)

#Plot regression with the distribution of the actual and the predicted cancer prevalence
sns.set_style("ticks")
r2Corr = sns.jointplot(
    x="r2", 
    y="Mean euclidean distance", 
    height=10, 
    data=df, 
    kind="reg", 
    xlim=(-100,100), 
    ratio=7,
)

print(r2)

r2Corr.set_axis_labels('r2 score', 'Mean euclidean distance', fontsize=13)

r2Corr.savefig("r2CorrED.png", format="png", dpi=1000)