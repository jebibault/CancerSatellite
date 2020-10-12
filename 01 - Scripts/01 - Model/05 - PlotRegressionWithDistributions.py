# Load libraries
import seaborn as sns
%matplotlib inline

# Load csv
CensusActualPredictedCancerPrevalence = pd.read_csv("CityModelActualPredictedCancerPrevalenceWithIndex.csv", dtype={"census": str}, header=0)

# Plot regression with the distribution of the actual and the predicted cancer prevalence
sns.set_style("ticks")
FigureCancerPrevalence = sns.jointplot(
    x="actualprevalence", 
    y="predictedprevalence", 
    height=10, 
    data=CensusActualPredictedCancerPrevalence, 
    kind="reg", 
    xlim=(0,12), 
    ylim=(0,12),
    ratio=7,
)

FigureCancerPrevalence = FigureCancerPrevalence.plot_joint(plt.scatter, edgecolor="white")

FigureCancerPrevalence.set_axis_labels('Actual cancer prevalence in City', 'Predicted cancer prevalence in City (ResNet50)', fontsize=13)

FigureCancerPrevalence.savefig("Figures/RegressionWithDistribution/CityModelRegressionPlot.png", format="png", dpi=1000)
print("Actual and predicted cancer prevalence plotted and saved as png.")