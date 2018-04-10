package machinelearning;

import java.awt.Font;
import java.util.ArrayList;
import java.util.Arrays;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import machinelearning.clustering.KMeans;
import machinelearning.dataset.Dataset;
import machinelearning.dataset.DatasetMatrices;
import machinelearning.dataset.Feature;
import machinelearning.dataset.FeatureType;

public class ClusteringExecutor {

	public static void main(String[] args) {
		// Load datasets
		Dataset dermatologyDataset = new Dataset(
				"data/dermatologyData.csv",
				generateContinuousFeatureList(34),
				new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3", "4", "5", "6"})));
		DatasetMatrices dermMatrices = new DatasetMatrices(
				dermatologyDataset.getFeatures(),
				dermatologyDataset.getInstances(),
				false);
		dermMatrices.setDesignMatrix(dermMatrices.getZScoreNormalizedDesignMatrix());

		Dataset vowelDataset = new Dataset(
				"data/vowelsData.csv",
				generateContinuousFeatureList(10),
				new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"})));
		DatasetMatrices vowelMatrices = new DatasetMatrices(
				vowelDataset.getFeatures(),
				vowelDataset.getInstances(),
				false);
		vowelMatrices.setDesignMatrix(vowelMatrices.getZScoreNormalizedDesignMatrix());

		Dataset glassDataset = new Dataset(
				"data/glassData.csv",
				generateContinuousFeatureList(9),
				new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3", "4", "5", "6"})));
		DatasetMatrices glassMatrices = new DatasetMatrices(
				glassDataset.getFeatures(),
				glassDataset.getInstances(),
				false);
		glassMatrices.setDesignMatrix(glassMatrices.getZScoreNormalizedDesignMatrix());

		Dataset ecoliDataset = new Dataset(
				"data/ecoliData.csv",
				generateContinuousFeatureList(7),
				new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3", "4", "5"})));
		DatasetMatrices ecoliMatrices = new DatasetMatrices(
				ecoliDataset.getFeatures(),
				ecoliDataset.getInstances(),
				false);
		ecoliMatrices.setDesignMatrix(ecoliMatrices.getZScoreNormalizedDesignMatrix());

		Dataset yeastDataset = new Dataset(
				"data/yeastData.csv",
				generateContinuousFeatureList(8),
				new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3", "4", "5", "6", "7", "8", "9"})));
		DatasetMatrices yeastMatrices = new DatasetMatrices(
				yeastDataset.getFeatures(),
				yeastDataset.getInstances(),
				false);
		yeastMatrices.setDesignMatrix(yeastMatrices.getZScoreNormalizedDesignMatrix());

		Dataset soybeanDataset = new Dataset(
				"data/soybeanData.csv",
				generateContinuousFeatureList(35),
				new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"})));
		DatasetMatrices soybeanMatrices = new DatasetMatrices(
				soybeanDataset.getFeatures(),
				soybeanDataset.getInstances(),
				false);
		soybeanMatrices.setDesignMatrix(soybeanMatrices.getZScoreNormalizedDesignMatrix());

		// --------------------------
		// K-Means Clustering
		// --------------------------
		System.out.println("--------------------------");
		System.out.println("K-Means Clustering");
		System.out.println("--------------------------");

		System.out.println("Dermatology Dataset");
		System.out.println("====================");
		XYChart dermChart = testKMeansClustering(dermMatrices, 1, 10, "Dermatology - K-Means Clustering");
		new SwingWrapper<XYChart>(dermChart).displayChart();

		System.out.println("Vowels Dataset");
		System.out.println("====================");
		XYChart vowelChart = testKMeansClustering(vowelMatrices, 1, 15, "Vowels - K-Means Clustering");
		new SwingWrapper<XYChart>(vowelChart).displayChart();

		System.out.println("Glass Dataset");
		System.out.println("====================");
		XYChart glassChart = testKMeansClustering(glassMatrices, 1, 10, "Glass - K-Means Clustering");
		new SwingWrapper<XYChart>(glassChart).displayChart();

		System.out.println("Ecoli Dataset");
		System.out.println("====================");
		XYChart ecoliChart = testKMeansClustering(ecoliMatrices, 1, 10, "Ecoli - K-Means Clustering");
		new SwingWrapper<XYChart>(ecoliChart).displayChart();

		System.out.println("Yeast Dataset");
		System.out.println("====================");
		XYChart yeastChart = testKMeansClustering(ecoliMatrices, 1, 15, "Yeast - K-Means Clustering");
		new SwingWrapper<XYChart>(yeastChart).displayChart();

		System.out.println("Soybean Dataset");
		System.out.println("====================");
		XYChart soybeanChart = testKMeansClustering(soybeanMatrices, 1, 20, "Soybean - K-Means Clustering");
		new SwingWrapper<XYChart>(soybeanChart).displayChart();
	}

	/**
	 * Generates an array of continuous features with the specified size. Each feature
	 * is given a generic name with an ID for its position in the list.
	 *
	 * @param count The number of features to include
	 * @return The array of continuous features
	 */
	private static Feature[] generateContinuousFeatureList(int count) {
		Feature[] features = new Feature[count];
		for(int i = 0; i < count; i++) {
			features[i] = new Feature("feature_" + (i + 1), FeatureType.Continuous);
		}
		return features;
	}

	private static XYChart testKMeansClustering(DatasetMatrices data, int minK, int maxK, String chartTitle) {
		int bestKSse = -1, bestKNmi = -1;
		double lowestSse = Double.POSITIVE_INFINITY, highestNmi = Double.NEGATIVE_INFINITY;
		double[] xValues = new double[maxK - minK + 1];
		double[] yValues = new double[maxK - minK + 1];

		for(int k = minK; k <= maxK; k++) {
			System.out.println("\n" + k + " Clusters");
			KMeans clustering = new KMeans(data, k);
			clustering.createClusters(0.001);
			double sse = clustering.getSumSquaredErrors();
			double nmi = clustering.getNormalizedMutualInformation();
			int plotDataIndex = k - minK;
			xValues[plotDataIndex] = k;
			yValues[plotDataIndex] = sse;
			System.out.println("SSE: " + sse);
			System.out.println("NMI: " + nmi);
			if(sse < lowestSse) {
				lowestSse = sse;
				bestKSse = k;
			}
			if(nmi > highestNmi) {
				highestNmi = nmi;
				bestKNmi = k;
			}
		}

		System.out.println("\nBest K for minimizing SSE: " + bestKSse);
		System.out.println("Best K for maximizing NMI: " + bestKNmi);

		// Create a chart of SSE vs. K
		XYChart chart = new XYChartBuilder().width(1200).height(800).title(chartTitle)
				.xAxisTitle("K").yAxisTitle("Sum of Squared Errors").build();
		Font font = new Font("Default", Font.PLAIN, 24);
		chart.getStyler().setAxisTickLabelsFont(font);
		chart.getStyler().setAxisTitleFont(font);
		chart.getStyler().setChartTitleFont(font);
		chart.getStyler().setLegendVisible(false);

		chart.addSeries("SSE", xValues, yValues);
		return chart;
	}
}
