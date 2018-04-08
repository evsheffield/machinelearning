package machinelearning;

import java.util.ArrayList;
import java.util.Arrays;

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

		// --------------------------
		// K-Means Clustering
		// --------------------------
		System.out.println("--------------------------");
		System.out.println("K-Means Clustering");
		System.out.println("--------------------------");

		System.out.println("Dermatology Dataset");
		System.out.println("====================");
		testKMeansClustering(dermMatrices, 1, 10);
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

	private static void testKMeansClustering(DatasetMatrices data, int minK, int maxK) {
		for(int k = minK; k <= maxK; k++) {
			System.out.println("\n" + k + " Clusters");
			KMeans clustering = new KMeans(data, k);
			clustering.createClusters(0.001);
			double sse = clustering.getSumSquaredErrors();
			double nmi = clustering.getNormalizedMutualInformation();
			System.out.println("SSE: " + sse);
			System.out.println("NMI: " + nmi);
		}
	}
}
