package machinelearning.clustering;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import machinelearning.dataset.DatasetMatrices;

public class GaussianMixtureModel extends ClusteringModel {
	private DatasetMatrices data;
	private double[] priors;
	private RealVector[] means;
	private RealMatrix[] covarianceMatrices;
	private double[][] responsibilities;

	public GaussianMixtureModel(DatasetMatrices datasetMatrices, int k) {
		super(datasetMatrices, k);
		data = datasetMatrices;
		priors = new double[k];
		means = new RealVector[k];
		covarianceMatrices = new RealMatrix[k];
		responsibilities = new double[N][k];
		// TODO Auto-generated constructor stub
	}

	@Override
	public void createClusters(double tolerance) {
		initializeModelParameters();

		double previousIdl = getIncompleteDataLikelihood();
		double diffFromPreviousIdl = Double.NEGATIVE_INFINITY;

		while(diffFromPreviousIdl >= tolerance) {
			// E-step
			evaluateResponsibilities();
			// M-step
			estimateModelParameters();
		}
	}

	private void initializeModelParameters() {
		// TODO
		// Run K-means on the data. Use its centroids as our cluster
		// means and the labels to inform our cluster responsibilities.
		KMeans clustering = new KMeans(data, K);
		clustering.createClusters(0.001);
		means = clustering.getCentroids();
		int[] labels = clustering.getClusterLabels();
		for(int i = 0; i < N; i++) {
			int label = labels[i];
			for(int j = 0; j < K; j++) {
				if(j == label)
					responsibilities[i][j] = 1;
				else
					responsibilities[i][j] = 0;
			}
		}
	}

}
