package machinelearning.clustering;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import machinelearning.dataset.DatasetMatrices;

public class GaussianMixtureModel extends ClusteringModel {
	private DatasetMatrices data;
	private double[] priors;
	private RealMatrix[] covarianceMatrices;
	private double[][] responsibilities;

	public GaussianMixtureModel(DatasetMatrices datasetMatrices, int k) {
		super(datasetMatrices, k);
		data = datasetMatrices;
		priors = new double[k];
		covarianceMatrices = new RealMatrix[k];
		responsibilities = new double[N][k];
		// TODO Auto-generated constructor stub
	}

	@Override
	public void createClusters(double tolerance) {
		KMeans clustering = new KMeans(data, K);
		clustering.createClusters(0.001);
		createClusters(tolerance, clustering);
	}

	public void createClusters(double tolerance, KMeans initialCluster) {
		initializeModelParameters(initialCluster);

		double previousIdl = getIncompleteDataLogLikelihood();
		double diffFromPreviousIdl = Double.POSITIVE_INFINITY;

		while(diffFromPreviousIdl >= tolerance) {
			System.out.println("iter " + diffFromPreviousIdl);
			// E-step
			evaluateResponsibilities();
			// M-step
			estimateModelParameters();

			double idl = getIncompleteDataLogLikelihood();
			diffFromPreviousIdl = idl - previousIdl;
			previousIdl = idl;
		}

		setLabelsFromResponsibilities();
	}

	private void initializeModelParameters(KMeans initialCluster) {
		// Run K-means on the data. Use its centroids as our cluster
		// means and the labels to inform our cluster responsibilities.
		centroids = initialCluster.getCentroids();
		int[] labels = initialCluster.getClusterLabels();
		for(int i = 0; i < N; i++) {
			int label = labels[i];
			for(int j = 0; j < K; j++) {
				if(j == label)
					responsibilities[i][j] = 1;
				else
					responsibilities[i][j] = 0;
			}
		}
		estimateCovarianceMatrices();
		estimatePriors();
	}

	private void evaluateResponsibilities() {
		for(int i = 0; i < N; i++) {
			double[] instance = designMatrix.getRowVector(i).toArray();
			for(int clusterIx = 0; clusterIx < K; clusterIx++) {
				MultivariateNormalDistribution gauss = new MultivariateNormalDistribution(
						centroids[clusterIx].toArray(),
						covarianceMatrices[clusterIx].getData());
				double numerator = priors[clusterIx] * gauss.density(instance);

				double denominator = 0;
				for(int j = 0; j < K; j++) {
					MultivariateNormalDistribution denomGauss = new MultivariateNormalDistribution(
							centroids[j].toArray(),
							covarianceMatrices[j].getData());
					denominator += (priors[j] * denomGauss.density(instance));
				}

				responsibilities[i][clusterIx] = numerator / denominator;
				if(Double.isNaN(responsibilities[i][clusterIx])) {
					System.out.println("uh oh");
				}
			}
		}
	}

	private void estimateModelParameters() {
		estimateMeans();
		estimateCovarianceMatrices();
		estimatePriors();
	}

	private void estimateMeans() {
		for(int clusterIx = 0; clusterIx < K; clusterIx++) {
			double nK = getCountClusterInstances(clusterIx);
			RealVector sum = MatrixUtils.createRealVector(new double[m]);
			for(int i = 0; i < N; i++) {
				sum = sum.add(designMatrix.getRowVector(i).mapMultiply(responsibilities[i][clusterIx]));
			}
			centroids[clusterIx] = sum.mapDivide(nK);
		}
	}

	private void estimateCovarianceMatrices() {
		for(int clusterIx = 0; clusterIx < K; clusterIx++) {
			double nKFrac = 1.0 / getCountClusterInstances(clusterIx);
			RealMatrix covarianceMatrix = MatrixUtils.createRealMatrix(m, m);
			RealVector mean = centroids[clusterIx];
			for(int i = 0; i < N; i++) {
				RealVector row = designMatrix.getRowVector(i).subtract(mean);
				covarianceMatrix = covarianceMatrix.add(row.outerProduct(row).scalarMultiply(responsibilities[i][clusterIx]));
			}
			covarianceMatrices[clusterIx] = covarianceMatrix.scalarMultiply(nKFrac).add(MatrixUtils.createRealIdentityMatrix(m).scalarMultiply(0.000000001));
		}
	}

	private void estimatePriors() {
		for(int clusterIx = 0; clusterIx < K; clusterIx++) {
			double prior = getCountClusterInstances(clusterIx) / N;
			priors[clusterIx] = prior;
		}
	}

	private double getIncompleteDataLogLikelihood() {
		double sum = 0;
		for(int i = 0; i < N; i++) {
			double[] instance = designMatrix.getRowVector(i).toArray();
			double innerSum = 0;
			for(int clusterIx = 0; clusterIx < K; clusterIx++) {
				MultivariateNormalDistribution gauss = new MultivariateNormalDistribution(
						centroids[clusterIx].toArray(),
						covarianceMatrices[clusterIx].getData());
				innerSum += (priors[clusterIx] * gauss.density(instance));
			}
			sum += Math.log(innerSum);
		}

		return sum;
	}

	private void setLabelsFromResponsibilities() {
		for(int i = 0; i < N; i++) {
			int bestLabel = -1;
			double highestResponsibility = Double.NEGATIVE_INFINITY;

			for(int cluster = 0; cluster < K; cluster++) {
				if(responsibilities[i][cluster] > highestResponsibility) {
					bestLabel = cluster;
					highestResponsibility = responsibilities[i][cluster];
				}
			}
			clusterLabels[i] = bestLabel;
		}
	}

	/**
	 * Gets the effective "count" of instances in a cluster.
	 * Because this is soft-clustering, this is a real number
	 * instead of an integer as a single instance may contribute to
	 * one or more instances.
	 *
	 * @param clusterIndex
	 * @return The count of instances in the cluster
	 */
	private double getCountClusterInstances(int clusterIndex) {
		double sum = 0;
		for(int i = 0; i < N; i++) {
			sum += responsibilities[i][clusterIndex];
		}
		return sum;
	}

}
