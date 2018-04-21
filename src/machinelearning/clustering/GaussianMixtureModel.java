package machinelearning.clustering;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import machinelearning.dataset.DatasetMatrices;

/**
 * A Gaussian Mixture Model clustering.
 *
 * @author evanc
 */
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
	}

	/**
	 * Uses the EM algorithm to produce a GMM clustering. Creates
	 * a new K-Means clustering to use for initialization of model
	 * parameters.
	 *
	 * @param tolerance The threshold for improved incomplete data
	 * log likelihood. Iteration will cease once the change in LL
	 * falls below this threshold.
	 */
	@Override
	public void createClusters(double tolerance) {
		KMeans clustering = new KMeans(data, K);
		clustering.createClusters(0.001);
		createClusters(tolerance, clustering);
	}

	/**
	 * Uses the EM algorithm to produce a GMM clustering.
	 *
	 * @param tolerance The threshold for improved incomplete data
	 * log likelihood. Iteration will cease once the change in LL
	 * falls below this threshold.
	 * @param initialCluster The K-Means clustering used to
	 * initialize the model parameters.
	 */
	public void createClusters(double tolerance, KMeans initialCluster) {
		initializeModelParameters(initialCluster);

		double previousIdl = getIncompleteDataLogLikelihood();
		double diffFromPreviousIdl = Double.POSITIVE_INFINITY;

		while(diffFromPreviousIdl >= tolerance) {
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

	/**
	 * Initializes the model parameters based on the provided K-Means
	 * clustering.
	 *
	 * The cluster centers are used to initialize means, and initial
	 * covariance matrices and priors are calculating using the means
	 * and resulting cluster counts of the K-Means clustering.
	 *
	 * @param initialCluster The K-Means clustering to use for initialization
	 */
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

	/**
	 * Evaluate the expected responsibilities of each cluster
	 * for each instance based on the current model parameters.
	 *
	 * This constitutes the E-step of the EM algorithm.
	 */
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
			}
		}
	}

	/**
	 * Updates the model parameters based on the current responsibilities
	 * in order to maximize the incomplete data likelihood.
	 */
	private void estimateModelParameters() {
		estimateMeans();
		estimateCovarianceMatrices();
		estimatePriors();
	}

	/**
	 * Updates the model means (cluster centroids) based on the current
	 * responsibilities.
	 */
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

	/**
	 * Updates the model covariance matrices based on the current responsibilities
	 * and cluster means.
	 */
	private void estimateCovarianceMatrices() {
		// We add a small value to the diagonal of each covariance matrix to ensure that
		// it will be invertible.
		RealMatrix smallDiagonal = MatrixUtils.createRealIdentityMatrix(m).scalarMultiply(1e-9);
		for(int clusterIx = 0; clusterIx < K; clusterIx++) {
			double nKFrac = 1.0 / getCountClusterInstances(clusterIx);
			RealMatrix covarianceMatrix = MatrixUtils.createRealMatrix(m, m);
			RealVector mean = centroids[clusterIx];
			for(int i = 0; i < N; i++) {
				RealVector row = designMatrix.getRowVector(i).subtract(mean);
				covarianceMatrix = covarianceMatrix.add(row.outerProduct(row).scalarMultiply(responsibilities[i][clusterIx]));
			}
			covarianceMatrices[clusterIx] = covarianceMatrix.scalarMultiply(nKFrac).add(smallDiagonal);
		}
	}

	/**
	 * Updates the priors for each cluster.
	 */
	private void estimatePriors() {
		for(int clusterIx = 0; clusterIx < K; clusterIx++) {
			double prior = getCountClusterInstances(clusterIx) / N;
			priors[clusterIx] = prior;
		}
	}

	/**
	 * Gets the current incomplete data log likelihood based on the current
	 * state of the model parameters and responsibilities.
	 *
	 * @return The incomplete data log likelihood.
	 */
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

	/**
	 * Makes a "hard-clustering" assignment based on the current responsibilities.
	 * The cluster with the highest value in the z-vector of each instance is used
	 * as the hard cluster label.
	 */
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
