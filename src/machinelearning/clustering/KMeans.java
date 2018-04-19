package machinelearning.clustering;

import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

import machinelearning.dataset.DatasetMatrices;

/**
 * A K-means clustering of data.
 *
 * @author evanc
 */
public class KMeans extends ClusteringModel {

	public KMeans(DatasetMatrices datasetMatrices, int k) {
		super(datasetMatrices, k);
	}

	/**
	 * Create the clusters using a specialized version of the EM
	 * algorithm
	 *
	 * @param tolerance Iteration will cease when change in SSE
	 * decreases below this threshold
	 */
	@Override
	public void createClusters(double tolerance) {
		initializeCentroids();

		double previousSse = getSumSquaredErrors();
		double diffFromPreviousSse = Double.POSITIVE_INFINITY;

		// Converge when change in SSE drops below the given threshold
		while(diffFromPreviousSse >= tolerance) {
			makeClusterAssignments();
			updateCentroids();
			double sse = getSumSquaredErrors();
			diffFromPreviousSse = previousSse - sse;
			previousSse = sse;
		}
	}

	/**
	 * Initialize the cluster centroids by selecting K random
	 * points from our training data.
	 */
	private void initializeCentroids() {
		// Pick random data points to use as the initial centroids
		// Make sure that we don't accidentally pick the same point
		// more than once
		ArrayList<Integer> usedPoints = new ArrayList<Integer>();
		for(int cluster = 0; cluster < K; cluster++ ) {
			int i;
			do {
				i = ThreadLocalRandom.current().nextInt(N);
			} while(usedPoints.contains(i));
			usedPoints.add(i);
			centroids[cluster] = designMatrix.getRowVector(i);
		}
	}

	/**
	 * Updates the cluster assignments for each instance based on the
	 * current centroids. Each instance is assigned to the cluster
	 * whose centroid is the smallest distance from the instance.
	 */
	private void makeClusterAssignments() {
		for(int i = 0; i < N; i++) {
			RealVector instance = designMatrix.getRowVector(i);
			// Check the distance from each centroid and select the closest one
			int bestCluster = 0;
			double smallestDistance = Double.POSITIVE_INFINITY;
			for(int cluster = 0; cluster < K; cluster++) {
				RealVector centroid = centroids[cluster];
				double distance = instance.getDistance(centroid);
				if(distance < smallestDistance) {
					bestCluster = cluster;
					smallestDistance = distance;
				}
			}
			// Record the cluster selected for the instance
			clusterLabels[i] = bestCluster;
		}
	}

	/**
	 * Updates the centroidss of the clusters to be the mean of all the instances
	 * in the cluster
	 */
	private void updateCentroids() {
		for(int cluster = 0; cluster < K; cluster++) {
			ArrayList<RealVector> clusterInstances = getCluster(cluster);
			RealVector clusterSum = clusterInstances.stream()
					.reduce(MatrixUtils.createRealVector(new double[m]), (a, b) -> a.add(b));
			RealVector centroid = clusterSum.mapDivide(clusterInstances.size());
			centroids[cluster] = centroid;
		}
	}

	/**
	 * Gets all of the instances in the given cluster.
	 *
	 * @param clusterIx The index of the cluster
	 * @return List of feature vectors for instances in the cluster
	 */
	private ArrayList<RealVector> getCluster(int clusterIx)
	{
		ArrayList<RealVector> cluster = new ArrayList<RealVector>();
		for(int i = 0; i < N; i++) {
			if(clusterLabels[i] == clusterIx)
				cluster.add(designMatrix.getRowVector(i));
		}
		return cluster;
	}

	public RealVector[] getCentroids() {
		return centroids;
	}
}
