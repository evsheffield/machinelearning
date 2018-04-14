package machinelearning.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import machinelearning.dataset.DatasetMatrices;

/**
 * An abstract class which stores shared implementation
 * details for clustering models.
 *
 * @author evanc
 */
public abstract class ClusteringModel {
	protected RealMatrix designMatrix;
	protected double[] labels;
	protected Set<Double> distinctLabels;
	protected int N;
	protected int m;
	protected int K;
	protected int[] clusterLabels;
	protected RealVector[] centroids;

	public ClusteringModel(DatasetMatrices datasetMatrices, int k) {
		designMatrix = MatrixUtils.createRealMatrix(datasetMatrices.getDesignMatrix());
		labels = datasetMatrices.getLabelVector();
		distinctLabels = datasetMatrices.getDistinctLabels();
		N = datasetMatrices.getN();
		m = designMatrix.getColumnDimension();
		this.K = k;
		clusterLabels = new int[N];
		centroids = new RealVector[k];
	}

	public abstract void createClusters(double tolerance);

	public int[] getClusterLabels() {
		return clusterLabels;
	}

	/**
	 * Gets the sum of squared errors across all clusters.
	 * Error for an individual instance is defined as the Euclidean
	 * distance between that point and the centroid of its cluster.
	 *
	 * @return The sum of squared errors across all K clusters.
	 */
	public double getSumSquaredErrors() {
		double sum = 0;
		for(int i = 0; i < N; i++) {
			RealVector instance = designMatrix.getRowVector(i);
			RealVector clusterCentroid = centroids[clusterLabels[i]];
			sum += Math.pow(instance.getDistance(clusterCentroid), 2);
		}
		return sum;
	}

	/**
	 * Gets the normalized mutual information (NMI) for the cluster.
	 * NMI is defined as 2 x I(Y;C) / [H(Y) + H(C)]
	 * @return
	 */
	public double getNormalizedMutualInformation() {
		return (2 * getMutualInformation()) / (getEntropy(clusterLabels) + getEntropy(labels));
	}

	/**
	 * Gets all of the instance indices in the given cluster.
	 *
	 * @param clusterIx The index of the cluster
	 * @return List of instance indices of instances in the cluster
	 */
	private ArrayList<Integer> getClusterInstanceIndices(int clusterIx)
	{
		ArrayList<Integer> cluster = new ArrayList<Integer>();
		for(int i = 0; i < N; i++) {
			if(clusterLabels[i] == clusterIx)
				cluster.add(i);
		}
		return cluster;
	}

	/**
	 * Gets the log base 2 of a number.
	 *
	 * @param x The number
	 * @return log2(x)
	 */
	private static double log2(double x) {
		return Math.log(x) / Math.log(2);
	}

	/**
	 * Get the entropy of a set of instances in a node.
	 *
	 * @param instances The instances in the node
	 * @return The entropy
	 */
	private static double getEntropy(ArrayList<Double> values) {
		// Get the counts of each value
		HashMap<Double, Integer> valueCounts = new HashMap<Double, Integer>();
		for(int i = 0; i < values.size(); i++) {
			Double currValue = values.get(i);
			if(valueCounts.containsKey(currValue)) {
				int count = valueCounts.get(currValue) + 1;
				valueCounts.put(currValue, count);
			} else {
				valueCounts.put(currValue, 1);
			}
		}

		// Convert to probabilities by dividing by the total instances
		HashMap<Double, Double> probabilities = new HashMap<Double, Double>();
		double count = values.size();
		for(Double key : valueCounts.keySet()) {
			probabilities.put(key, (double)valueCounts.get(key) / count);
		}

		double entropy = 0;
		for(Double key : probabilities.keySet()) {
			double prob = probabilities.get(key);
			entropy -= (prob * log2(prob));
		}
		return entropy;
	}

	/**
	 * Get the entropy of a set of instances in a node.
	 *
	 * @param instances The instances in the node
	 * @return The entropy
	 */
	private static double getEntropy(double[] values) {
		ArrayList<Double> valuesList = DoubleStream.of(values).boxed()
				.collect(Collectors.toCollection(ArrayList::new));
		return getEntropy(valuesList);
	}

	/**
	 * Get the entropy of a set of instances in a node.
	 *
	 * @param instances The instances in the node
	 * @return The entropy
	 */
	private double getEntropy(int[] values) {
		return getEntropy(Arrays.stream(values).asDoubleStream().toArray());
	}

	/**
	 * Gets the class entropy within a specific cluster, H(Y|C)
	 *
	 * @param clusterIx The index of the cluster to get the class entropy for
	 * @return The conditional class entropy of the cluster
	 */
	private double getConditionalClassEntropyForCluster(int clusterIx) {
		ArrayList<Integer> cluster = getClusterInstanceIndices(clusterIx);
		double clusterProbability = (double)cluster.size() / N;

		// Count the instances of each class in the cluster
		HashMap<Double, Integer> classCounts = new HashMap<Double, Integer>();
		for(Double classLabel : distinctLabels) {
			classCounts.put(classLabel, 0);
		}
		for(int instanceIx : cluster) {
			// Get the class of the instance
			double instanceClass = labels[instanceIx];
			classCounts.put(instanceClass, classCounts.get(instanceClass) + 1);
		}

		// Get the conditional probability of each class for the cluster
		HashMap<Double, Double> probabilities = new HashMap<Double, Double>();
		double totalInstances = cluster.size();
		for(Double key : classCounts.keySet()) {
			probabilities.put(key, (double)classCounts.get(key) / totalInstances);
		}

		double sum = 0;
		for(Double key : probabilities.keySet()) {
			double prob = probabilities.get(key);
			// Skip 0 probabilities to avoid calculating log(0)
			if(prob != 0)
				sum += (prob * log2(prob));
		}

		return -(totalInstances / N) * sum;
	}

	/**
	 * Gets the mutual information for the clustering based
	 * on the actual labels. Mutual information is defined as:
	 * I(Y;C) = H(Y) - H(Y|C)
	 *
	 * @return The mutual information of the clustering
	 */
	private double getMutualInformation() {
		double mutualInformation = getEntropy(labels);
		for(int cluster = 0; cluster < K; cluster++) {
			mutualInformation -= getConditionalClassEntropyForCluster(cluster);
		}
		return mutualInformation;
	}
}
