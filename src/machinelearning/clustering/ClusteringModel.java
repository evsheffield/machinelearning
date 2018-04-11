package machinelearning.clustering;

import java.util.Set;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

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

	public ClusteringModel(DatasetMatrices datasetMatrices, int k) {
		designMatrix = MatrixUtils.createRealMatrix(datasetMatrices.getDesignMatrix());
		labels = datasetMatrices.getLabelVector();
		distinctLabels = datasetMatrices.getDistinctLabels();
		N = datasetMatrices.getN();
		m = designMatrix.getColumnDimension();
		this.K = k;
		clusterLabels = new int[N];
	}

	public abstract void createClusters(double tolerance);

	public int[] getClusterLabels() {
		return clusterLabels;
	}
}
