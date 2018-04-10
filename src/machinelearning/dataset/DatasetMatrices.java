package machinelearning.dataset;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * A matrix-based representation of a dataset with only continuous data.
 * Feature values are stored in a design matrix while label values are
 * stored in a label vector.
 *
 * This representation of a dataset eases the process of building
 * linear regression models, which depends heavily on matrix operations.
 *
 * @author evanc
 */
public class DatasetMatrices {
	private int N;
	private int m;
	private double[][] designMatrix;
	private double[] labelVector;
	private boolean addedConstantFeature;
	private Set<Double> distinctLabels;

	public DatasetMatrices(Feature[] features, ArrayList<Instance> instances, boolean addConstantFeature) {
		N = instances.size();
		m = features.length;
		if(addConstantFeature) {
			m++;
		}
		addedConstantFeature = addConstantFeature;

		designMatrix = new double[N][m];
		labelVector = new double[N];

		distinctLabels = new HashSet<Double>();

		for(int row = 0; row < N; row++) {
			Instance instance = instances.get(row);
			// Set the value in the label vector
			labelVector[row] = instance.getInstanceClass();
			distinctLabels.add(labelVector[row]);

			// Add a 1 for the constant feature
			int col = 0;
			if(addConstantFeature) {
				designMatrix[row][col] = 1;
				col++;
			}
			for(Feature f : features) {
				designMatrix[row][col++] = (double)instance.getFeatureValue(f.getName());
			}
		}
	}

	public DatasetMatrices(double[][] designMatrix, double[] labelVector, int N, int m) {
		this.N = N;
		this.m = m;
		this.designMatrix = designMatrix;
		this.labelVector = labelVector;
	}

	/**
	 * Returns a new dataset matrix object expanded with additional features up
	 * to the specified polynomial maxP.
	 *
	 * If maxP = 1, this will be identical to the original.
	 * If maxP = 2, then we will add a feature for each original feature which is the value squared.
	 * For maxP = 3, then we will add two features for each original feature for the value squared
	 * and the value cubed.
	 * ...and so on...
	 *
	 * @param maxP The maximum level of the polynomial expansion
	 * @return The new dataset matrix object with the expanded feature set
	 */
	public DatasetMatrices getPolynomialExpansion(int maxP) {
		int newM = !addedConstantFeature
				? m * maxP
				: (m - 1) * maxP + 1;
		double[][] newDesignMatrix = new double[N][newM];
		double[] newLabelVector = labelVector.clone();

		int startingColumn = 0;
		if(addedConstantFeature) {
			for(int i = 0; i < N; i++) {
				newDesignMatrix[i][0] = 1;
			}
			startingColumn = 1;
		}

		for(int row = 0; row < N; row++) {
			for(int col = startingColumn; col < m; col++) {
				for(int p = 1; p <= maxP; p++) {
					newDesignMatrix[row][col + (p - 1) * (m - startingColumn)] = Math.pow(designMatrix[row][col], p);
				}
			}
		}

		return new DatasetMatrices(newDesignMatrix, newLabelVector, N, newM);
	}

	/**
	 * Gets an array of the mean values of each of the m features.
	 *
	 * @return An array of mean values for features 0 .. m
	 */
	public double[] getFeatureMeans() {
		DescriptiveStatistics stats;
		RealMatrix matrix = MatrixUtils.createRealMatrix(designMatrix);
		double[] means = new double[m];
		for(int col = 0; col < m; col++) {
			stats = new DescriptiveStatistics(matrix.getColumn(col));
			means[col] = stats.getMean();
		}
		return means;
	}

	/**
	 * Gets the mean value of the data labels.
	 *
	 * @return
	 */
	public double getLabelMean() {
		DescriptiveStatistics stats = new DescriptiveStatistics(labelVector);
		return stats.getMean();
	}

	/**
	 * Gets the design matrix centered around the given mean values for
	 * each feature.
	 *
	 * @param mean The mean values to center the design matrix around
	 * @return The centered design matrix
	 */
	public double[][] getCenteredDesignMatrix(double[] means) {
		double[][] centered = new double[N][m];

		for(int row = 0; row < N; row++) {
			for(int col = 0; col < m; col++) {
				centered[row][col] = designMatrix[row][col] - means[col];
			}
		}
		return centered;
	}

	/**
	 * Gets the centered version of the label vector,
	 * i.e. the label vector with the mean value subtracted from each
	 * entry.
	 *
	 * @param mean The mean value to center the label vector around
	 * @return The centered label vector
	 */
	public double[] getCenteredLabelVector(double mean) {
		double[] centered = new double[N];
		for(int i = 0; i < N; i++) {
			centered[i] = labelVector[i] - mean;
		}

		return centered;
	}

	/**
	 * Normalizes all the continuous features of the training and test datasets to
	 * their z-score.
	 *
	 * z-score of an individual value x is defined as:
	 * (x - mean) / sd
	 * Where mean and sd refer to the mean and standard deviation of the population
	 * (i.e. training set values)
	 */
	public double[][] getZScoreNormalizedDesignMatrix() {
		double[][] normDesignMatrix = designMatrix;
		// For each column
		int startingColumn = addedConstantFeature ? 1 : 0;
		for(int col = startingColumn; col < m; col++) {
			// Get the training mean and SD of the column
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for(int row = 0; row < N; row++) {
				stats.addValue(normDesignMatrix[row][col]);
			}
			double mean = stats.getMean();
			double standardDeviation = stats.getStandardDeviation();

			// Normalize the value of this feature for all instances
			for(int row = 0; row < N; row++) {
				double normalizedValue = (normDesignMatrix[row][col] - mean) / standardDeviation;
				if(Double.isNaN(normalizedValue))
					normalizedValue = 0;
				normDesignMatrix[row][col] = normalizedValue;
			}
		}
		return normDesignMatrix;
	}

	public int getN() {
		return N;
	}

	public int getM() {
		return m;
	}

	public double[][] getDesignMatrix() {
		return designMatrix;
	}

	public double[] getLabelVector() {
		return labelVector;
	}

	public Set<Double> getDistinctLabels() {
		return distinctLabels;
	}

	public void setDesignMatrix(double[][] designMatrix) {
		this.designMatrix = designMatrix;
	}

	public void setLabelVector(double[] labelVector) {
		this.labelVector = labelVector;
	}
}
