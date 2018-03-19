package machinelearning.regression;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import machinelearning.dataset.DatasetMatrices;

/**
 * An abstract class for a regression model which includes
 * weights, dimensions, and the training designMatrix and labelVector.
 *
 * Additionally provides functionality for calculating root mean
 * squared error.
 *
 * @author evanc
 *
 */
public abstract class Regression {
	protected RealVector weights;
	protected RealMatrix designMatrix;
	protected RealVector labelVector;
	protected int N;
	protected int m;

	/**
	 * Gets the root mean squared errors of the given instances.
	 *
	 * @param features The features of the instances
	 * @param instances The instances to test
	 * @return The RMSE of the given test instances
	 */
	public double getRootMeanSquaredErrors(DatasetMatrices testData) {
		double sse = getSumOfSquaredErrors(
				MatrixUtils.createRealMatrix(testData.getDesignMatrix()),
				MatrixUtils.createRealVector(testData.getLabelVector()));
		return getRootMeanSquaredErrorHelper(sse, testData.getN());
	}

	/**
	 * Gets the root mean squared errors of the model on the training set
	 *
	 * @return RMSE on the training set
	 */
	public double getTrainingRootMeanSquaredError() {
		return getRootMeanSquaredErrorHelper(getTrainingSumOfSquaredErrors(), N);
	}

	/**
	 * Gets the sum of squared errors for the given design matrix and label vector
	 *
	 * @param designMat The design matrix of the input data
	 * @param labelVec The label vector of the input data
	 * @return The sum of squared errors of the model in the input data
	 */
	protected double getSumOfSquaredErrors(RealMatrix designMat, RealVector labelVec) {
		DescriptiveStatistics errors = new DescriptiveStatistics();
		for(int i = 0; i < designMat.getRowDimension(); i++) {
			// Get the ith row of the designMatrix
			RealVector xi = designMat.getRowVector(i);
			double yi = labelVec.getEntry(i);
			double error = weights.dotProduct(xi) - yi;
			errors.addValue(error);
		}
		return errors.getSumsq();
	}

	/**
	 * Gets the root mean squared errors given the sum of squared errors and count
	 *
	 * RMSE = sqrt(sse / N)
	 *
	 * @param sse The sum of squared errors
	 * @param count The number of instances, N
	 * @return The RMSE
	 */
	protected double getRootMeanSquaredErrorHelper(double sse, double count) {
		return Math.sqrt(sse / count);
	}

	/**
	 * Gets the sum of squared errors on the training set.
	 *
	 * @return The SSE of the training set
	 */
	protected double getTrainingSumOfSquaredErrors() {
		return getSumOfSquaredErrors(designMatrix, labelVector);
	}
}
