package machinelearning.regression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import machinelearning.dataset.DatasetMatrices;

/**
 * A linear regression model created on a given set of training data.
 *
 * Supports training by gradient descent and normal equations.
 *
 * @author evanc
 *
 */
public class LinearRegression extends Regression {

	private ArrayList<Double> trainingRmses;

	private static final int MAX_ITERATIONS = 1000;

	public LinearRegression(DatasetMatrices datasetMatrices) {
		// Convert the data arrays to matrices
		designMatrix = MatrixUtils.createRealMatrix(datasetMatrices.getDesignMatrix());
		labelVector = MatrixUtils.createRealVector(datasetMatrices.getLabelVector());
		N = datasetMatrices.getN();

		m = designMatrix.getColumnDimension();
		double[] initWeights = new double[m];
		Arrays.fill(initWeights, 0);
		// Initialize the weight vector to all zeros
		weights = MatrixUtils.createRealVector(initWeights);

		trainingRmses = new ArrayList<Double>();
	}

	/**
	 * Trains the regressor by using gradient descent to estimate weights.
	 *
	 * The number of iterations is hard-capped at 1000 iterations, but may
	 * stop earlier depending on the tolerance parameter.
	 *
	 * @param learningRate The learning rate term applied to the update step
	 * @param tolerance The minimum difference in RMSE between iterations. If
	 * the difference in RMSE between iterations is less than the tolerance,
	 * iteration will terminate.
	 */
	public void trainByGradientDescent(double learningRate, double tolerance) {
		int i = 0;
		double previousRmse = getTrainingRootMeanSquaredError();
		double diffFromPreviousRmse = Double.POSITIVE_INFINITY;
		trainingRmses.add(previousRmse);
		while(i++ < MAX_ITERATIONS && diffFromPreviousRmse >= tolerance) {
			// Estimate the gradient
			RealVector gradient = getGradientVector();
			// Update the weights
			weights = weights.subtract(gradient.mapMultiply(learningRate));
			// Update the current RMSE and difference from previous
			double rmse = getTrainingRootMeanSquaredError();
			trainingRmses.add(rmse);
			diffFromPreviousRmse = previousRmse - rmse;
			previousRmse = rmse;
		}
	}

	/**
	 * Trains the regressor by using gradient descent to estimate weights.
	 *
	 * The number of iterations is hard-capped at 1000 iterations, but may
	 * stop earlier depending on the tolerance parameter.
	 *
	 * @param learningRate The learning rate term applied to the update step
	 * @param tolerance The minimum difference in RMSE between iterations. If
	 * the difference in RMSE between iterations is less than the tolerance,
	 * iteration will terminate.
	 * @param randomizeWeights If true, all the weights will be randomly initialized
	 * between -1 and 1 before starting gradient descent
	 */
	public void trainByGradientDescent(double learningRate, double tolerance, boolean randomizeWeights) {
		if(randomizeWeights) {
			double[] initWeights = new double[m];
			Random r = new Random();
			for(int i = 0; i < initWeights.length; i++) {
				initWeights[i] = r.nextDouble() * 2 + 1;
			}
			weights = MatrixUtils.createRealVector(initWeights);
		}
		trainByGradientDescent(learningRate, tolerance);
	}

	/**
	 * Estimates the weight vector by solving the normal equation for least squares
	 *
	 * w = (X_transpose * X)^-1 * X_transpose * y
	 * where X is the design matrix, y is the label vector, and w is the weight vector
	 */
	public void trainByNormalEquations() {
		// Note - the LUDecomposition solver fails for Sinusoid maxP 11 because
		// it claims the matrix is singular
		RealMatrix XdotProductXInverse = new SingularValueDecomposition(designMatrix.transpose().multiply(designMatrix))
				.getSolver().getInverse();
		RealMatrix labelMatrix = MatrixUtils.createColumnRealMatrix(labelVector.toArray());
		RealMatrix XtransposeY = designMatrix.transpose().multiply(labelMatrix);
		// This is possible because matrix multiplication is associative (i.e. A(BC) = (AB)C)
		RealMatrix result = XdotProductXInverse.multiply(XtransposeY);
		weights = result.getColumnVector(0);
	}

	/**
	 * Get a list of the RMSE on the training data for each iteration of training
	 * by gradient descent.
	 *
	 * @return List of RMSE for training iterations
	 */
	public ArrayList<Double> getTrainingRmses() {
		return trainingRmses;
	}

	/**
	 * Calculates the gradient vector for the current weights
	 *
	 * @return The gradient vector
	 */
	private RealVector getGradientVector() {
		double[] init = new double[m];
		Arrays.fill(init, 0);
		RealVector sum = MatrixUtils.createRealVector(init);
		for(int i = 0; i < N; i++) {
			// Get the ith row of the designMatrix
			RealVector xi = designMatrix.getRowVector(i);
			double yi = labelVector.getEntry(i);

			sum = sum.add(xi.mapMultiply(weights.dotProduct(xi) - yi));
		}
		return sum;
	}

}
