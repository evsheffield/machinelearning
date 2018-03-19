package machinelearning.regression;

import java.util.Arrays;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import machinelearning.dataset.DatasetMatrices;

/**
 * A ridge regression model created on a given set of training data.
 *
 * Supports training by normal equations.
 *
 * @author evanc
 *
 */
public class RidgeRegression extends Regression {

	private double w0;

	/** Stores the mean value of each feature for centering test data */
	private double[] trainingFeatureMeans;
	private double trainingLabelMean;

	public RidgeRegression(DatasetMatrices datasetMatrices) {
		// Store the mean values of the training features and
		// labels. These will be used for centering the training
		// and test data.
		trainingFeatureMeans = datasetMatrices.getFeatureMeans();
		trainingLabelMean = datasetMatrices.getLabelMean();

		// Center the training data and labels
		designMatrix = MatrixUtils.createRealMatrix(datasetMatrices.getCenteredDesignMatrix(trainingFeatureMeans));
		labelVector = MatrixUtils.createRealVector(datasetMatrices.getCenteredLabelVector(trainingLabelMean));

		// Initialize the weights to 0;
		w0 = 0;
		double[] initWeights = new double[m];
		Arrays.fill(initWeights, 0);
		weights = MatrixUtils.createRealVector(initWeights);

		// Store the dimensions of the training data
		N = datasetMatrices.getN();
		m = designMatrix.getColumnDimension();
	}

	/**
	 * Estimates the weight vector by solving the normal equation
	 * for least squares with a weight penalty.
	 *
	 * w_ridge = (Z_transpose * Z + lambda * I)^-1 * Z_transpose * y
	 *
	 * @param lambda The scalar value applied to the weight minimizing term
	 */
	public void trainByNormalEquations(double lambda) {
		// Set the intercept to the mean of the training label
		w0 = trainingLabelMean;

		// Calculate the w_ridge
		RealMatrix zTransposeZ = designMatrix.transpose().multiply(designMatrix);
		RealMatrix lambdaIdentity = MatrixUtils.createRealIdentityMatrix(m).scalarMultiply(lambda);
		RealMatrix zTzPlusaLamdaIdentityInverse = new SingularValueDecomposition(zTransposeZ.add(lambdaIdentity))
				.getSolver().getInverse();
		RealMatrix labelMatrix = MatrixUtils.createColumnRealMatrix(labelVector.toArray());
		RealMatrix zTransposeY = designMatrix.transpose().multiply(labelMatrix);
		RealMatrix result = zTzPlusaLamdaIdentityInverse.multiply(zTransposeY);
		weights = result.getColumnVector(0);
	}

	/**
	 * Gets the root mean squared error on the provided test set.
	 *
	 * The test set should NOT be centered ahead of time.
	 *
	 * @param testData The test set to get the RMSE for
	 * @return The RMSE for the test set
	 */
	public double getTestRootMeanSquaredError(DatasetMatrices testData) {
		// Center the test data using the means from our training set
		RealMatrix centeredTestDesign = MatrixUtils.createRealMatrix(testData.getCenteredDesignMatrix(trainingFeatureMeans));
		DescriptiveStatistics errors = new DescriptiveStatistics();
		for(int i = 0; i < centeredTestDesign.getRowDimension(); i++) {
			RealVector xi = centeredTestDesign.getRowVector(i);
			double yi = testData.getLabelVector()[i];

			double yEstimated = weights.dotProduct(xi) + w0;
			double error = yEstimated - yi;
			errors.addValue(error);
		}
		double sse = errors.getSumsq();

		return getRootMeanSquaredErrorHelper(sse, centeredTestDesign.getRowDimension());
	}
}
