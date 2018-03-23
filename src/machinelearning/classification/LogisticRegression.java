package machinelearning.classification;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import machinelearning.dataset.DatasetMatrices;

/**
 * A logistic regression model for binary classification based
 * on the sigmoid function.
 *
 * Supports training by gradient descent.
 *
 * @author evanc
 */
public class LogisticRegression {
	private RealVector weights;
	private RealMatrix designMatrix;
	private RealVector labelVector;
	private int N;
	private int m;

	private static final int MAX_ITERATIONS = 1000;

	private ArrayList<Double> trainingLosses;

	public LogisticRegression(DatasetMatrices datasetMatrices) {
		// Convert the data arrays to matrices
		designMatrix = MatrixUtils.createRealMatrix(datasetMatrices.getDesignMatrix());
		labelVector = MatrixUtils.createRealVector(datasetMatrices.getLabelVector());
		N = datasetMatrices.getN();

		m = designMatrix.getColumnDimension();
		double[] initWeights = new double[m];
		Arrays.fill(initWeights, 0);
		// Initialize the weight vector to all zeros
		weights = MatrixUtils.createRealVector(initWeights);

		trainingLosses = new ArrayList<Double>();
	}

	/**
	 * Trains the logistic regression model by using gradient descent to estimate
	 * the weights that minimize the negative log-likelihood function.
	 *
	 * @param learningRate The learning rate applied to the update step
	 * @param tolerance The minimum change in loss between iterations. If the change
	 * in loss drops below this threshold between iterations, the method will stop.
	 * @param regularization The lambda value to apply to L2 regularization. A value of 0
	 * is equivalent to no regularization.
	 */
	public void trainByGradientDescent(double learningRate, double tolerance, double regularization) {
		int i = 0;
		double previousLoss = getNegativeLogLikelihood();
		double diffFromPreviousLoss = Double.POSITIVE_INFINITY;
		trainingLosses.add(previousLoss);
		while(i++ < MAX_ITERATIONS && diffFromPreviousLoss >= tolerance) {
			// Calculate the gradient
			RealVector gradient = getGradientVector(regularization);
			// Update the weights
			weights = weights.subtract(gradient.mapMultiply(learningRate));
			// Update the change in loss
			double loss = getNegativeLogLikelihood();
			diffFromPreviousLoss = previousLoss - loss;
			previousLoss = loss;
			trainingLosses.add(loss);
		}
	}

	/**
	 * Trains the logistic regression model by using gradient descent to estimate
	 * the weights that minimize the negative log-likelihood function.
	 *
	 * @param learningRate The learning rate applied to the update step
	 * @param tolerance The minimum change in loss between iterations. If the change
	 * in loss drops below this threshold between iterations, the method will stop.
	 */
	public void trainByGradientDescent(double learningRate, double tolerance) {
		trainByGradientDescent(learningRate, tolerance, 0);
	}

	/**
	 * Gets the percent accuracy of the model on the given set of data.
	 *
	 * @param data The data to evaluate the model with
	 * @return The accuracy, as a percentage
	 */
	public double getAccuracyPercentage(DatasetMatrices data) {
		RealMatrix sampleDesignMatrix = MatrixUtils.createRealMatrix(data.getDesignMatrix());
		RealVector sampleLabelVector = MatrixUtils.createRealVector(data.getLabelVector());
		int correctGuesses = 0;
		for(int i = 0; i < data.getN(); i++) {
			int prediction = getPrediction(sampleDesignMatrix.getRowVector(i));
			if((int)sampleLabelVector.getEntry(i) == prediction) {
				correctGuesses++;
			}
		}
		return ((double)correctGuesses / (double)data.getN()) * 100;
	}

	/**
	 * Gets the precision percentage for the given data set. Precision is defined as the
	 * percentage of positive predictions that were correct.
	 *
	 * @param data The data to evaluate the model with
	 * @return The precision, as a percentage
	 */
	public double getPrecisionPercentage(DatasetMatrices data) {
		RealMatrix sampleDesignMatrix = MatrixUtils.createRealMatrix(data.getDesignMatrix());
		RealVector sampleLabelVector = MatrixUtils.createRealVector(data.getLabelVector());
		int correctPositivePredictions = 0;
		int positivePredictions = 0;
		for(int i = 0; i < data.getN(); i++) {
			int label = (int)sampleLabelVector.getEntry(i);
			int prediction = getPrediction(sampleDesignMatrix.getRowVector(i));
			// Only consider examples that were predicted to be positive
			if(prediction == 1) {
				positivePredictions++;
				if(prediction == label) {
					correctPositivePredictions++;
				}
			}
		}
		return (double)correctPositivePredictions / (double)positivePredictions * 100;
	}

	/**
	 * Gets the recall percentage for the given data set. Recall is defined as the percentage
	 * of positive examples that are correctly classified.
	 *
	 * @param data The data to evaluate the model with
	 * @return The recall, as a percentage
	 */
	public double getRecallPercentage(DatasetMatrices data) {
		RealMatrix sampleDesignMatrix = MatrixUtils.createRealMatrix(data.getDesignMatrix());
		RealVector sampleLabelVector = MatrixUtils.createRealVector(data.getLabelVector());
		int correctPositiveExamples = 0;
		int positiveExamples = 0;
		for(int i = 0; i < data.getN(); i++) {
			int label = (int)sampleLabelVector.getEntry(i);
			// Only consider positive examples
			if(label == 1) {
				int prediction = getPrediction(sampleDesignMatrix.getRowVector(i));
				positiveExamples++;
				if(prediction == label) {
					correctPositiveExamples++;
				}
			}
		}
		return (double)correctPositiveExamples / (double)positiveExamples * 100;
	}

	/**
	 * Gets a list of the loss values for each iteration of gradient descent
	 *
	 * @return The list of loss values
	 */
	public ArrayList<Double> getTrainingLosses() {
		return trainingLosses;
	}

	/**
	 * Predicts a binary label value (0 or 1) for the given set of features
	 *
	 * @param featureVector The input features to the model
	 * @return The prediction, either 0 or 1
	 */
	private int getPrediction(RealVector featureVector) {
		return weights.dotProduct(featureVector) >= 0 ? 1 : 0;
	}

	/**
	 * Gets the result of the sigmoid function at x.
	 *
	 * @param x The input to the sigmoid function
	 * @return The output of the sigmoid function
	 */
	private double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	/**
	 * Gets the negative log-likelihood (aka cross-entropy) for the current
	 * weight values and training set.
	 *
	 * NLL is the "loss" function for logistic regression, i.e. the function
	 * being minimized.
	 *
	 * @return The negative log-likelihood
	 */
	private double getNegativeLogLikelihood() {
		double sum = 0;
		for(int i = 0; i < N; i++) {
			double sigmoidWTransposeX = sigmoid(weights.dotProduct(designMatrix.getRowVector(i)));
			double yi = labelVector.getEntry(i);
			// Note: beware about cases when we try to take the log of zero, since this is undefined
			// This can happen sometimes when the learning rate is too high.
			sum += (yi == 1 ? Math.log(sigmoidWTransposeX) : Math.log(1 - sigmoidWTransposeX));
		}
		return -sum;
	}

	/**
	 * Calculates the gradient vector used to update the weights
	 * in each iteration during gradient descent
	 *
	 * @return The gradient vector
	 */
	private RealVector getGradientVector(double lambda) {
		// Construct "O" vector, which is a vector of the sigmoid function applied
		// to the dot product of the weight vector with each input feature vector
		// in the training set
		double[] O = new double[N];
		for(int i = 0; i < N; i++) {
			O[i] = sigmoid(weights.dotProduct(designMatrix.getRowVector(i)));
		}
		RealVector sigmoidVector = MatrixUtils.createRealVector(O);
		RealMatrix oMinusY = MatrixUtils.createColumnRealMatrix((sigmoidVector.subtract(labelVector)).toArray());
		RealVector gradient = designMatrix.transpose().multiply(oMinusY).getColumnVector(0);

		// Perform regularization - add lambda*w_i to each ith entry in the gradient, EXCEPT w0
		// since we don't want to penalize the intercept
		RealVector regularizationVector = weights.copy().mapMultiply(lambda);
		regularizationVector.setEntry(0, 0);

		return gradient.add(regularizationVector);
	}
}
