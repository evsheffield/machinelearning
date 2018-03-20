package machinelearning.classification;

import java.util.Arrays;
import java.util.function.BiFunction;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import machinelearning.dataset.DatasetMatrices;
import machinelearning.validation.BinaryAPRStatistics;

public class Perceptron {
	private RealVector weights;
	private RealMatrix designMatrix;
	private RealVector labelVector;
	private int N;
	private int m;

	private static final int MAX_EPOCHS = 500;

	public Perceptron(DatasetMatrices datasetMatrices) {
		// Convert the data arrays to matrices
		designMatrix = MatrixUtils.createRealMatrix(datasetMatrices.getDesignMatrix());
		labelVector = MatrixUtils.createRealVector(datasetMatrices.getLabelVector());
		N = datasetMatrices.getN();

		m = designMatrix.getColumnDimension();
		double[] initWeights = new double[m];
		Arrays.fill(initWeights, 0);
		// Initialize the weight vector to all zeros
		weights = MatrixUtils.createRealVector(initWeights);
	}

	public void trainByPerceptronAlgorithm(double learningRate) {
		int currentEpoch = 0;
		while(++currentEpoch <= MAX_EPOCHS) {
			boolean madeMistake = false;
			// For one epoch, iterate over all the training samples
			for(int i = 0; i < N; i++) {
				RealVector x = designMatrix.getRowVector(i);
				// Map class 0 to -1
				double y = getLabel(labelVector, i);
				double yHat = weights.dotProduct(x);
				// Update the weights when there is a misclassification
				if(y * yHat <= 0) {
					madeMistake = true;
					weights = weights.add(x.mapMultiply(y * learningRate));
				}
			}
			// We have converged when we process an entire epoch without
			// making a mistake
			if(!madeMistake) {
				return;
			}
		}
	}

	public void trainByDualPerceptron(BiFunction<RealVector, RealVector, Double> kernelFunction) {
		// Initialize our list of alphas, which represent the number of times we
		// have made a mistake for each item in the training data
		double[] initAlphas = new double[N];
		Arrays.fill(initAlphas, 0);
		RealVector alphas = MatrixUtils.createRealVector(initAlphas);

		int currentEpoch = 0;
		while(++currentEpoch <= MAX_EPOCHS) {
			// For one epoch, iterate over all the training samples
			for(int i = 0; i < N; i++) {
				RealVector x = designMatrix.getRowVector(i);
				// Map class 0 to -1
				double y = getLabel(labelVector, i);
				double estimate = 0;
				for(int j = 0; j < N; j++) {
					RealVector xj = designMatrix.getRowVector(j);
					// TODO
//					double yj = getLabel(labelVector, j)
				}
			}
		}
	}

	public BinaryAPRStatistics getPerformance(DatasetMatrices data) {
		RealMatrix sampleDesignMatrix = MatrixUtils.createRealMatrix(data.getDesignMatrix());
		RealVector sampleLabelVector = MatrixUtils.createRealVector(data.getLabelVector());
		int tp = 0, tn = 0, fp = 0, fn = 0;

		for(int i = 0; i < data.getN(); i++) {
			int label = getLabel(sampleLabelVector, i);
			int prediction = getPrediction(sampleDesignMatrix.getRowVector(i));
			if(label == 1) {
				if(prediction == 1)
					tp++;
				else
					fn++;
			} else {
				if(prediction == 1)
					fp++;
				else
					tn++;
			}
		}

		double accuracy = (double)(tp + tn) / (tp + tn + fp + fn) * 100;
		double precision = (double)tp / (tp + fp) * 100;
		double recall = (double)tp / (tp + fn) * 100;

		return new BinaryAPRStatistics(accuracy, precision, recall);
	}

	private int getPrediction(RealVector featureVector) {
		return weights.dotProduct(featureVector) >= 0 ? 1 : -1;
	}

	private int getLabel(RealVector labels, int i) {
		return labelVector.getEntry(i) == 0 ? -1 : 1;
	}
}
