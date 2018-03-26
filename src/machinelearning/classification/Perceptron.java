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
	private RealVector alphas;
	private RealMatrix designMatrix;
	private RealVector labelVector;
	private int N;
	private int m;
	/** The bandwidth (gamma) used for the RBF kernel */
	private double bandwidth;

	private static final int MAX_EPOCHS = 100;

	private BiFunction<RealVector, RealVector, Double> linearKernel = (x, z) -> x.dotProduct(z);
	private BiFunction<RealVector, RealVector, Double> rbfKernel = (x, z) -> {
		RealVector xMinusZ = x.subtract(z);
		return Math.exp(-bandwidth * xMinusZ.dotProduct(xMinusZ));
	};

	public Perceptron(DatasetMatrices datasetMatrices) {
		// Convert the data arrays to matrices
		designMatrix = MatrixUtils.createRealMatrix(datasetMatrices.getDesignMatrix());
		labelVector = MatrixUtils.createRealVector(datasetMatrices.getLabelVector());
		N = datasetMatrices.getN();

		m = designMatrix.getColumnDimension();
	}

	public void trainByPerceptronAlgorithm(double learningRate) {
		// Initialize the weight vector to all zeros
		double[] initWeights = new double[m];
		Arrays.fill(initWeights, 0);
		weights = MatrixUtils.createRealVector(initWeights);

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

	public void trainByDualLinearKernel() {
		trainByDualPerceptron(linearKernel);
	}

	public void trainByDualGaussianKernel(double bandwidth) {
		this.bandwidth = bandwidth;
		trainByDualPerceptron(rbfKernel);
	}

	public void trainByDualPerceptron(BiFunction<RealVector, RealVector, Double> kernelFunction) {
		// Initialize our list of alphas, which represent the number of times we
		// have made a mistake for each item in the training data
		double[] initAlphas = new double[N];
		Arrays.fill(initAlphas, 0);
		alphas = MatrixUtils.createRealVector(initAlphas);

		int currentEpoch = 0;
		while(++currentEpoch <= MAX_EPOCHS) {
			boolean madeMistake = false;
			// For one epoch, iterate over all the training samples
			for(int i = 0; i < N; i++) {
				RealVector x = designMatrix.getRowVector(i);
				// Map class 0 to -1
				double y = getLabel(labelVector, i);
				double estimate = 0;
				for(int j = 0; j < N; j++) {
					RealVector xj = designMatrix.getRowVector(j);
					double yj = getLabel(labelVector, j);
					double alphaj = alphas.getEntry(j);

					estimate += (alphaj * yj * kernelFunction.apply(xj, x));
				}
				double yHat = estimate >= 0 ? 1 : -1;
				if(y != yHat) {
					alphas.setEntry(i, alphas.getEntry(i) + 1);
					madeMistake = true;
				}
			}

			if(!madeMistake) {
				break;
			}
		}
	}

	public BinaryAPRStatistics getPerformance(DatasetMatrices data, PerceptronTrainingType trainingType) {
		RealMatrix sampleDesignMatrix = MatrixUtils.createRealMatrix(data.getDesignMatrix());
		RealVector sampleLabelVector = MatrixUtils.createRealVector(data.getLabelVector());
		int tp = 0, tn = 0, fp = 0, fn = 0;

		for(int i = 0; i < data.getN(); i++) {
			int label = getLabel(sampleLabelVector, i);
			int prediction = getPrediction(sampleDesignMatrix.getRowVector(i), trainingType);
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

		return new BinaryAPRStatistics(tp, tn, fp, fn);
	}

	private int getPrediction(RealVector featureVector, PerceptronTrainingType trainingType) {
		switch(trainingType) {
			case DualLinearKernel:
				double sum = 0;
				for(int i = 0; i < N; i++) {
					double ai = alphas.getEntry(i);
					double yi = getLabel(labelVector, i);
					RealVector xi = designMatrix.getRowVector(i);
					double kernelVal = linearKernel.apply(xi, featureVector);
					sum += (ai * yi * kernelVal);
				}
				return sum >= 0 ? 1 : -1;
			case DualGaussianKernel:
				double sum1 = 0;
				for(int i = 0; i < N; i++) {
					double ai = alphas.getEntry(i);
					double yi = getLabel(labelVector, i);
					RealVector xi = designMatrix.getRowVector(i);
					double kernelVal = rbfKernel.apply(xi, featureVector);
					sum1 += (ai * yi * kernelVal);
				}
				return sum1 >= 0 ? 1 : -1;
			case Perceptron:
			default:
				return weights.dotProduct(featureVector) >= 0 ? 1 : -1;
		}
	}

	private int getLabel(RealVector labels, int i) {
		return labelVector.getEntry(i) == 0 ? -1 : 1;
	}
}
