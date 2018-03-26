package machinelearning.classification;

import java.util.ArrayList;
import java.util.Iterator;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import machinelearning.dataset.DatasetMatrices;
import machinelearning.validation.BinaryAPRStatistics;

/**
 * An abstraction layer for working with the LIBSVM library.
 * https://github.com/cjlin1/libsvm
 *
 * @author evanc
 */
public class SVM {

	public static svm_problem createSvmProblem(DatasetMatrices trainingData) {
		svm_problem trainingProblem = new svm_problem();
		// Components of svm problem:
		// l - Number of training instances
		// y[] - Label vector
		// x[][] - design matrix of svm_nodes
		trainingProblem.l = trainingData.getN();
		trainingProblem.y = trainingData.getLabelVector();

		double[][] designMatrix = trainingData.getDesignMatrix();
		svm_node[][] nodes = new svm_node[trainingData.getN()][trainingData.getM()];
		for(int i = 0; i < trainingData.getN(); i++) {
			for(int j = 0; j < trainingData.getM(); j++) {
				svm_node node = new svm_node();
				node.index = j + 1; // 1-based index for features
				node.value = designMatrix[i][j];
				nodes[i][j] = node;
			}
		}
		trainingProblem.x = nodes;

		return trainingProblem;
	}

	public static svm_model trainModel(svm_problem trainingProblem, KernelType kernel, double c, double gamma, boolean includeProb) {
		// Create parameters for the model training
		svm_parameter parameters = new svm_parameter();
		parameters.svm_type = svm_parameter.C_SVC;
		parameters.kernel_type = (kernel == KernelType.Linear ? svm_parameter.LINEAR : svm_parameter.RBF);
		parameters.C = c;
		parameters.gamma = gamma;
		parameters.probability = includeProb ? 1 : 0;
		parameters.cache_size = 3000;
		parameters.eps = 0.001;
		parameters.shrinking = 1;

		String check = svm.svm_check_parameter(trainingProblem, parameters);

		if(check != null) {
			System.out.println("ERROR with SVM parameters! " + check);
			System.exit(1);
		}
		return svm.svm_train(trainingProblem, parameters);
	}
	public static svm_model trainModel(svm_problem trainingProblem, KernelType kernel, double c, double gamma) {
		return trainModel(trainingProblem, kernel, c, gamma, false);
	}

	public static BinaryAPRStatistics getModelPerformance(svm_model model, DatasetMatrices testData) {
		double[] testLabels = testData.getLabelVector();
		double[][] testDesignMatrix = testData.getDesignMatrix();

		int tp = 0, tn = 0, fp = 0, fn = 0;
		for(int i = 0; i < testData.getN(); i++) {
			double label = testLabels[i];
			svm_node[] nodes = new svm_node[testData.getM()];
			for(int j = 0; j < testData.getM(); j++) {
				svm_node node = new svm_node();
				node.index = j + 1;
				node.value = testDesignMatrix[i][j];
				nodes[j] = node;
			}

			// Get the predictions
			double predicted = svm.svm_predict(model, nodes);

			if (label == 1) {
				if(predicted == 1)
					tp++;
				else
					fn++;
			} else {
				if(predicted == 1)
					fp++;
				else
					tn++;
			}
		}

		return new BinaryAPRStatistics(tp, tn, fp, fn);
	}

	public static double calculateAuc(svm_model model, DatasetMatrices testData) {
//		double[] testLabels = testData.getLabelVector();
//		double[][] testDesignMatrix = testData.getDesignMatrix();
//		double[] probabilities = new double[2];

		ArrayList<SVMResult> results = getProbabilityPredictions(model, testData);
		double posCount = 0;
		double negCount = 0;
		// TODO try to obtain this while doing the predictions
		for(SVMResult result : results) {
			if(result.actualLabel == 0)
				negCount++;
			else
				posCount++;
		}
//		for(int i = 0; i < testData.getN(); i++) {
//			double label = testLabels[i];
//			if(label == 0)
//				negCount++;
//			else
//				posCount++;
//			svm_node[] nodes = new svm_node[testData.getM()];
//			for(int j = 0; j < testData.getM(); j++) {
//				svm_node node = new svm_node();
//				node.index = j + 1;
//				node.value = testDesignMatrix[i][j];
//				nodes[j] = node;
//			}
//
//			// Get the predictions
//			double predicted = svm.svm_predict_probability(model, nodes, probabilities);
//			results.add(new SVMResult(label, probabilities[1]));
//		}

		// Sort the results in decreasing order of probability
		Iterator<SVMResult> sortedResults = results.stream().sorted((r1, r2) -> Double.compare(r2.probability, r1.probability)).iterator();
		double auc = 0, height = 0;
		double posFrac = 1.0 / posCount;
		double negFrac = 1.0 / negCount;

		while(sortedResults.hasNext()) {
			SVMResult res = sortedResults.next();
			if(res.actualLabel == 1) {
				height = height + posFrac;
			} else {
				auc = auc + (height * negFrac);
			}
		}

		return auc;
	}

	public static ArrayList<SVMResult> getProbabilityPredictions(svm_model model, DatasetMatrices testData) {
		double[] testLabels = testData.getLabelVector();
		double[][] testDesignMatrix = testData.getDesignMatrix();
		double[] probabilities = new double[2];

		ArrayList<SVMResult> results = new ArrayList<SVMResult>();
		double posCount = 0;
		double negCount = 0;
		for(int i = 0; i < testData.getN(); i++) {
			double label = testLabels[i];
			if(label == 0)
				negCount++;
			else
				posCount++;
			svm_node[] nodes = new svm_node[testData.getM()];
			for(int j = 0; j < testData.getM(); j++) {
				svm_node node = new svm_node();
				node.index = j + 1;
				node.value = testDesignMatrix[i][j];
				nodes[j] = node;
			}

			// Get the predictions
			double predicted = svm.svm_predict_probability(model, nodes, probabilities);
			results.add(new SVMResult(label, probabilities[1]));
		}
		return results;
	}

	public static double[][] getRocPoints(ArrayList<SVMResult> results) {
		Iterator<SVMResult> sortedResults = results.stream().sorted((r1, r2) -> Double.compare(r2.probability, r1.probability)).iterator();
		double[] xData = new double[results.size() + 1];
		double[] yData = new double[results.size() + 1];
		xData[0] = 0;
		xData[0] = 0;

		double posCount = 0;
		double negCount = 0;
		// TODO try to obtain this while doing the predictions
		for(SVMResult result : results) {
			if(result.actualLabel == 0)
				negCount++;
			else
				posCount++;
		}
		double posFrac = 1.0 / posCount;
		double negFrac = 1.0 / negCount;

		int i = 1;
		double currX = 0, currY = 0;
		while(sortedResults.hasNext()) {
			SVMResult res = sortedResults.next();
			if(res.actualLabel == 1) {
				// Move up
				currY += posFrac;
			} else {
				// Move right
				currX += negFrac;
			}
			xData[i] = currX;
			yData[i] = currY;
			i++;
		}

		return new double[][] { xData, yData };
	}
}
