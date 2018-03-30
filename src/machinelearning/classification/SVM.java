package machinelearning.classification;

import java.util.ArrayList;
import java.util.Iterator;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import machinelearning.dataset.DatasetMatrices;
import machinelearning.validation.APRStatistics;
import machinelearning.validation.BinaryAPRStatistics;

/**
 * An abstraction layer for working with the LIBSVM library.
 * https://github.com/cjlin1/libsvm
 *
 * @author evanc
 */
public class SVM {

	/**
	 * Creates an SVM problem used to train a LIBSVM model
	 *
	 * @param trainingData The training data to formulate for the model
	 * @param positiveClass The class to use as the positive class. All other classes
	 * will be considered negative
	 * @return An svm_problem which can be used to train an SVM model
	 */
	public static svm_problem createSvmProblem(DatasetMatrices trainingData, double positiveClass) {
		svm_problem trainingProblem = new svm_problem();
		// Components of svm problem:
		// l - Number of training instances
		// y[] - Label vector
		// x[][] - design matrix of svm_nodes
		trainingProblem.l = trainingData.getN();
		trainingProblem.y = trainingData.getLabelVector().clone();
		for(int i = 0; i < trainingProblem.y.length; i++) {
			double label = (trainingProblem.y[i] == positiveClass ? 1.0 : 0.0);
			trainingProblem.y[i] = label;
		}

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
	public static svm_problem createSvmProblem(DatasetMatrices trainingData) {
		return createSvmProblem(trainingData, 1.0);
	}

	/**
	 * Gets a trained SVM model based on the given data and hyperparameters.
	 *
	 * @param trainingProblem The training data
	 * @param kernel The type of kernel to use (linear or RBF)
	 * @param c The value of the hyperparameter C (soft-margin cost)
	 * @param gamma Gamma value for RBF kernel which controls the variance of the Gaussian
	 * @param includeProb Whether or not to include probability information in the model.
	 * Including probability generally results in slower training.
	 * @return The trained SVM model
	 */
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

	/**
	 * Gets the performance of a given model on a set of test data.
	 *
	 * @param model The SVM model whose performance should be evaluated
	 * @param testData The test data to evaluate the model against
	 * @param positiveClass The class label to use as the positive class
	 * @return A collection of accuracy, precision, and recall statistics
	 */
	public static BinaryAPRStatistics getModelPerformance(svm_model model, DatasetMatrices testData, double positiveClass) {
		double[] testLabels = testData.getLabelVector();
		double[][] testDesignMatrix = testData.getDesignMatrix();

		int tp = 0, tn = 0, fp = 0, fn = 0;
		for(int i = 0; i < testData.getN(); i++) {
			double label = testLabels[i] == positiveClass ? 1 : 0;
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
	public static BinaryAPRStatistics getModelPerformance(svm_model model, DatasetMatrices testData) {
		return getModelPerformance(model, testData, 1.0);
	}

	/**
	 * Calculates the AUC (area under the curve) of a model.
	 *
	 * Requires that the passed model has been trained with probability information included.
	 *
	 * @param model The model to get the AUC for
	 * @param testData The test data
	 * @param positiveClass The class label to use as the positive class
	 * @return The AUC
	 */
	public static double calculateAuc(svm_model model, DatasetMatrices testData, double positiveClass) {
		ArrayList<SVMResult> results = getProbabilityPredictions(model, testData, positiveClass);
		double posCount = 0;
		double negCount = 0;
		// TODO try to obtain this while doing the predictions
		for(SVMResult result : results) {
			if(result.actualLabel == 0)
				negCount++;
			else
				posCount++;
		}

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
	public static double calculateAuc(svm_model model, DatasetMatrices testData) {
		return calculateAuc(model, testData, 1.0);
	}

	/**
	 * Predicts positive class probabilities for each of the instances in the test data
	 * set using the given model.
	 *
	 * @param model The model to use for prediction
	 * @param testData The test data to get predictions for
	 * @param positiveClass The class label to use as the positive class
	 * @return A list of probability predictions for each of the test data instances
	 */
	public static ArrayList<SVMResult> getProbabilityPredictions(svm_model model, DatasetMatrices testData, double positiveClass) {
		double[] testLabels = testData.getLabelVector();
		double[][] testDesignMatrix = testData.getDesignMatrix();
		double[] probabilities = new double[2];

		// LIBSVM will sometimes make the positive class the 0 index in probability arrays,
		// other times it will be 1 index.
		int[] classLabels = new int[2];
		svm.svm_get_labels(model, classLabels);
		int posClassIndex = classLabels[0] == 0 ? 1 : 0;

		ArrayList<SVMResult> results = new ArrayList<SVMResult>();
		double posCount = 0;
		double negCount = 0;
		for(int i = 0; i < testData.getN(); i++) {
			double label = (testLabels[i] == positiveClass ? 1 : 0);
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
			results.add(new SVMResult(label, probabilities[posClassIndex]));
		}
		return results;
	}
	public static ArrayList<SVMResult> getProbabilityPredictions(svm_model model, DatasetMatrices testData) {
		return getProbabilityPredictions(model, testData, 1.0);
	}

	/**
	 * Generates the set of (X,Y) points in a ROC curve for a given set of predictions.
	 *
	 * @param results The probability predictions and actual class labels from a set of test data
	 * @return The (X,Y) data points to use in drawing the ROC-AUC curve
	 */
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

	/**
	 * Gets the performance of a multiclass SVM problem, given a set of 1-vs-all SVM models that have been trained
	 * for each class label.
	 *
	 * @param classModels The 1-vs-all models trained for each class. These should be in the same order as the indices
	 * of the class labels themselves.
	 * @param testData The test dataset
	 * @return Acurracy, precision, and recall information for each class
	 */
	public static APRStatistics getMulticlassPerformance(ArrayList<svm_model> classModels, DatasetMatrices testData) {
		ArrayList<ArrayList<SVMResult>> modelResults = new ArrayList<ArrayList<SVMResult>>();
		for(int positiveClass = 0; positiveClass < classModels.size(); positiveClass++) {
			svm_model model = classModels.get(positiveClass);
			modelResults.add(getProbabilityPredictions(model, testData, positiveClass));
		}
		// Now we have collected probability results for each of the models. We predict the class that has the highest
		// probability prediction of all the models
		ArrayList<Double> predictions = new ArrayList<Double>();
		for(int i = 0; i < testData.getN(); i++) {
			double currentPrediction = 0;
			double currentBestProbability = 0;
			for(int classLabel = 0; classLabel < modelResults.size(); classLabel++) {
				SVMResult result = modelResults.get(classLabel).get(i);
				if(result.probability > currentBestProbability) {
					currentPrediction = classLabel;
					currentBestProbability = result.probability;
				}
			}
			predictions.add(currentPrediction);
		}

		// Calculate accuracy
		int correctPredictions = 0;
		double[] testLabels = testData.getLabelVector();
		for(int i = 0; i < testLabels.length; i++) {
			double actual = testLabels[i];
			double prediction = predictions.get(i);
			if(actual == prediction) {
				correctPredictions++;
			}
		}
		double accuracy = (double)correctPredictions / (double)testLabels.length * 100;

		// Calculate precision and recall for each class
		ArrayList<Double> precisions = new ArrayList<Double>();
		ArrayList<Double> recalls = new ArrayList<Double>();
		ArrayList<Integer> classLabels = new ArrayList<Integer>();
		for(int classLabel = 0; classLabel < classModels.size(); classLabel++) {
			classLabels.add(classLabel);
			int tp = 0, fp = 0, fn = 0;
			for(int i = 0; i < testLabels.length; i++) {
				double actual = testLabels[i];
				double prediction = predictions.get(i);
				if(actual == classLabel && prediction == classLabel) {
					tp++;
				} else if (prediction == classLabel && actual != classLabel) {
					fp++;
				} else if (actual == classLabel && prediction != classLabel) {
					fn++;
				}
			}
			double precision = (double)tp / (tp + fp) * 100;
			double recall = (double)tp / (tp + fn) * 100;
			// Sometimes a class won't be represented in a test set. In that case, just say we have
			// 100% precision or recall to prevent NaN results.
			if(Double.isNaN(precision))
				precision = 100;
			if(Double.isNaN(recall))
				recall = 100;
			precisions.add(precision);
			recalls.add(recall);
		}

		return new APRStatistics(accuracy, classLabels, precisions, recalls, null);
	}
}
