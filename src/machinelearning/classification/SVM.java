package machinelearning.classification;

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

	public static svm_model trainModel(svm_problem trainingProblem, KernelType kernel, double c, double gamma) {
		// Create parameters for the model training
		svm_parameter parameters = new svm_parameter();
		parameters.svm_type = svm_parameter.C_SVC;
		parameters.kernel_type = (kernel == KernelType.Linear ? svm_parameter.LINEAR : svm_parameter.RBF);
		parameters.C = c;
		parameters.gamma = gamma;
		parameters.probability = 0;
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
}
