package machinelearning;

import java.awt.Color;
import java.awt.Font;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle;
import org.knowm.xchart.style.markers.SeriesMarkers;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_print_interface;
import libsvm.svm_problem;
import machinelearning.classification.KernelType;
import machinelearning.classification.LogisticRegression;
import machinelearning.classification.Perceptron;
import machinelearning.classification.PerceptronTrainingType;
import machinelearning.classification.SVM;
import machinelearning.classification.SVMResult;
import machinelearning.dataset.Dataset;
import machinelearning.dataset.Feature;
import machinelearning.dataset.FeatureType;
import machinelearning.dataset.TrainingValidationMatrixSet;
import machinelearning.dataset.TrainingValidationSet;
import machinelearning.validation.BinaryAPRStatistics;
import machinelearning.validation.KFoldCrossValidation;

/**
 * The main class for testing out various classification models
 * including perceptron, logistic regression, and SVM. (Assignment 4)
 * @author evanc
 *
 */
public class ClassificationExecutor2 {

	public static final ExecutorService TASK_EXECUTOR = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

	public static void main(String[] args) {
		// --------------------------------
		// Problem 1 - Perceptron
		// --------------------------------
//		System.out.println("--------------------------------");
//		System.out.println("Problem 1 - Perceptron");
//		System.out.println("--------------------------------");

//		System.out.println("\nPerceptron Dataset");
//		System.out.println("*******************\n");
//		Dataset perceptronDataset = new Dataset(
//				"data/perceptronData.csv",
//				generateContinuousFeatureList(4),
//				new ArrayList<String>(Arrays.asList(new String[] {"-1", "1"})));
//		KFoldCrossValidation perceptronCross = new KFoldCrossValidation(10, perceptronDataset);
//
//		testPerceptron(perceptronCross, new PerceptronTrainingType[] {PerceptronTrainingType.Perceptron, PerceptronTrainingType.DualLinearKernel});

//		System.out.println("\nSpiral Dataset");
//		System.out.println("*******************\n");
//		Dataset spiralDataset = new Dataset(
//				"data/twoSpirals.csv",
//				generateContinuousFeatureList(2),
//				new ArrayList<String>(Arrays.asList(new String[] {"-1", "1"})));
//		KFoldCrossValidation spiralCross = new KFoldCrossValidation(10, spiralDataset);

//		System.out.print("Finding best gamma for Gaussian kernel... ");
//		double bandwidth = gridSearchBandwidth(0.04, 0.25, 0.01, spiralCross);
//		System.out.println(bandwidth);

//		testPerceptron(spiralCross, new PerceptronTrainingType[] {PerceptronTrainingType.DualLinearKernel, PerceptronTrainingType.DualGaussianKernel});

		// --------------------------------------------
		// Problem 2 - Regularized Logistic Regression
		// --------------------------------------------
//		System.out.println("--------------------------------------------");
//		System.out.println("Problem 2 - Regularized Logistic Regression");
//		System.out.println("--------------------------------------------");
//		Dataset spamDataset = new Dataset(
//				"data/spambase.csv",
//				generateContinuousFeatureList(57),
//				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
//		KFoldCrossValidation spamCross = new KFoldCrossValidation(10, spamDataset);
//
//		System.out.println("\nTesting Spam Dataset");
//		System.out.println("*********************");
//		testLogisticRegression(spamCross,
//				0.002,
//				0.001,
//				new double[] {0, 1, 10, 50, 100},
//				"Spam Dataset - Logistic Regression",
//				"Spam Dataset - Mean Accuracy for Regularization coefficients (lambda)");
//
//		Dataset breastCancerDataset = new Dataset(
//				"data/breastcancer.csv",
//				generateContinuousFeatureList(30),
//				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
//		KFoldCrossValidation bcCross = new KFoldCrossValidation(10, breastCancerDataset);
//
//		System.out.println("\nTesting Breast Cancer Dataset");
//		System.out.println("*********************************");
//		testLogisticRegression(bcCross,
//				0.002,
//				0.01,
//				new double[] {0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 450, 500},
//				"Breast Cancer Dataset - Logistic Regression",
//				"Breast Cancer Dataset - Mean Accuracy for Regularization coefficients (lambda)");
//
//		Dataset diabetesDataset = new Dataset(
//				"data/diabetes.csv",
//				generateContinuousFeatureList(8),
//				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
//		KFoldCrossValidation diabetesCross = new KFoldCrossValidation(10, diabetesDataset);
//
//		System.out.println("\nTesting Diabetes Dataset");
//		System.out.println("****************************");
//		testLogisticRegression(diabetesCross,
//				0.001,
//				0.001,
//				new double[] {0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500},
//				"Diabetes Dataset - Logistic Regression",
//				"Diabetes Dataset - Mean Accuracy for Regularization coefficients (lambda)");

		// --------------------------------------------
		// Problem 3 - SVM Model Hyper-parameters
		// --------------------------------------------
//		System.out.println("--------------------------------------------");
//		System.out.println("Problem 3 - SVM Model Hyper-parameters");
//		System.out.println("--------------------------------------------");
//
//		System.out.println("\nTesting Spam Dataset");
//		System.out.println("*********************");
//		testSvmGridSearch(spamCross, 5, -5, 3, -15, 5);
//
//		System.out.println("\nTesting Breast Cancer Dataset");
//		System.out.println("*********************************");
//		new SwingWrapper<XYChart>(testSvmGridSearch(bcCross, 5, -5, 3, -5, 5, false, "Breast Cancer ROC Curve")).displayChart();
//
//		System.out.println("\nTesting Diabetes Dataset");
//		System.out.println("****************************");

		// --------------------------------------------
		// Problem 4 - Multiclass SVM
		// --------------------------------------------
		System.out.println("--------------------------------------------");
		System.out.println("Problem 4 - Multiclass SVM");
		System.out.println("--------------------------------------------");
		ArrayList<String> wineClasses = new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3"}));
		Dataset wineDataset = new Dataset(
				"data/wine.data",
				generateContinuousFeatureList(13),
				new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3"})));
		List<XYChart> wineCharts = new ArrayList<XYChart>();
		for(String wineClass : wineClasses) {
			Dataset binaryWineDataset = wineDataset.getBinaryClassDatasetFromMulticlassDataset(wineClass);
			KFoldCrossValidation wineCross = new KFoldCrossValidation(10, binaryWineDataset);
			wineCharts.add(testSvmGridSearch(wineCross, 5, -5, 10, -15, 5, false, "Wine Dataset Class " + wineClass + " ROC Curve"));
		}
		new SwingWrapper<XYChart>(wineCharts).displayChartMatrix("Wine Dataset Classifiers");

		System.out.println("Done!");
	}

	/**
	 * Generates an array of continuous features with the specified size. Each feature
	 * is given a generic name with an ID for its position in the list.
	 *
	 * @param count The number of features to include
	 * @return The array of continuous features
	 */
	private static Feature[] generateContinuousFeatureList(int count) {
		Feature[] features = new Feature[count];
		for(int i = 0; i < count; i++) {
			features[i] = new Feature("feature_" + (i + 1), FeatureType.Continuous);
		}
		return features;
	}

	private static void testPerceptron(KFoldCrossValidation cross, PerceptronTrainingType[] trainingTypes) {
		// Convert the folds to matrix form, add the constant feature,
		// and z-score normalize the features
		ArrayList<TrainingValidationMatrixSet> matrixFolds = new ArrayList<TrainingValidationMatrixSet>();
		for(TrainingValidationSet fold : cross.getFolds()) {
			TrainingValidationMatrixSet newFold = new TrainingValidationMatrixSet(fold, true);
			matrixFolds.add(newFold);
		}

		ArrayList<ArrayList<DescriptiveStatistics>> allStats = new ArrayList<ArrayList<DescriptiveStatistics>>();
		for(int i = 0; i < trainingTypes.length; i++) {
			ArrayList<DescriptiveStatistics> stats = new ArrayList<DescriptiveStatistics>();
			// Training APR and Test APR
			stats.add(new DescriptiveStatistics());
			stats.add(new DescriptiveStatistics());
			stats.add(new DescriptiveStatistics());
			stats.add(new DescriptiveStatistics());
			stats.add(new DescriptiveStatistics());
			stats.add(new DescriptiveStatistics());

			allStats.add(stats);
		}

		for(TrainingValidationMatrixSet fold : matrixFolds) {

			for(int i = 0; i < trainingTypes.length; i++) {
				PerceptronTrainingType trainingType = trainingTypes[i];
				// Create and train a perceptron
				Perceptron percy = new Perceptron(fold.getTrainingSet());
				switch(trainingType) {
					case DualLinearKernel:
						percy.trainByDualLinearKernel();
						break;
					case DualGaussianKernel:
						percy.trainByDualGaussianKernel(0.08);
						break;
					case Perceptron:
					default:
						percy.trainByPerceptronAlgorithm(1);
						break;

				}

				BinaryAPRStatistics trainingStats = percy.getPerformance(fold.getTrainingSet(), trainingType);
				BinaryAPRStatistics testStats = percy.getPerformance(fold.getTestSet(), trainingType);

				ArrayList<DescriptiveStatistics> stats = allStats.get(i);
				stats.get(0).addValue(trainingStats.getAccuracy());
				stats.get(1).addValue(trainingStats.getRecall());
				stats.get(2).addValue(trainingStats.getPrecision());
				stats.get(3).addValue(testStats.getAccuracy());
				stats.get(4).addValue(testStats.getRecall());
				stats.get(5).addValue(testStats.getPrecision());
			}
		}

		for(int i = 0; i < trainingTypes.length; i++) {
			PerceptronTrainingType trainingType = trainingTypes[i];
			ArrayList<DescriptiveStatistics> stats = allStats.get(i);

			// Print summary statistics about the model's performance
			System.out.println(trainingType);
			System.out.println("=====================");
			System.out.println("\nTraining Mean Accuracy  : " + stats.get(0).getMean());
			System.out.println("Training Accuracy SD    : " + stats.get(0).getStandardDeviation());
			System.out.println("Training Mean Recall    : " + stats.get(1).getMean());
			System.out.println("Training Recall SD      : " + stats.get(1).getStandardDeviation());
			System.out.println("Training Mean Precision : " + stats.get(2).getMean());
			System.out.println("Training Precision SD   : " + stats.get(2).getStandardDeviation());
			System.out.println("\nTest Mean Accuracy  : " + stats.get(3).getMean());
			System.out.println("Test Accuracy SD    : " + stats.get(3).getStandardDeviation());
			System.out.println("Test Mean Recall    : " + stats.get(4).getMean());
			System.out.println("Test Recall SD      : " + stats.get(4).getStandardDeviation());
			System.out.println("Test Mean Precision : " + stats.get(5).getMean());
			System.out.println("Test Precision SD   : " + stats.get(5).getStandardDeviation());
		}
	}

	private static double gridSearchBandwidth(double min, double max, double incr, KFoldCrossValidation cross) {
		// Convert the folds to matrix form, add the constant feature,
		// and z-score normalize the features
		ArrayList<TrainingValidationMatrixSet> matrixFolds = new ArrayList<TrainingValidationMatrixSet>();
		for(TrainingValidationSet fold : cross.getFolds()) {
			TrainingValidationMatrixSet newFold = new TrainingValidationMatrixSet(fold, true);
			matrixFolds.add(newFold);
		}

		double bandwidth = min;
		double bestBandwidth = min;
		double bestAccuracy = 0;
		while(bandwidth <= max) {
			System.out.println("testing " + bandwidth);
			DescriptiveStatistics testAccuracyStats = new DescriptiveStatistics();
			for(TrainingValidationMatrixSet fold : matrixFolds) {
				// Create and train a perceptron
				Perceptron percy = new Perceptron(fold.getTrainingSet());
				percy.trainByDualGaussianKernel(bandwidth);

				// Evaluate performance
				BinaryAPRStatistics testStats = percy.getPerformance(fold.getTestSet(), PerceptronTrainingType.DualGaussianKernel);
				testAccuracyStats.addValue(testStats.getAccuracy());
			}
			double meanAccuracy = testAccuracyStats.getMean();
			if(meanAccuracy > bestAccuracy) {
				bestBandwidth = bandwidth;
				bestAccuracy = meanAccuracy;
			}

			bandwidth += incr;
		}
		return bestBandwidth;
	}

	/**
	 * Tests logistic regression for various combinations of hyperparameters on the given
	 * cross-validation set.
	 *
	 * @param cross The set of K folds to run the cross validation for
	 * @param learningRates The list of learning rates to run gradient descent with
	 * @param tolerance The tolerance parameter to run gradient descent with
	 * @param lambdas The lambda parameters controlling regularization
	 * @param plotTitle The title to use for the plot of gradient descent progress
	 * @param plotDescription The description to use for the plot of gradient descent progress
	 */
	private static void testLogisticRegression(KFoldCrossValidation cross, double learningRate,
			double tolerance, double[] lambdas, String plotTitle, String plotDescription) {
		// Convert the folds to matrix form and z-score normalize the features
		ArrayList<TrainingValidationMatrixSet> matrixFolds = new ArrayList<TrainingValidationMatrixSet>();
		for(TrainingValidationSet fold : cross.getFolds()) {
			TrainingValidationMatrixSet newFold = new TrainingValidationMatrixSet(fold, true);
			newFold.zScoreNormalizeContinuousFeatures();
			matrixFolds.add(newFold);
		}

		ArrayList<String> seriesNames = new ArrayList<String>();
		ArrayList<Double> xData = new ArrayList<Double>();
		ArrayList<Double> trainingData = new ArrayList<Double>();
		ArrayList<Double> trainingError = new ArrayList<Double>();
		ArrayList<Double> testData = new ArrayList<Double>();
		ArrayList<Double> testError = new ArrayList<Double>();

		for(double lambda : lambdas) {
			xData.add(lambda);
			seriesNames.add("Tolerance: " + tolerance);
			DescriptiveStatistics trainingAccuracyStats = new DescriptiveStatistics();
			DescriptiveStatistics testAccuracyStats = new DescriptiveStatistics();
			DescriptiveStatistics trainingRecallStats = new DescriptiveStatistics();
			DescriptiveStatistics testRecallStats = new DescriptiveStatistics();
			DescriptiveStatistics trainingPrecisionStats = new DescriptiveStatistics();
			DescriptiveStatistics testPrecisionStats = new DescriptiveStatistics();

			int foldNum = 0;
			System.out.println("\nLearning Rate: " + learningRate + ", Tolerance: " + tolerance + ", Regularization: " + lambda);
			System.out.println("---------------------------");
			for(TrainingValidationMatrixSet fold : matrixFolds) {
				foldNum++;

				// Create a logistic regression model
				LogisticRegression logReg = new LogisticRegression(fold.getTrainingSet());
				logReg.trainByGradientDescent(learningRate, tolerance, lambda);

				// Evaluate the performance of the model
				trainingAccuracyStats.addValue(logReg.getAccuracyPercentage(fold.getTrainingSet()));
				testAccuracyStats.addValue(logReg.getAccuracyPercentage(fold.getTestSet()));
				trainingRecallStats.addValue(logReg.getRecallPercentage(fold.getTrainingSet()));
				testRecallStats.addValue(logReg.getRecallPercentage(fold.getTestSet()));
				trainingPrecisionStats.addValue(logReg.getPrecisionPercentage(fold.getTrainingSet()));
				testPrecisionStats.addValue(logReg.getPrecisionPercentage(fold.getTestSet()));
			}

			// Print summary statistics about the model's performance
			System.out.println("\nTraining Mean Accuracy  : " + trainingAccuracyStats.getMean());
			System.out.println("Training Accuracy SD    : " + trainingAccuracyStats.getStandardDeviation());
			System.out.println("Training Mean Recall    : " + trainingRecallStats.getMean());
			System.out.println("Training Recall SD      : " + trainingRecallStats.getStandardDeviation());
			System.out.println("Training Mean Precision : " + trainingPrecisionStats.getMean());
			System.out.println("Training Precision SD   : " + trainingPrecisionStats.getStandardDeviation());
			System.out.println("\nTest Mean Accuracy  : " + testAccuracyStats.getMean());
			System.out.println("Test Accuracy SD    : " + testAccuracyStats.getStandardDeviation());
			System.out.println("Test Mean Recall    : " + testRecallStats.getMean());
			System.out.println("Test Recall SD      : " + testRecallStats.getStandardDeviation());
			System.out.println("Test Mean Precision : " + testPrecisionStats.getMean());
			System.out.println("Test Precision SD   : " + testPrecisionStats.getStandardDeviation());

			trainingData.add(trainingAccuracyStats.getMean());
			trainingError.add(trainingAccuracyStats.getStandardDeviation());
			testData.add(testAccuracyStats.getMean());
			testError.add(testAccuracyStats.getStandardDeviation());
		}

		// Create Chart
	    XYChart chart = new XYChartBuilder().width(1200).height(800).title(plotDescription).xAxisTitle("Lambda").yAxisTitle("Mean Accuracy").build();
	    chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter);
	    Font font = new Font("Default", Font.PLAIN, 24);
		chart.getStyler().setAxisTickLabelsFont(font);
		chart.getStyler().setAxisTitleFont(font);
		chart.getStyler().setChartTitleFont(font);
		chart.getStyler().setLegendFont(font);

	    XYSeries trainingSeries = chart.addSeries("Training", xData, trainingData, trainingError);
	    trainingSeries.setMarkerColor(Color.RED);
	    trainingSeries.setLineColor(Color.RED);
	    trainingSeries.setMarker(SeriesMarkers.SQUARE);

	    XYSeries testSeries = chart.addSeries("Test", xData, testData, testError);
	    testSeries.setMarkerColor(Color.BLUE);
	    testSeries.setLineColor(Color.BLUE);
	    testSeries.setMarker(SeriesMarkers.CIRCLE);

	    new SwingWrapper<XYChart>(chart).displayChart();
	}

	private static XYChart testSvmGridSearch(KFoldCrossValidation cross, int m, int cMin, int cMax, int gammaMin, int gammaMax, boolean maximizeAuc,
			String plotTitle) {

		// LIBSVM is too chatty - override the print function to do nothing
		svm_print_interface printFunc = new svm_print_interface() {
			@Override
			public void print(String arg0) {
				// Do nothing
			}
		};
		svm.svm_set_print_string_function(printFunc);

		DescriptiveStatistics trainingAccuracy = new DescriptiveStatistics();
		DescriptiveStatistics testAccuracy = new DescriptiveStatistics();
		DescriptiveStatistics trainingPrecision = new DescriptiveStatistics();
		DescriptiveStatistics testPrecision = new DescriptiveStatistics();
		DescriptiveStatistics trainingRecall = new DescriptiveStatistics();
		DescriptiveStatistics testRecall = new DescriptiveStatistics();
		DescriptiveStatistics rbfTrainingAccuracy = new DescriptiveStatistics();
		DescriptiveStatistics rbfTestAccuracy = new DescriptiveStatistics();
		DescriptiveStatistics rbfTrainingPrecision = new DescriptiveStatistics();
		DescriptiveStatistics rbfTestPrecision = new DescriptiveStatistics();
		DescriptiveStatistics rbfTrainingRecall = new DescriptiveStatistics();
		DescriptiveStatistics rbfTestRecall = new DescriptiveStatistics();
		ArrayList<SVMResult> allLinearResults = new ArrayList<SVMResult>();
		ArrayList<SVMResult> allRbfResults = new ArrayList<SVMResult>();

		int foldNum = 0;
		for(TrainingValidationSet fold : cross.getFolds()) {
			System.out.print("Fold " + ++foldNum);
			fold.zScoreNormalizeContinuousFeatures();

			// Break the folds down into m further folds
			ArrayList<TrainingValidationSet> subFolds = fold.getTrainingSetFolds(m);
			ArrayList<TrainingValidationMatrixSet> subFoldMatrices = new ArrayList<TrainingValidationMatrixSet>();
			for(TrainingValidationSet subFold : subFolds) {
				// TODO should we add a constant feature here??
				subFoldMatrices.add(new TrainingValidationMatrixSet(subFold, true));
			}

			// Perform an inner cross-validation loop to select the best model hyper parameters
			// for a linear kernel
			double linearBestC = 0;
			double bestLinearPerf = 0;
			for(int cExp = cMin; cExp <= cMax; cExp++) {

				double c = Math.pow(2, cExp);

				DescriptiveStatistics accuracy = new DescriptiveStatistics();

				for(TrainingValidationMatrixSet matrixSubFold : subFoldMatrices) {
					// Get a matrix representation of the subfold
					// TODO can we not recalculate this too many times?
					svm_problem trainingProblem = SVM.createSvmProblem(matrixSubFold.getTrainingSet());

					// Train a linear model
					svm_model linearModel = SVM.trainModel(trainingProblem, KernelType.Linear, c, 0, maximizeAuc);
					if(maximizeAuc) {
						accuracy.addValue(SVM.calculateAuc(linearModel, matrixSubFold.getTestSet()));
					} else {
						BinaryAPRStatistics performance = SVM.getModelPerformance(linearModel, matrixSubFold.getTestSet());
						accuracy.addValue(performance.getAccuracy());
					}

				}

				if(accuracy.getMean() > bestLinearPerf) {
					linearBestC = c;
					bestLinearPerf = accuracy.getMean();
				}
				System.out.print(".");
			}

			// Perform an inner cross-validation loop to select the best model hyper parameters
			// for an RBF kernel
			double rbfBestC = 0;
			double bestGamma = 0;
			double bestRbfPerf = 0;
			for(int cExp = cMin; cExp <= cMax; cExp++) {
				for(int gammaExp = gammaMin; gammaExp <= gammaMax; gammaExp++) {
					double c = Math.pow(2, cExp);
					double gamma = Math.pow(2, gammaExp);

					DescriptiveStatistics accuracy = new DescriptiveStatistics();

					for(TrainingValidationMatrixSet matrixSubFold : subFoldMatrices) {
						// Get a matrix representation of the subfold
						// TODO can we not recalculate this too many times?
						svm_problem trainingProblem = SVM.createSvmProblem(matrixSubFold.getTrainingSet());

						// Train a linear model
						svm_model rbfModel = SVM.trainModel(trainingProblem, KernelType.RBF, c, gamma, maximizeAuc);
						if(maximizeAuc) {
							accuracy.addValue(SVM.calculateAuc(rbfModel, matrixSubFold.getTestSet()));
						} else {
							BinaryAPRStatistics performance = SVM.getModelPerformance(rbfModel, matrixSubFold.getTestSet());
							accuracy.addValue(performance.getAccuracy());
						}
					}

					if(accuracy.getMean() > bestRbfPerf) {
						rbfBestC = c;
						bestGamma = gamma;
						bestRbfPerf = accuracy.getMean();
					}
					System.out.print("*");
				}
			}

			// Now we have selected the best parameter
			System.out.println("\nLinear Kernel: C = " + linearBestC);
			System.out.println("RBF Kernel: C = " + rbfBestC);
			System.out.println("RBF Kernel: Gamma = " + bestGamma + "\n");

			TrainingValidationMatrixSet foldMatrix = new TrainingValidationMatrixSet(fold, true);

			// Build a model for this entire fold using the parameters we identified
			svm_problem trainingProblem = SVM.createSvmProblem(foldMatrix.getTrainingSet());
			svm_model linearModel = SVM.trainModel(trainingProblem, KernelType.Linear, linearBestC, 0, true);
			svm_model rbfModel = SVM.trainModel(trainingProblem, KernelType.RBF, rbfBestC, bestGamma, true);
			allLinearResults.addAll(SVM.getProbabilityPredictions(linearModel, foldMatrix.getTestSet()));
			allRbfResults.addAll(SVM.getProbabilityPredictions(rbfModel, foldMatrix.getTestSet()));

			// Evaluate training and testing performance
			BinaryAPRStatistics trainingPerformance = SVM.getModelPerformance(linearModel, foldMatrix.getTrainingSet());
			BinaryAPRStatistics testPerformance = SVM.getModelPerformance(linearModel, foldMatrix.getTestSet());
			trainingAccuracy.addValue(trainingPerformance.getAccuracy());
			testAccuracy.addValue(testPerformance.getAccuracy());
			trainingPrecision.addValue(trainingPerformance.getPrecision());
			testPrecision.addValue(testPerformance.getPrecision());
			trainingRecall.addValue(trainingPerformance.getRecall());
			testRecall.addValue(testPerformance.getRecall());

			BinaryAPRStatistics trainingRbfPerformance = SVM.getModelPerformance(rbfModel, foldMatrix.getTrainingSet());
			BinaryAPRStatistics testRbfPerformance = SVM.getModelPerformance(rbfModel, foldMatrix.getTestSet());
			rbfTrainingAccuracy.addValue(trainingRbfPerformance.getAccuracy());
			rbfTestAccuracy.addValue(testRbfPerformance.getAccuracy());
			rbfTrainingPrecision.addValue(trainingRbfPerformance.getPrecision());
			rbfTestPrecision.addValue(testRbfPerformance.getPrecision());
			rbfTrainingRecall.addValue(trainingRbfPerformance.getRecall());
			rbfTestRecall.addValue(testRbfPerformance.getRecall());
		}

		// Print summary statistics about the model's performance
		System.out.println("\n------------ Linear Kernel ------------");
		System.out.println("Training Mean Accuracy  : " + trainingAccuracy.getMean());
		System.out.println("Training Accuracy SD    : " + trainingAccuracy.getStandardDeviation());
		System.out.println("Training Mean Recall    : " + trainingRecall.getMean());
		System.out.println("Training Recall SD      : " + trainingRecall.getStandardDeviation());
		System.out.println("Training Mean Precision : " + trainingPrecision.getMean());
		System.out.println("Training Precision SD   : " + trainingPrecision.getStandardDeviation());
		System.out.println("\nTest Mean Accuracy  : " + testAccuracy.getMean());
		System.out.println("Test Accuracy SD    : " + testAccuracy.getStandardDeviation());
		System.out.println("Test Mean Recall    : " + testRecall.getMean());
		System.out.println("Test Recall SD      : " + testRecall.getStandardDeviation());
		System.out.println("Test Mean Precision : " + testPrecision.getMean());
		System.out.println("Test Precision SD   : " + testPrecision.getStandardDeviation());

		System.out.println("\n------------ RBF Kernel ------------");
		System.out.println("Training Mean Accuracy  : " + rbfTrainingAccuracy.getMean());
		System.out.println("Training Accuracy SD    : " + rbfTrainingAccuracy.getStandardDeviation());
		System.out.println("Training Mean Recall    : " + rbfTrainingRecall.getMean());
		System.out.println("Training Recall SD      : " + rbfTrainingRecall.getStandardDeviation());
		System.out.println("Training Mean Precision : " + rbfTrainingPrecision.getMean());
		System.out.println("Training Precision SD   : " + rbfTrainingPrecision.getStandardDeviation());
		System.out.println("\nTest Mean Accuracy  : " + rbfTestAccuracy.getMean());
		System.out.println("Test Accuracy SD    : " + rbfTestAccuracy.getStandardDeviation());
		System.out.println("Test Mean Recall    : " + rbfTestRecall.getMean());
		System.out.println("Test Recall SD      : " + rbfTestRecall.getStandardDeviation());
		System.out.println("Test Mean Precision : " + rbfTestPrecision.getMean());
		System.out.println("Test Precision SD   : " + rbfTestPrecision.getStandardDeviation());

		// Draw the ROC curve
	    XYChart chart = new XYChartBuilder().width(1200).height(800).title(plotTitle).xAxisTitle("False Positive Rate").yAxisTitle("True Positive Rate").build();
	    chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter);
	    Font font = new Font("Default", Font.PLAIN, 24);
		chart.getStyler().setAxisTickLabelsFont(font);
		chart.getStyler().setAxisTitleFont(font);
		chart.getStyler().setChartTitleFont(font);
		chart.getStyler().setLegendFont(font);

		double[][] linearData = SVM.getRocPoints(allLinearResults);
	    XYSeries linearSeries = chart.addSeries("Linear", linearData[0], linearData[1]);
	    linearSeries.setMarkerColor(Color.RED);
	    linearSeries.setMarker(SeriesMarkers.SQUARE);

	    double[][] rbfData = SVM.getRocPoints(allRbfResults);
	    XYSeries rbfSeries = chart.addSeries("RBF", rbfData[0], rbfData[1]);
	    rbfSeries.setMarkerColor(Color.BLUE);
	    rbfSeries.setMarker(SeriesMarkers.SQUARE);

	    XYSeries baselineSeries = chart.addSeries("Baseline", new double[] {0, 1}, new double[] {0, 1});
	    baselineSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
	    baselineSeries.setMarkerColor(Color.DARK_GRAY);

	    return chart;
	}
}
