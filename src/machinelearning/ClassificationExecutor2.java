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
import machinelearning.validation.APRStatistics;
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
		System.out.println("--------------------------------");
		System.out.println("Problem 1 - Perceptron");
		System.out.println("--------------------------------");

		System.out.println("\nPerceptron Dataset");
		System.out.println("*******************\n");
		Dataset perceptronDataset = new Dataset(
				"data/perceptronData.csv",
				generateContinuousFeatureList(4),
				new ArrayList<String>(Arrays.asList(new String[] {"-1", "1"})));
		KFoldCrossValidation perceptronCross = new KFoldCrossValidation(10, perceptronDataset);

		testPerceptron(perceptronCross, new PerceptronTrainingType[] {PerceptronTrainingType.Perceptron, PerceptronTrainingType.DualLinearKernel});

		System.out.println("\nSpiral Dataset");
		System.out.println("*******************\n");
		Dataset spiralDataset = new Dataset(
				"data/twoSpirals.csv",
				generateContinuousFeatureList(2),
				new ArrayList<String>(Arrays.asList(new String[] {"-1", "1"})));
		KFoldCrossValidation spiralCross = new KFoldCrossValidation(10, spiralDataset);

		System.out.print("Finding best gamma for Gaussian kernel... ");
		double bandwidth = gridSearchBandwidth(0.04, 0.25, 0.01, spiralCross);
		System.out.println("Best bandwidth, first iteration: " + bandwidth);
		double nextBandwidth = gridSearchBandwidth(bandwidth - 0.01, bandwidth + 0.01, 0.001, spiralCross);
		System.out.println("Best bandwidth, second iteration: " + nextBandwidth);

		testPerceptron(spiralCross, new PerceptronTrainingType[] {PerceptronTrainingType.DualLinearKernel, PerceptronTrainingType.DualGaussianKernel});

		// --------------------------------------------
		// Problem 2 - Regularized Logistic Regression
		// --------------------------------------------
		System.out.println("--------------------------------------------");
		System.out.println("Problem 2 - Regularized Logistic Regression");
		System.out.println("--------------------------------------------");
		Dataset spamDataset = new Dataset(
				"data/spambase.csv",
				generateContinuousFeatureList(57),
				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
		KFoldCrossValidation spamCross = new KFoldCrossValidation(10, spamDataset);

		System.out.println("\nTesting Spam Dataset");
		System.out.println("*********************");
		testLogisticRegression(spamCross,
				0.002,
				0.001,
				new double[] {0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400},
				"Spam Dataset - Logistic Regression",
				"Spam Dataset - Mean Accuracy for Regularization coefficients (lambda)");

		Dataset breastCancerDataset = new Dataset(
				"data/breastcancer.csv",
				generateContinuousFeatureList(30),
				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
		KFoldCrossValidation bcCross = new KFoldCrossValidation(10, breastCancerDataset);

		System.out.println("\nTesting Breast Cancer Dataset");
		System.out.println("*********************************");
		testLogisticRegression(bcCross,
				0.002,
				0.01,
				new double[] {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1},
				"Breast Cancer Dataset - Logistic Regression",
				"Breast Cancer Dataset - Mean Accuracy for Regularization coefficients (lambda)");

		Dataset diabetesDataset = new Dataset(
				"data/diabetes.csv",
				generateContinuousFeatureList(8),
				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
		KFoldCrossValidation diabetesCross = new KFoldCrossValidation(10, diabetesDataset);

		System.out.println("\nTesting Diabetes Dataset");
		System.out.println("****************************");
		testLogisticRegression(diabetesCross,
				0.001,
				0.001,
				new double[] {0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500},
				"Diabetes Dataset - Logistic Regression",
				"Diabetes Dataset - Mean Accuracy for Regularization coefficients (lambda)");

		// --------------------------------------------
		// Problem 3 - SVM Model Hyper-parameters
		// --------------------------------------------
		System.out.println("--------------------------------------------");
		System.out.println("Problem 3 - SVM Model Hyper-parameters");
		System.out.println("--------------------------------------------");

		System.out.println("\nTesting Spam Dataset");
		System.out.println("*********************");
		List<XYChart> spamCharts = new ArrayList<XYChart>();
		System.out.println("-- Optimizing Accuracy --");
		spamCharts.add(testSvmGridSearch(spamCross, 5, -2, 3, -8, -5, false, "Spam ROC Curve - Accuracy Optimized"));
		System.out.println("-- Optimizing AUC --");
		spamCharts.add(testSvmGridSearch(spamCross, 5, -2, 3, -8, -5, true, "Spam ROC Curve - AUC Optimized"));
		new SwingWrapper<XYChart>(spamCharts).displayChartMatrix("Spam ROC Curves");

		System.out.println("\nTesting Breast Cancer Dataset");
		System.out.println("*********************************");
		List<XYChart> bcCharts = new ArrayList<XYChart>();
		System.out.println("-- Optimizing Accuracy --");
		bcCharts.add(testSvmGridSearch(bcCross, 5, -5, 10, -15, 5, false, "Breast Cancer ROC Curve - Accuracy Optimized"));
		System.out.println("-- Optimizing AUC --");
		bcCharts.add(testSvmGridSearch(bcCross, 5, -5, 10, -15, 5, true, "Breast Cancer ROC Curve - AUC Optimized"));
		new SwingWrapper<XYChart>(bcCharts).displayChartMatrix("Breast Cancer ROC Curves");

		System.out.println("\nTesting Diabetes Dataset");
		System.out.println("****************************");
		List<XYChart> diabetesCharts = new ArrayList<XYChart>();
		System.out.println("-- Optimizing Accuracy --");
		diabetesCharts.add(testSvmGridSearch(diabetesCross, 5, -5, 10, -15, 5, false, "Diabetes ROC Curve - Accuracy Optimized"));
		System.out.println("-- Optimizing AUC --");
		diabetesCharts.add(testSvmGridSearch(diabetesCross, 5, -5, 10, -15, 5, true, "Diabetes ROC Curve - AUC Optimized"));
		new SwingWrapper<XYChart>(diabetesCharts).displayChartMatrix("Diabetes ROC Curves");

		// --------------------------------------------
		// Problem 4 - Multiclass SVM
		// --------------------------------------------
		System.out.println("--------------------------------------------");
		System.out.println("Problem 4 - Multiclass SVM");
		System.out.println("--------------------------------------------");
		Dataset wineDataset = new Dataset(
				"data/wine.data",
				generateContinuousFeatureList(13),
				new ArrayList<String>(Arrays.asList(new String[] {"1", "2", "3"})));
		KFoldCrossValidation wineCross = new KFoldCrossValidation(10, wineDataset);
		testSvmGridSearchMulticlass(wineCross, 3, 5, -5, 10, -15, 5, false, "Wine Dataset");

		Dataset digitsTraining = new Dataset(
				"data/optdigits.tra",
				generateContinuousFeatureList(64),
				new ArrayList<String>(Arrays.asList(new String[] {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})));
		Dataset digitsTest = new Dataset(
				"data/optdigits.tes",
				generateContinuousFeatureList(64),
				new ArrayList<String>(Arrays.asList(new String[] {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})));
		TrainingValidationSet digitsTvSet = new TrainingValidationSet(
				digitsTraining.getInstances(),
				digitsTest.getInstances(),
				generateContinuousFeatureList(64));
		KFoldCrossValidation digitsCross = new KFoldCrossValidation(1, new ArrayList<TrainingValidationSet>(Arrays.asList(digitsTvSet)));
		testSvmGridSearchMulticlass(digitsCross, 10, 5, -5, 10, -15, 5, false, "Digits Dataset");

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

	/**
	 * Tests the perceptron model trained using various training types
	 * @param cross The KFoldCrossValidation set
	 * @param trainingTypes The types of training to test
	 */
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

	/**
	 * Performs a search in order to select a gamma value for the RBF kernel
	 * of the dual perceptron. Selects the one with the highest accuracy.
	 *
	 * @param min The minimum value to test
	 * @param max The maximum value to test
	 * @param incr The value to increment the gamma by for each iteration
	 * @param cross The K-fold cross validation set
	 * @return The value of gamma searched which produced the highest
	 * mean test accuracy for cross-validation
	 */
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

	    XYSeries trainingSeries = chart.addSeries("Training", xData, trainingData);
	    trainingSeries.setMarkerColor(Color.RED);
	    trainingSeries.setLineColor(Color.RED);
	    trainingSeries.setMarker(SeriesMarkers.SQUARE);

	    XYSeries testSeries = chart.addSeries("Test", xData, testData, testError);
	    testSeries.setMarkerColor(Color.BLUE);
	    testSeries.setLineColor(Color.BLUE);
	    testSeries.setMarker(SeriesMarkers.CIRCLE);

	    new SwingWrapper<XYChart>(chart).displayChart();
	}

	/**
	 * Performs a grid search hyper-parameter selection using Linear and RBF
	 * kernel based SVM models. Uses a nested cross-validation strategy where
	 * the training set of each fold is partitioned into m sub-folds and used
	 * for hyperparameter selection.
	 *
	 * @param cross The cross fold validation set to test with.
	 * @param m The number of folds to use in the inner cross-validation loop.
	 * @param cMin The minimum exponent as a power of two of hyper-parameter C
	 * to use in the grid search.
	 * e.g. -5 will mean the minimum value will be 2^-5, then 2^-4, etc.
	 * @param cMax The exponent for a power of two of the maximum value of hyperparameter C
	 * @param gammaMin The minimum exponent of hyper-parameter gamma for powers of 2.
	 * @param gammaMax The maximum exponent of hyper-parameter gamma for powers of 2.
	 * @param maximizeAuc Should hyperparameters be selected by maximizing AUC? If false,
	 * uses accuracy
	 * @param plotTitle The title to use for the generated ROC-AUC curve
	 * @return A chart containing the ROC-AUC curve for the entire k-folds
	 */
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

	/**
	 * Performs a grid search hyper-parameter selection using Linear and RBF
	 * kernel based SVM models for a multi-class dataset.
	 * Uses a nested cross-validation strategy where
	 * the training set of each fold is partitioned into m sub-folds and used
	 * for hyperparameter selection.
	 *
	 * @param cross The cross fold validation set to test with.
	 * @param nrClasses The number of classes in the multi-class problem
	 * @param m The number of folds to use in the inner cross-validation loop.
	 * @param cMin The minimum exponent as a power of two of hyper-parameter C
	 * to use in the grid search.
	 * e.g. -5 will mean the minimum value will be 2^-5, then 2^-4, etc.
	 * @param cMax The exponent for a power of two of the maximum value of hyperparameter C
	 * @param gammaMin The minimum exponent of hyper-parameter gamma for powers of 2.
	 * @param gammaMax The maximum exponent of hyper-parameter gamma for powers of 2.
	 * @param maximizeAuc Should hyperparameters be selected by maximizing AUC? If false,
	 * uses accuracy
	 * @param plotTitle The title to use for the generated ROC-AUC curve
	 * @return A matrix of ROC-AUC charts for each class across k-folds
	 */
	private static void testSvmGridSearchMulticlass(KFoldCrossValidation cross, int nrClasses, int m, int cMin,
			int cMax, int gammaMin, int gammaMax, boolean maximizeAuc, String plotTitle) {

		// LIBSVM is too chatty - override the print function to do nothing
		svm_print_interface printFunc = new svm_print_interface() {
			@Override
			public void print(String arg0) {
				// Do nothing
			}
		};
		svm.svm_set_print_string_function(printFunc);

		ArrayList<APRStatistics> linearTrainingPerformances = new ArrayList<APRStatistics>();
		ArrayList<APRStatistics> linearTestPerformances = new ArrayList<APRStatistics>();
		ArrayList<APRStatistics> rbfTrainingPerformances = new ArrayList<APRStatistics>();
		ArrayList<APRStatistics> rbfTestPerformances = new ArrayList<APRStatistics>();

		ArrayList<ArrayList<SVMResult>> allLinearResults = new ArrayList<ArrayList<SVMResult>>();
		ArrayList<ArrayList<SVMResult>> allRbfResults = new ArrayList<ArrayList<SVMResult>>();
		for(int i = 0; i < nrClasses; i++) {
			allLinearResults.add(new ArrayList<SVMResult>());
			allRbfResults.add(new ArrayList<SVMResult>());
		}

		int foldNum = 0;
		for(TrainingValidationSet fold : cross.getFolds()) {
			System.out.println("Fold " + ++foldNum);
			System.out.println("---------------");
			fold.zScoreNormalizeContinuousFeatures();

			// Prepare the data for entire fold
			TrainingValidationMatrixSet foldMatrix = new TrainingValidationMatrixSet(fold, true);

			// Break the folds down into m further folds
			ArrayList<TrainingValidationSet> subFolds = fold.getTrainingSetFolds(m);
			ArrayList<TrainingValidationMatrixSet> subFoldMatrices = new ArrayList<TrainingValidationMatrixSet>();
			for(TrainingValidationSet subFold : subFolds) {
				subFoldMatrices.add(new TrainingValidationMatrixSet(subFold, true));
			}

			ArrayList<svm_model> linearClassKernels = new ArrayList<svm_model>();
			ArrayList<svm_model> rbfClassKernels = new ArrayList<svm_model>();
			// Iterate over each of the classes and build a model for each of them
			for(int positiveClass = 0; positiveClass < nrClasses; positiveClass++) {
				System.out.println("Finding hyperparameters for positive class " + positiveClass);

				// Perform an inner cross-validation loop to select the best model hyper parameters
				// for a linear kernel
				double linearBestC = 0;
				double bestLinearPerf = 0;
				for(int cExp = cMin; cExp <= cMax; cExp++) {

					double c = Math.pow(2, cExp);

					DescriptiveStatistics accuracy = new DescriptiveStatistics();

					for(TrainingValidationMatrixSet matrixSubFold : subFoldMatrices) {
						// Get a matrix representation of the subfold
						svm_problem trainingProblem = SVM.createSvmProblem(matrixSubFold.getTrainingSet(), positiveClass);

						// Train a linear model
						svm_model linearModel = SVM.trainModel(trainingProblem, KernelType.Linear, c, 0, maximizeAuc);
						if(maximizeAuc) {
							accuracy.addValue(SVM.calculateAuc(linearModel, matrixSubFold.getTestSet(), positiveClass));
						} else {
							BinaryAPRStatistics performance = SVM.getModelPerformance(linearModel, matrixSubFold.getTestSet(), positiveClass);
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
							svm_problem trainingProblem = SVM.createSvmProblem(matrixSubFold.getTrainingSet(), positiveClass);

							// Train a linear model
							svm_model rbfModel = SVM.trainModel(trainingProblem, KernelType.RBF, c, gamma, maximizeAuc);
							if(maximizeAuc) {
								accuracy.addValue(SVM.calculateAuc(rbfModel, matrixSubFold.getTestSet(), positiveClass));
							} else {
								BinaryAPRStatistics performance = SVM.getModelPerformance(rbfModel, matrixSubFold.getTestSet(), positiveClass);
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

				// Now we have selected the best parameters. Train models using these hyperparameters and add them to our set
				System.out.println("\nLinear Kernel: C = " + linearBestC);
				System.out.println("RBF Kernel: C = " + rbfBestC);
				System.out.println("RBF Kernel: Gamma = " + bestGamma + "\n");
				svm_problem trainingProblem = SVM.createSvmProblem(foldMatrix.getTrainingSet(), positiveClass);
				svm_model linearModel = SVM.trainModel(trainingProblem, KernelType.Linear, linearBestC, 0, true);
				svm_model rbfModel = SVM.trainModel(trainingProblem, KernelType.RBF, rbfBestC, bestGamma, true);
				linearClassKernels.add(linearModel);
				rbfClassKernels.add(rbfModel);
				allLinearResults.get(positiveClass).addAll(SVM.getProbabilityPredictions(linearModel, foldMatrix.getTestSet(), positiveClass));
				allRbfResults.get(positiveClass).addAll(SVM.getProbabilityPredictions(rbfModel, foldMatrix.getTestSet(), positiveClass));
			}

			// We have now accumulated models for each class trained with the best hyper-parameters. We will now collect
			// performance results based on these models.
			linearTrainingPerformances.add(SVM.getMulticlassPerformance(linearClassKernels, foldMatrix.getTrainingSet()));
			linearTestPerformances.add(SVM.getMulticlassPerformance(linearClassKernels, foldMatrix.getTestSet()));
			rbfTrainingPerformances.add(SVM.getMulticlassPerformance(rbfClassKernels, foldMatrix.getTrainingSet()));
			rbfTestPerformances.add(SVM.getMulticlassPerformance(rbfClassKernels, foldMatrix.getTestSet()));
		}

		// Get summary statistics and ROC curves for each class
		DescriptiveStatistics trainingAccuracy = new DescriptiveStatistics();
		DescriptiveStatistics testAccuracy = new DescriptiveStatistics();
		DescriptiveStatistics rbfTrainingAccuracy = new DescriptiveStatistics();
		DescriptiveStatistics rbfTestAccuracy = new DescriptiveStatistics();

		ArrayList<DescriptiveStatistics> trainingPrecision = new ArrayList<DescriptiveStatistics>();
		ArrayList<DescriptiveStatistics> testPrecision = new ArrayList<DescriptiveStatistics>();
		ArrayList<DescriptiveStatistics> trainingRecall = new ArrayList<DescriptiveStatistics>();
		ArrayList<DescriptiveStatistics> testRecall = new ArrayList<DescriptiveStatistics>();
		ArrayList<DescriptiveStatistics> rbfTrainingPrecision = new ArrayList<DescriptiveStatistics>();
		ArrayList<DescriptiveStatistics> rbfTestPrecision = new ArrayList<DescriptiveStatistics>();
		ArrayList<DescriptiveStatistics> rbfTrainingRecall = new ArrayList<DescriptiveStatistics>();
		ArrayList<DescriptiveStatistics> rbfTestRecall = new ArrayList<DescriptiveStatistics>();
		for(int classLabel = 0; classLabel < nrClasses; classLabel++) {
			trainingPrecision.add(new DescriptiveStatistics());
			testPrecision.add(new DescriptiveStatistics());
			trainingRecall.add(new DescriptiveStatistics());
			testRecall.add(new DescriptiveStatistics());
			rbfTrainingPrecision.add(new DescriptiveStatistics());
			rbfTestPrecision.add(new DescriptiveStatistics());
			rbfTrainingRecall.add(new DescriptiveStatistics());
			rbfTestRecall.add(new DescriptiveStatistics());
		}

		for(APRStatistics stats : linearTrainingPerformances) {
			trainingAccuracy.addValue(stats.getAccuracy());
			for(int i = 0; i < nrClasses; i++) {
				trainingPrecision.get(i).addValue(stats.getPrecisions().get(i));
				trainingRecall.get(i).addValue(stats.getRecalls().get(i));
			}
		}
		for(APRStatistics stats : linearTestPerformances) {
			testAccuracy.addValue(stats.getAccuracy());
			for(int i = 0; i < nrClasses; i++) {
				testPrecision.get(i).addValue(stats.getPrecisions().get(i));
				testRecall.get(i).addValue(stats.getRecalls().get(i));
			}
		}
		for(APRStatistics stats : rbfTrainingPerformances) {
			rbfTrainingAccuracy.addValue(stats.getAccuracy());
			for(int i = 0; i < nrClasses; i++) {
				rbfTrainingPrecision.get(i).addValue(stats.getPrecisions().get(i));
				rbfTrainingRecall.get(i).addValue(stats.getRecalls().get(i));
			}
		}
		for(APRStatistics stats : rbfTestPerformances) {
			rbfTestAccuracy.addValue(stats.getAccuracy());
			for(int i = 0; i < nrClasses; i++) {
				rbfTestPrecision.get(i).addValue(stats.getPrecisions().get(i));
				rbfTestRecall.get(i).addValue(stats.getRecalls().get(i));
			}
		}

		for(int classLabel = 0; classLabel < nrClasses; classLabel++) {
			System.out.println("======= Class " + classLabel + " Results =========");
			System.out.println("\n------------ Linear Kernel ------------");
			System.out.println("Training Mean Accuracy  : " + trainingAccuracy.getMean());
			System.out.println("Training Accuracy SD    : " + trainingAccuracy.getStandardDeviation());
			System.out.println("Training Mean Recall    : " + trainingRecall.get(classLabel).getMean());
			System.out.println("Training Recall SD      : " + trainingRecall.get(classLabel).getStandardDeviation());
			System.out.println("Training Mean Precision : " + trainingPrecision.get(classLabel).getMean());
			System.out.println("Training Precision SD   : " + trainingPrecision.get(classLabel).getStandardDeviation());
			System.out.println("\nTest Mean Accuracy  : " + testAccuracy.getMean());
			System.out.println("Test Accuracy SD    : " + testAccuracy.getStandardDeviation());
			System.out.println("Test Mean Recall    : " + testRecall.get(classLabel).getMean());
			System.out.println("Test Recall SD      : " + testRecall.get(classLabel).getStandardDeviation());
			System.out.println("Test Mean Precision : " + testPrecision.get(classLabel).getMean());
			System.out.println("Test Precision SD   : " + testPrecision.get(classLabel).getStandardDeviation());

			System.out.println("\n------------ RBF Kernel ------------");
			System.out.println("Training Mean Accuracy  : " + rbfTrainingAccuracy.getMean());
			System.out.println("Training Accuracy SD    : " + rbfTrainingAccuracy.getStandardDeviation());
			System.out.println("Training Mean Recall    : " + rbfTrainingRecall.get(classLabel).getMean());
			System.out.println("Training Recall SD      : " + rbfTrainingRecall.get(classLabel).getStandardDeviation());
			System.out.println("Training Mean Precision : " + rbfTrainingPrecision.get(classLabel).getMean());
			System.out.println("Training Precision SD   : " + rbfTrainingPrecision.get(classLabel).getStandardDeviation());
			System.out.println("\nTest Mean Accuracy  : " + rbfTestAccuracy.getMean());
			System.out.println("Test Accuracy SD    : " + rbfTestAccuracy.getStandardDeviation());
			System.out.println("Test Mean Recall    : " + rbfTestRecall.get(classLabel).getMean());
			System.out.println("Test Recall SD      : " + rbfTestRecall.get(classLabel).getStandardDeviation());
			System.out.println("Test Mean Precision : " + rbfTestPrecision.get(classLabel).getMean());
			System.out.println("Test Precision SD   : " + rbfTestPrecision.get(classLabel).getStandardDeviation());
		}

		// Draw the ROC Curves for all classes
		List<XYChart> charts = new ArrayList<XYChart>();
		for(int classLabel = 0; classLabel < nrClasses; classLabel++) {
			XYChart chart = new XYChartBuilder().width(1200).height(800).title(plotTitle + " - Class " + classLabel).xAxisTitle("False Positive Rate").yAxisTitle("True Positive Rate").build();
		    chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter);
		    Font font = new Font("Default", Font.PLAIN, 24);
			chart.getStyler().setAxisTickLabelsFont(font);
			chart.getStyler().setAxisTitleFont(font);
			chart.getStyler().setChartTitleFont(font);
			chart.getStyler().setLegendFont(font);

			double[][] linearData = SVM.getRocPoints(allLinearResults.get(classLabel));
		    XYSeries linearSeries = chart.addSeries("Linear", linearData[0], linearData[1]);
		    linearSeries.setMarkerColor(Color.RED);
		    linearSeries.setMarker(SeriesMarkers.SQUARE);

		    double[][] rbfData = SVM.getRocPoints(allRbfResults.get(classLabel));
		    XYSeries rbfSeries = chart.addSeries("RBF", rbfData[0], rbfData[1]);
		    rbfSeries.setMarkerColor(Color.BLUE);
		    rbfSeries.setMarker(SeriesMarkers.SQUARE);

		    XYSeries baselineSeries = chart.addSeries("Baseline", new double[] {0, 1}, new double[] {0, 1});
		    baselineSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
		    baselineSeries.setMarkerColor(Color.DARK_GRAY);

		    charts.add(chart);
		}

		new SwingWrapper<XYChart>(charts).displayChartMatrix("Classifier ROC Curves");
	}
}
