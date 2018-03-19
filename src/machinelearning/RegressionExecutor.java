package machinelearning;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import machinelearning.dataset.Dataset;
import machinelearning.dataset.DatasetMatrices;
import machinelearning.dataset.Feature;
import machinelearning.dataset.FeatureType;
import machinelearning.dataset.TrainingValidationMatrixSet;
import machinelearning.dataset.TrainingValidationSet;
import machinelearning.regression.LinearRegression;
import machinelearning.regression.RidgeRegression;
import machinelearning.regression.TrainingType;
import machinelearning.validation.KFoldCrossValidation;
import machinelearning.validation.ScatterPlot;

/**
 * The main class for testing out various regression models. (Assignment 2)
 * @author evanc
 *
 */
public class RegressionExecutor {

	public static void main(String[] args) {
		// --------------------------------------
		// Problem 2,3 - Linear Regression,
		// Gradient Descent and Normal Equations
		// --------------------------------------
		System.out.println("Housing Dataset");
		System.out.println("================");
		Dataset housingDataset = new Dataset(
				"data/housing.csv",
				new Feature[] {
						new Feature("CRIM", FeatureType.Continuous),
						new Feature("ZN", FeatureType.Continuous),
						new Feature("INDUS", FeatureType.Continuous),
						new Feature("CHAS", FeatureType.Continuous),
						new Feature("NOX", FeatureType.Continuous),
						new Feature("RM", FeatureType.Continuous),
						new Feature("AGE", FeatureType.Continuous),
						new Feature("DIS", FeatureType.Continuous),
						new Feature("RAD", FeatureType.Continuous),
						new Feature("TAX", FeatureType.Continuous),
						new Feature("PTRATIO", FeatureType.Continuous),
						new Feature("B", FeatureType.Continuous),
						new Feature("LSTAT", FeatureType.Continuous)
				},
				null);
		KFoldCrossValidation housingCross = new KFoldCrossValidation(10, housingDataset);
		testLinearRegression(housingCross,
				new TrainingType[] { TrainingType.GradientDescent, TrainingType.NormalEquations },
				0.0004, 0.005,
				"Housing Dataset", "Housing Dataset, Linear Regression - Training RMSE over iterations");

		System.out.println("\nYacht Dataset");
		System.out.println("================");
		Dataset yachtDataset = new Dataset(
				"data/yachtData.csv",
				new Feature[] {
						new Feature("center_of_buoyancy", FeatureType.Continuous),
						new Feature("prism_coeff", FeatureType.Continuous),
						new Feature("length_disp_ratio", FeatureType.Continuous),
						new Feature("beam_draught_ratio", FeatureType.Continuous),
						new Feature("length_beam_ratio", FeatureType.Continuous),
						new Feature("froude_num", FeatureType.Continuous)
				},
				null);
		KFoldCrossValidation yachtCross = new KFoldCrossValidation(10, yachtDataset);
		testLinearRegression(yachtCross,
				new TrainingType[] { TrainingType.GradientDescent, TrainingType.NormalEquations },
				0.001, 0.001,
				"Yacht Dataset", "Yacht Dataset, Linear Regression - Training RMSE over iterations");

		System.out.println("\nConcrete Dataset");
		System.out.println("================");
		Dataset concreteDataset = new Dataset(
				"data/concreteData.csv",
				new Feature[] {
						new Feature("cement", FeatureType.Continuous),
						new Feature("blast_furnace_slag", FeatureType.Continuous),
						new Feature("fly_ash", FeatureType.Continuous),
						new Feature("water", FeatureType.Continuous),
						new Feature("superplasticizer", FeatureType.Continuous),
						new Feature("course_aggregate", FeatureType.Continuous),
						new Feature("fine_aggregate", FeatureType.Continuous),
						new Feature("age", FeatureType.Continuous)
				},
				null);
		KFoldCrossValidation concreteCross = new KFoldCrossValidation(10, concreteDataset);
		testLinearRegression(concreteCross,
				new TrainingType[] { TrainingType.GradientDescent },
				0.0007, 0.0001,
				"Concrete Dataset", "Concrete Dataset, Linear Regression - Training RMSE over iterations");

		// -----------------------------
		// Problem 5 - Polynomial Regression
		// -----------------------------
		System.out.println("\nSinusoid Dataset");
		System.out.println("================");
		Dataset sinusoidTraining = new Dataset(
				"data/sinData_Train.csv",
				new Feature[] {new Feature("x", FeatureType.Continuous)},
				null);
		Dataset sinusoidTest = new Dataset(
				"data/sinData_Validation.csv",
				new Feature[] {new Feature("x", FeatureType.Continuous)},
				null);

		testPolynomialRegressionForTrainingValidation(sinusoidTraining, sinusoidTest);

		System.out.println("\nYacht Dataset Polynomial");
		System.out.println("================");
		testPolynomialExpansionNormalEquations(yachtCross,
				7,
				"Yacht Polynomial",
				"Yacht Dataset 10-fold cross-validation Training and Validation Mean RMSE vs Max P");

		// -----------------------------
		// Problem 7 - Ridge Regression
		// -----------------------------
		KFoldCrossValidation sinTrainingCross = new KFoldCrossValidation(10, sinusoidTraining);
		testRidgeRegression(sinTrainingCross, 5,
				"5-polynomial Sinusoid Ridge Regression",
				"Quintic Sinusoid 10-fold cross validation ridge regression mean RMSE vs lambda");
		testRidgeRegression(sinTrainingCross, 9,
				"9-polynomial Sinusoid Ridge Regression",
				"MaxP = 9 Sinusoid 10-fold cross validation ridge regression mean RMSE vs lambda");
	}

	/**
	 * Performs k-folder cross validation of regression models, using the given training types.
	 * Prints out mean RMSE information and also plots the progression of the gradient descent algorithm
	 * for one of the folds.
	 *
	 * @param cross The set of K folds to run the cross validation for
	 * @param trainingTypes The types of training to use for the linear regression model
	 * @param learningRate The learning rate to apply for gradient descent
	 * @param tolerance The tolerance threshold to apply for gradient descent
	 * @param plotTitle The title to use for the plot of gradient descent progress
	 * @param plotDescription The description to use for the plot of gradient descent progress
	 */
	private static void testLinearRegression(KFoldCrossValidation cross, TrainingType[] trainingTypes,
			double learningRate, double tolerance, String plotTitle, String plotDescription) {
		// Keep track of the root mean squared errors across folds
		DescriptiveStatistics trainingRmseStats = new DescriptiveStatistics();
		DescriptiveStatistics testRmseStats = new DescriptiveStatistics();

		// Convert the folds to matrix form for easier consumption by our regressor
		ArrayList<TrainingValidationMatrixSet> matrixFolds = new ArrayList<TrainingValidationMatrixSet>();
		for(TrainingValidationSet fold : cross.getFolds()) {
			matrixFolds.add(new TrainingValidationMatrixSet(fold, true));
		}

		// zScore normalize the data in each of the folds
		for(TrainingValidationMatrixSet fold : matrixFolds) {
			fold.zScoreNormalizeContinuousFeatures();
		}

		for(TrainingType trainingType : trainingTypes) {
			int foldNum = 0;
			System.out.println("\nTraining type: " + trainingType);
			System.out.println("---------------------------");
			for(TrainingValidationMatrixSet fold : matrixFolds) {
				System.out.println("*** Fold " + ++foldNum + " ***");

				// Create and train a regression model
				LinearRegression regressor = new LinearRegression(fold.getTrainingSet());
				switch(trainingType) {
					case NormalEquations:
						regressor.trainByNormalEquations();
						break;
					case GradientDescent:
					default:
						regressor.trainByGradientDescent(learningRate, tolerance);
						break;
				}

				// Evaluate the training and test RMSE
				double trainingRmse = regressor.getTrainingRootMeanSquaredError();
				double testRmse = regressor.getRootMeanSquaredErrors(fold.getTestSet());
				trainingRmseStats.addValue(trainingRmse);
				testRmseStats.addValue(testRmse);
				System.out.println("Training RMSE: " + trainingRmse);
				System.out.println("Test RMSE    : " + testRmse);

				// Create a plot of the last fold for gradient descent
				if(foldNum == cross.getK() && trainingType == TrainingType.GradientDescent) {
					ArrayList<ArrayList<Double>> plotData = new ArrayList<ArrayList<Double>>();
					for(int i = 0; i < regressor.getTrainingRmses().size(); i++) {
						ArrayList<Double> row =  new ArrayList<Double>();
						row.add((double)i);
						row.add(regressor.getTrainingRmses().get(i));
						plotData.add(row);
					}
					new ScatterPlot(plotTitle, plotDescription, plotData, new ArrayList<>(Arrays.asList("Training")), "Iterations", "Root Mean Squared Errors").showInFrame();
				}
			}

			// Print summary statistics for training and test results
			System.out.println("\nTraining Mean RMSE: " + trainingRmseStats.getMean());
			System.out.println("Training SD RMSE  : " + trainingRmseStats.getStandardDeviation());
			System.out.println("\nTest Mean RMSE: " + testRmseStats.getMean());
			System.out.println("Test SD RMSE  : " + testRmseStats.getStandardDeviation());
		}
	}

	/**
	 * Tests linear regression models with gradient descent using various parameters to the
	 * gradient descent algorithm.
	 *
	 * @param cross The set of K folds to run the cross validation for
	 * @param learningRates The list of learning rates to run gradient descent with
	 * @param tolerances The list of tolerance parameters to run gradient descent with
	 * @param plotTitle The title to use for the plot of gradient descent progress
	 * @param plotDescription The description to use for the plot of gradient descent progress
	 * @param randomizeWeights Whether or not to randomly initialize the weights at the beginning
	 * of gradient descent
	 */
	private static void testLinearRegressionVariations(KFoldCrossValidation cross,
			double[] learningRates, double[] tolerances, String plotTitle, String plotDescription,
			boolean randomizeWeights) {
		// Convert the folds to matrix form for easier consumption by our regressor
		ArrayList<TrainingValidationMatrixSet> matrixFolds = new ArrayList<TrainingValidationMatrixSet>();
		for(TrainingValidationSet fold : cross.getFolds()) {
			matrixFolds.add(new TrainingValidationMatrixSet(fold, true));
		}

		// zScore normalize the data in each of the folds
		for(TrainingValidationMatrixSet fold : matrixFolds) {
			fold.zScoreNormalizeContinuousFeatures();
		}

		ArrayList<ArrayList<Double>> plotData = new ArrayList<ArrayList<Double>>();
		ArrayList<String> seriesNames = new ArrayList<String>();
		for(double learningRate : learningRates) {
			for(double tolerance : tolerances) {
				seriesNames.add("LR: " + learningRate + ", Tolerance: " + tolerance);
				// Keep track of the root mean squared errors across folds
				DescriptiveStatistics trainingRmseStats = new DescriptiveStatistics();
				DescriptiveStatistics testRmseStats = new DescriptiveStatistics();

				int foldNum = 0;
				System.out.println("\nLearning Rate: " + learningRate + ", Tolerance: " + tolerance);
				System.out.println("---------------------------");
				for(TrainingValidationMatrixSet fold : matrixFolds) {
					foldNum++;

					// Create and train a regression model
					LinearRegression regressor = new LinearRegression(fold.getTrainingSet());
					regressor.trainByGradientDescent(learningRate, tolerance, randomizeWeights);

					// Evaluate the training and test RMSE
					double trainingRmse = regressor.getTrainingRootMeanSquaredError();
					double testRmse = regressor.getRootMeanSquaredErrors(fold.getTestSet());
					trainingRmseStats.addValue(trainingRmse);
					testRmseStats.addValue(testRmse);

					// Gather data to plot comparisons of the progression of gradient descent
					// with varying parameters
					if(foldNum == cross.getK()) {
						if(plotData.isEmpty()) {
							for(int i = 0; i < regressor.getTrainingRmses().size(); i++) {
								ArrayList<Double> row =  new ArrayList<Double>();
								row.add((double)i);
								row.add(regressor.getTrainingRmses().get(i));
								plotData.add(row);
							}
						} else if(regressor.getTrainingRmses().size() > plotData.size()) {
							for(int i = 0; i < regressor.getTrainingRmses().size(); i++) {
								if(i < plotData.size() ) {
									plotData.get(i).add(regressor.getTrainingRmses().get(i));
								} else {
									// Add a new row, pad with nulls as necessary to reach the row length
									ArrayList<Double> row =  new ArrayList<Double>();
									row.add((double)i);
									for(int j = 1; j < plotData.get(0).size() - 1; j++) {
										row.add(null);
									}
									row.add(regressor.getTrainingRmses().get(i));
									plotData.add(row);
								}

							}
						} else {
							for(int i = 0; i < plotData.size(); i++) {
								if(i < regressor.getTrainingRmses().size()) {
									plotData.get(i).add(regressor.getTrainingRmses().get(i));
								} else {
									plotData.get(i).add(null);
								}
							}
						}
					}
				}

				// Print summary statistics for training and test results
				System.out.println("\nTraining Mean RMSE: " + trainingRmseStats.getMean());
				System.out.println("Training SD RMSE  : " + trainingRmseStats.getStandardDeviation());
				System.out.println("\nTest Mean RMSE: " + testRmseStats.getMean());
				System.out.println("Test SD RMSE  : " + testRmseStats.getStandardDeviation());
			}
		}
		new ScatterPlot(plotTitle, plotDescription, plotData, seriesNames, "Iterations", "Root Mean Squared Errors").showInFrame();
	}

	/**
	 * Tests polynomial regression for degree 1 through degree 15 polynomials
	 * for the given training and test datasets.
	 *
	 * Does NOT use cross-validation or feature normalization.
	 *
	 * @param training The training set
	 * @param validation The validation set
	 */
	private static void testPolynomialRegressionForTrainingValidation(Dataset training, Dataset validation) {
		DatasetMatrices sinTrainingMatrices = new DatasetMatrices(training.getFeatures(), training.getInstances(), false);
		DatasetMatrices sinTestMatrices = new DatasetMatrices(validation.getFeatures(), validation.getInstances(), false);

		// Test each possible value of maxP
		ArrayList<ArrayList<Double>> plotPoints = new ArrayList<ArrayList<Double>>();
		for(int p = 1; p <= 15; p++) {
			System.out.println("\nPolynomial Level " + p);
			System.out.println("-------------------");
			DatasetMatrices trainingPoly = sinTrainingMatrices.getPolynomialExpansion(p);
			DatasetMatrices testPoly = sinTestMatrices.getPolynomialExpansion(p);

			// Train a regressor and check the training and validation RMSE
			LinearRegression regressor = new LinearRegression(trainingPoly);
			regressor.trainByNormalEquations();

			double trainingRmse = regressor.getRootMeanSquaredErrors(trainingPoly);
			double validationRmse = regressor.getRootMeanSquaredErrors(testPoly);
			ArrayList<Double> row = new ArrayList<Double>();
			row.add((double)p);
			row.add(trainingRmse);
			row.add(validationRmse);
			plotPoints.add(row);
			System.out.println("Training RMSE: " + trainingRmse);
			System.out.println("Validation RMSE: " + validationRmse);
		}
		new ScatterPlot("Sinusoid Data", "Sinusoid Dataset Training and Validation RMSE vs Max P",
				plotPoints,
				new ArrayList<>(Arrays.asList("Training", "Validation")),
				"Polynomial Degree, p", "Root Mean Squared Errors").showInFrame();
	}

	/**
	 * Performs k-fold cross-validation of the given folds at degrees of polynomial up to maxP.
	 *
	 * Prints mean RMSE values for the result of each cross-validation and plots a comparison
	 * of their performances
	 *
	 * @param cross The folds to use for k-fold cross validation
	 * @param pMax The maximum polynomial degree to go up to
	 * @param plotTitle The title of the plot of polynomial regression results
	 * @param plotDescription The description of the plot of the polynomial regression results
	 */
	private static void testPolynomialExpansionNormalEquations(KFoldCrossValidation cross, int pMax, String plotTitle, String plotDescription) {
		// Convert the folds to matrix form for easier consumption by our regressor
		ArrayList<TrainingValidationMatrixSet> matrixFolds = new ArrayList<TrainingValidationMatrixSet>();
		for(TrainingValidationSet fold : cross.getFolds()) {
			matrixFolds.add(new TrainingValidationMatrixSet(fold, true));
		}

		ArrayList<ArrayList<Double>> plotPoints = new ArrayList<ArrayList<Double>>();
		for(int p = 1; p <= pMax; p++) {
			System.out.println("\nPolynomial Level " + p);
			System.out.println("-------------------");
			DescriptiveStatistics trainingStats = new DescriptiveStatistics();
			DescriptiveStatistics testStats = new DescriptiveStatistics();
			for(TrainingValidationMatrixSet fold : matrixFolds) {
				// Get the polynomial expansion for this fold, then normalize
				// the features.
				TrainingValidationMatrixSet polyFold = fold.getPolynomialExpansion(p);
				polyFold.zScoreNormalizeContinuousFeatures();

				DatasetMatrices trainingPoly = polyFold.getTrainingSet();
				DatasetMatrices testPoly = polyFold.getTestSet();

				// Train a regression model using normal equations
				LinearRegression regressor = new LinearRegression(trainingPoly);
				regressor.trainByNormalEquations();

				double trainingRmse = regressor.getRootMeanSquaredErrors(trainingPoly);
				double testRmse = regressor.getRootMeanSquaredErrors(testPoly);
				trainingStats.addValue(trainingRmse);
				testStats.addValue(testRmse);
			}
			System.out.println("Training Mean RMSE: " + trainingStats.getMean());
			System.out.println("Training SD       : " + trainingStats.getStandardDeviation());
			System.out.println("Test Mean RMSE: " + testStats.getMean());
			System.out.println("Test SD       : " + testStats.getStandardDeviation());
			plotPoints.add(new ArrayList<>(
					Arrays.asList((double)p, trainingStats.getMean(), testStats.getMean())));
		}
		new ScatterPlot(plotTitle, plotDescription, plotPoints,
				new ArrayList<>(Arrays.asList("Training", "Validation")),
				"Polynomial Degree, p", "Root Mean Squared Errors").showInFrame();
	}

	/**
	 * Runs ridge regression for 51 values of lambda between 0 and 51 and plots the
	 * mean train and test RMSE on a graph.
	 *
	 * @param cross The KFoldCrossValidation set
	 * @param p The degree of polynomial to model
	 * @param plotTitle The title of the plot
	 * @param plotDescription The description of the plot
	 */
	private static void testRidgeRegression(KFoldCrossValidation cross, int p, String plotTitle, String plotDescription) {
		// Convert the folds to matrix form for easier consumption by our regressor,
		// but DON'T add the constant column of 1s.
		ArrayList<TrainingValidationMatrixSet> matrixFolds = new ArrayList<TrainingValidationMatrixSet>();
		for(TrainingValidationSet fold : cross.getFolds()) {
			matrixFolds.add(new TrainingValidationMatrixSet(fold, false));
		}

		ArrayList<ArrayList<Double>> plotPoints = new ArrayList<ArrayList<Double>>();

		// Test 51 values of lambda in [0, 10] by increments of 0.2
		double lambda = 0;
		while(lambda <= 10) {
			DescriptiveStatistics trainingStats = new DescriptiveStatistics();
			DescriptiveStatistics testStats = new DescriptiveStatistics();
			for(TrainingValidationMatrixSet fold : matrixFolds) {
				// Get the polynomial expansion of the fold
				TrainingValidationMatrixSet polyFold = fold.getPolynomialExpansion(p);

				DatasetMatrices trainingPoly = polyFold.getTrainingSet();
				DatasetMatrices testPoly = polyFold.getTestSet();

				// Train a ridge regression model
				RidgeRegression regressor = new RidgeRegression(trainingPoly);
				regressor.trainByNormalEquations(lambda);

				double trainingRmse = regressor.getTrainingRootMeanSquaredError();
				double testRmse = regressor.getTestRootMeanSquaredError(testPoly);
				trainingStats.addValue(trainingRmse);
				testStats.addValue(testRmse);
			}

			// Record the mean RMSE for the folds
			plotPoints.add(new ArrayList<>(
					Arrays.asList(lambda, trainingStats.getMean(), testStats.getMean())));

			lambda += 0.2;
		}
		new ScatterPlot(plotTitle, plotDescription, plotPoints,
				new ArrayList<>(Arrays.asList("Training", "Test")), "Lambda", "Root Mean Squared Errors").showInFrame();
	}

}
