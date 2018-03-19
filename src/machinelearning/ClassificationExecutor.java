package machinelearning;

import java.awt.Font;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.knowm.xchart.CategoryChart;
import org.knowm.xchart.CategoryChartBuilder;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.style.Styler.LegendPosition;

import machinelearning.classification.DocumentClassification;
import machinelearning.classification.DocumentClassificationModelType;
import machinelearning.classification.LogisticRegression;
import machinelearning.dataset.Dataset;
import machinelearning.dataset.DocumentDataset;
import machinelearning.dataset.Feature;
import machinelearning.dataset.FeatureType;
import machinelearning.dataset.TrainingValidationMatrixSet;
import machinelearning.dataset.TrainingValidationSet;
import machinelearning.validation.APRStatistics;
import machinelearning.validation.KFoldCrossValidation;
import machinelearning.validation.ScatterPlot;

/**
 * The main class for testing out various classification models. (Assignment 3)
 * @author evanc
 *
 */
public class ClassificationExecutor {

	public static final ExecutorService TASK_EXECUTOR = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

	public static void main(String[] args) {
		// --------------------------------
		// Problem 1 - Logistic Regression
		// --------------------------------
		System.out.println("--------------------------------");
		System.out.println("Problem 1 - Logistic Regression");
		System.out.println("--------------------------------");
		Dataset spamDataset = new Dataset(
				"data/spambase.csv",
				generateContinuousFeatureList(57),
				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
		KFoldCrossValidation spamCross = new KFoldCrossValidation(10, spamDataset);

		System.out.println("Testing Spam Dataset");
		System.out.println("=====================");
		testLogisticRegression(spamCross,
				0.002,
				new double[] {0.01},
				"Spam Dataset - Logistic Regression",
				"Spam Dataset - Loss (NLL) over iterations of Gradient Descent");
//
//		Dataset breastCancerDataset = new Dataset(
//				"data/breastcancer.csv",
//				generateContinuousFeatureList(30),
//				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
//		KFoldCrossValidation bcCross = new KFoldCrossValidation(10, breastCancerDataset);
//
//		System.out.println("Testing Breast Cancer Dataset");
//		System.out.println("==============================");
//		testLogisticRegression(bcCross,
//				0.002,
//				new double[] {0.0005, 0.001, 0.01, 0.1, 1},
//				"Breast Cancer Dataset - Logistic Regression",
//				"Breast Cancer Dataset - Loss (NLL) over iterations of Gradient Descent");
//
		Dataset diabetesDataset = new Dataset(
				"data/diabetes.csv",
				generateContinuousFeatureList(8),
				new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
		KFoldCrossValidation diabetesCross = new KFoldCrossValidation(10, diabetesDataset);

		System.out.println("Testing Diabetes Dataset");
		System.out.println("=========================");
		testLogisticRegression(diabetesCross,
				0.001,
				new double[] {0.00001, 0.0001, 0.001, 0.01, 0.1, 1},
				"Diabetes Dataset - Logistic Regression",
				"Diabetes Dataset - Loss (NLL) over iterations of Gradient Descent");

		// ------------------------------------
		// Problem 2 - Document Classification
		// ------------------------------------
//		System.out.println("------------------------------------");
//		System.out.println("Problem 2 - Document Classification");
//		System.out.println("------------------------------------");
//		DocumentDataset trainingDocumentSet = new DocumentDataset("data/vocabulary.txt", "data/train.data", "data/train.label", false);
//		DocumentDataset testDocumentSet = new DocumentDataset("data/vocabulary.txt", "data/test.data", "data/test.label", false);
//		testDocumentClassification(trainingDocumentSet, testDocumentSet, false);

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
	 * Tests logistic regression for various combinations of hyperparameters on the given
	 * cross-validation set.
	 *
	 * @param cross The set of K folds to run the cross validation for
	 * @param learningRates The list of learning rates to run gradient descent with
	 * @param tolerances The list of tolerance parameters to run gradient descent with
	 * @param plotTitle The title to use for the plot of gradient descent progress
	 * @param plotDescription The description to use for the plot of gradient descent progress
	 */
	private static void testLogisticRegression(KFoldCrossValidation cross, double learningRate,
			double[] tolerances, String plotTitle, String plotDescription) {
		// Convert the folds to matrix form and z-score normalize the features
		ArrayList<TrainingValidationMatrixSet> matrixFolds = new ArrayList<TrainingValidationMatrixSet>();
		for(TrainingValidationSet fold : cross.getFolds()) {
			TrainingValidationMatrixSet newFold = new TrainingValidationMatrixSet(fold, true);
			newFold.zScoreNormalizeContinuousFeatures();
			matrixFolds.add(newFold);
		}

		ArrayList<ArrayList<Double>> plotData = new ArrayList<ArrayList<Double>>();
		ArrayList<String> seriesNames = new ArrayList<String>();

		for(double tolerance : tolerances) {
			seriesNames.add("Tolerance: " + tolerance);
			DescriptiveStatistics trainingAccuracyStats = new DescriptiveStatistics();
			DescriptiveStatistics testAccuracyStats = new DescriptiveStatistics();
			DescriptiveStatistics trainingRecallStats = new DescriptiveStatistics();
			DescriptiveStatistics testRecallStats = new DescriptiveStatistics();
			DescriptiveStatistics trainingPrecisionStats = new DescriptiveStatistics();
			DescriptiveStatistics testPrecisionStats = new DescriptiveStatistics();

			int foldNum = 0;
			System.out.println("\nLearning Rate: " + learningRate + ", Tolerance: " + tolerance);
			System.out.println("---------------------------");
			for(TrainingValidationMatrixSet fold : matrixFolds) {
				foldNum++;

				// Create a logistic regression model
				LogisticRegression logReg = new LogisticRegression(fold.getTrainingSet());
				logReg.trainByGradientDescent(learningRate, tolerance);

				// Evaluate the performance of the model
				trainingAccuracyStats.addValue(logReg.getAccuracyPercentage(fold.getTrainingSet()));
				testAccuracyStats.addValue(logReg.getAccuracyPercentage(fold.getTestSet()));
				trainingRecallStats.addValue(logReg.getRecallPercentage(fold.getTrainingSet()));
				testRecallStats.addValue(logReg.getRecallPercentage(fold.getTestSet()));
				trainingPrecisionStats.addValue(logReg.getPrecisionPercentage(fold.getTrainingSet()));
				testPrecisionStats.addValue(logReg.getPrecisionPercentage(fold.getTestSet()));

				// Gather data to plot comparisons of the progression of gradient descent
				// with varying parameters
				if(foldNum == cross.getK()) {
					if(plotData.isEmpty()) {
						for(int i = 0; i < logReg.getTrainingLosses().size(); i++) {
							ArrayList<Double> row =  new ArrayList<Double>();
							row.add((double)i);
							row.add(logReg.getTrainingLosses().get(i));
							plotData.add(row);
						}
					} else if(logReg.getTrainingLosses().size() > plotData.size()) {
						for(int i = 0; i < logReg.getTrainingLosses().size(); i++) {
							if(i < plotData.size() ) {
								plotData.get(i).add(logReg.getTrainingLosses().get(i));
							} else {
								// Add a new row, pad with nulls as necessary to reach the row length
								ArrayList<Double> row =  new ArrayList<Double>();
								row.add((double)i);
								for(int j = 1; j < plotData.get(0).size() - 1; j++) {
									row.add(null);
								}
								row.add(logReg.getTrainingLosses().get(i));
								plotData.add(row);
							}

						}
					} else {
						for(int i = 0; i < plotData.size(); i++) {
							if(i < logReg.getTrainingLosses().size()) {
								plotData.get(i).add(logReg.getTrainingLosses().get(i));
							} else {
								plotData.get(i).add(null);
							}
						}
					}
				}
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
		}

		// Display a scatter plot of the progression of gradient descent for various hyperparameters
		new ScatterPlot(plotTitle, plotDescription, plotData, seriesNames, "Iterations", "Training Loss").showInFrame();
	}

	/**
	 * Tests the performance of document classifiers trained using
	 * a multivariate and multinomial model.
	 *
	 * @param trainingData The data to train the model with
	 * @param testData The data to evaluate the model against
	 */
	private static void testDocumentClassification(DocumentDataset trainingData, DocumentDataset testData, boolean testMapEstimates) {
		int[] vocabSizes = new int[] {1000};

		ArrayList<ArrayList<Double>> plotPoints = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> precisionData = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> recallData = new ArrayList<ArrayList<Double>>();

		long operationStartTime;
		for(int vocabSize : vocabSizes) {
			System.out.println("\nVocabulary Size: " + (vocabSize == DocumentClassification.VOCAB_SIZE_ALL ? "All" : vocabSize));
			System.out.println("========================");

			// Train a multivariate model and a multinomial model
			System.out.print("Training Bernoulli (MLE)... ");
			operationStartTime = System.currentTimeMillis();
			DocumentClassification bernoulli = new DocumentClassification(trainingData, vocabSize);
			bernoulli.trainMultivariateBernoulliModel();
			System.out.println("Done! (" + ((System.currentTimeMillis() - operationStartTime) / 1000.0) + "s)");

			System.out.print("Training Multinomial (MLE)... ");
			operationStartTime = System.currentTimeMillis();
			DocumentClassification multinomial = new DocumentClassification(trainingData, vocabSize);
			multinomial.trainMultinomialEventModel();
			System.out.println("Done! (" + ((System.currentTimeMillis() - operationStartTime) / 1000.0) + "s)");

			// Evaluate the performance of each
			System.out.print("Testing Bernoulli (MLE)... ");
			operationStartTime = System.currentTimeMillis();
			APRStatistics bernStats = bernoulli.getClassifierPerformance(DocumentClassificationModelType.Bernoulli, testData);
			System.out.println("Done! (" + ((System.currentTimeMillis() - operationStartTime) / 1000.0) + "s)");
			System.out.print("Testing Multinomial (MLE)... ");
			operationStartTime = System.currentTimeMillis();
			APRStatistics multiStats = multinomial.getClassifierPerformance(DocumentClassificationModelType.Multinomial, testData);
			System.out.println("Done! (" + ((System.currentTimeMillis() - operationStartTime) / 1000.0) + "s)");
			if(testMapEstimates) {
				System.out.print("Training Bernoulli (MAP)... ");
				operationStartTime = System.currentTimeMillis();
				bernoulli.trainMultivariateBernoulliModelMap(2, 2);
				System.out.println("Done! (" + ((System.currentTimeMillis() - operationStartTime) / 1000.0) + "s)");


				ArrayList<Integer> alphas = new ArrayList<Integer>(Collections.nCopies(bernoulli.getVocabularySize(), 2));
				System.out.print("Training Multinomial (MAP)... ");
				operationStartTime = System.currentTimeMillis();
				multinomial.trainMultinomialEventModelMap(alphas);
				System.out.println("Done! (" + ((System.currentTimeMillis() - operationStartTime) / 1000.0) + "s)");

				System.out.print("Testing Bernoulli (MAP)... ");
				operationStartTime = System.currentTimeMillis();
				APRStatistics bernMapStats = bernoulli.getClassifierPerformance(DocumentClassificationModelType.Bernoulli, testData);
				System.out.println("Done! (" + ((System.currentTimeMillis() - operationStartTime) / 1000.0) + "s)");
				System.out.print("Testing Multinomial (MAP)... ");
				operationStartTime = System.currentTimeMillis();
				APRStatistics multiMapStats = multinomial.getClassifierPerformance(DocumentClassificationModelType.Multinomial, testData);
				System.out.println("Done! (" + ((System.currentTimeMillis() - operationStartTime) / 1000.0) + "s)");

				plotPoints.add(new ArrayList<Double>(
						Arrays.asList((double)bernoulli.getVocabularySize(), bernStats.getAccuracy(), bernMapStats.getAccuracy(), multiStats.getAccuracy(), multiMapStats.getAccuracy())));

				if(vocabSize == 1000) {
					precisionData.add(bernStats.getPrecisions());
					precisionData.add(bernMapStats.getPrecisions());
					precisionData.add(multiStats.getPrecisions());
					precisionData.add(multiMapStats.getPrecisions());
					recallData.add(bernStats.getRecalls());
					recallData.add(bernMapStats.getRecalls());
					recallData.add(multiStats.getRecalls());
					recallData.add(multiMapStats.getRecalls());
				}
			} else {
				plotPoints.add(new ArrayList<Double>(
						Arrays.asList((double)bernoulli.getVocabularySize(), bernStats.getAccuracy(), multiStats.getAccuracy())));

				// Use a single a vocab size to generate the precision and recall charts
				if(vocabSize == 1000) {
					precisionData.add(bernStats.getPrecisions());
					precisionData.add(multiStats.getPrecisions());
					recallData.add(bernStats.getRecalls());
					recallData.add(multiStats.getRecalls());
				}
			}
		}

		// Plot the accuracy vs vocabulary sizes for both models
		ArrayList<String> seriesNames = !testMapEstimates
				? new ArrayList<String>(Arrays.asList("Bernoulli", "Multinomial"))
				: new ArrayList<String>(Arrays.asList("Bernoulli (MLE)", "Bernoulli (MAP)", "Multinomial (MLE)", "Multinomial (MAP)"));

		new ScatterPlot("Document Classification Accuracies",
				"Document Classification Accuracy vs. Vocabulary Size",
				plotPoints,
				seriesNames,
				"Vocabulary Size",
				"Accuracy %").showInFrame();

		String[] barSeriesNames = !testMapEstimates
				? new String[] {"Bernoulli", "Multinomial"}
				: new String[] {"Bernoulli (MLE)", "Bernoulli (MAP)", "Multinomial (MLE)", "Multinomial (MAP)"};
		CategoryChart precisionChart = getGroupedBarChart("Precision percentage for each class",
				"Class Label",
				"Precision %",
				precisionData,
				new ArrayList<Integer>(Arrays.asList(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)),
				barSeriesNames);
		CategoryChart recallChart = getGroupedBarChart("Recall percentage for each class",
				"Class Label",
				"Recall %",
				recallData,
				new ArrayList<Integer>(Arrays.asList(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)),
				barSeriesNames);

		new SwingWrapper<CategoryChart>(precisionChart).displayChart();
		new SwingWrapper<CategoryChart>(recallChart).displayChart();
	}

	/**
	 * Creates a grouped bar chart with the given data
	 *
	 * @param chartTitle The title of the chart
	 * @param xAxisLabel The x axis label
	 * @param yAxisLabel The y axis label
	 * @param data List of data series to populate the chart with
	 * @param classLabels The class labels in the grouping
	 * @param seriesNames Names of the various data series
	 * @return A chart which can be rendered in a JFrame
	 */
	private static CategoryChart getGroupedBarChart(String chartTitle, String xAxisLabel, String yAxisLabel,
			ArrayList<ArrayList<Double>> data, ArrayList<Integer> classLabels, String[] seriesNames) {
		CategoryChart chart = new CategoryChartBuilder().width(1800).height(600)
				.title(chartTitle).xAxisTitle(xAxisLabel).yAxisTitle(yAxisLabel).build();
		chart.getStyler().setLegendPosition(LegendPosition.InsideNW);
		chart.getStyler().setAvailableSpaceFill(.90);
		Font font = new Font("Default", Font.PLAIN, 24);
		chart.getStyler().setAxisTickLabelsFont(font);
		chart.getStyler().setAxisTitleFont(font);
		chart.getStyler().setChartTitleFont(font);
		chart.getStyler().setLegendFont(font);
		chart.getStyler().setAvailableSpaceFill(0.75);

		int i = 0;
		for(ArrayList<Double> dataSeries : data) {
			chart.addSeries(seriesNames[i], classLabels, dataSeries);
			i++;
		}

		return chart;
	}
}
