package machinelearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import machinelearning.dataset.Dataset;
import machinelearning.dataset.Feature;
import machinelearning.dataset.FeatureType;
import machinelearning.dataset.Instance;
import machinelearning.dataset.TrainingValidationSet;
import machinelearning.decisiontree.Node;
import machinelearning.decisiontree.TreeBuilder;
import machinelearning.exception.NotImplementedException;
import machinelearning.validation.KFoldCrossValidation;

/**
 * The main class from which to run the building and evaluation of models.
 * @author evanc
 */
public class DTExecutor {
	public static final ExecutorService TASK_EXECUTOR = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

	public static void main(String[] args) {
		try {
			Dataset irisDataset = new Dataset(
					"data/iris.csv",
					new Feature[] {
							new Feature("sepal-length", FeatureType.Continuous),
							new Feature("sepal-width", FeatureType.Continuous),
							new Feature("petal-length", FeatureType.Continuous),
							new Feature("petal-width", FeatureType.Continuous)
					},
					new ArrayList<String>(Arrays.asList(new String[] {"Iris-setosa", "Iris-versicolor", "Iris-virginica"})));
			KFoldCrossValidation irisCross = new KFoldCrossValidation(10, irisDataset);

			System.out.println("Testing Iris dataset");
			System.out.println("=====================");
			testDecisionTree(irisCross, new double[] {0.05, 0.10, 0.15, 0.20}, false, new String[] {"Iris-setosa", "Iris-versicolor", "Iris-virginica"});

			Dataset spamDataset = new Dataset(
					"data/spambase.csv",
					new Feature[] {
							new Feature("word_freq_make", FeatureType.Continuous),
							new Feature("word_freq_address", FeatureType.Continuous),
							new Feature("word_freq_all", FeatureType.Continuous),
							new Feature("word_freq_3d", FeatureType.Continuous),
							new Feature("word_freq_our", FeatureType.Continuous),
							new Feature("word_freq_over", FeatureType.Continuous),
							new Feature("word_freq_remove", FeatureType.Continuous),
							new Feature("word_freq_internet", FeatureType.Continuous),
							new Feature("word_freq_order", FeatureType.Continuous),
							new Feature("word_freq_mail", FeatureType.Continuous),
							new Feature("word_freq_receive", FeatureType.Continuous),
							new Feature("word_freq_will", FeatureType.Continuous),
							new Feature("word_freq_people", FeatureType.Continuous),
							new Feature("word_freq_report", FeatureType.Continuous),
							new Feature("word_freq_addresses", FeatureType.Continuous),
							new Feature("word_freq_free", FeatureType.Continuous),
							new Feature("word_freq_business", FeatureType.Continuous),
							new Feature("word_freq_email", FeatureType.Continuous),
							new Feature("word_freq_you", FeatureType.Continuous),
							new Feature("word_freq_credit", FeatureType.Continuous),
							new Feature("word_freq_your", FeatureType.Continuous),
							new Feature("word_freq_font", FeatureType.Continuous),
							new Feature("word_freq_000", FeatureType.Continuous),
							new Feature("word_freq_money", FeatureType.Continuous),
							new Feature("word_freq_hp", FeatureType.Continuous),
							new Feature("word_freq_hpl", FeatureType.Continuous),
							new Feature("word_freq_george", FeatureType.Continuous),
							new Feature("word_freq_650", FeatureType.Continuous),
							new Feature("word_freq_lab", FeatureType.Continuous),
							new Feature("word_freq_labs", FeatureType.Continuous),
							new Feature("word_freq_telnet", FeatureType.Continuous),
							new Feature("word_freq_857", FeatureType.Continuous),
							new Feature("word_freq_data", FeatureType.Continuous),
							new Feature("word_freq_415", FeatureType.Continuous),
							new Feature("word_freq_85", FeatureType.Continuous),
							new Feature("word_freq_technology", FeatureType.Continuous),
							new Feature("word_freq_1999", FeatureType.Continuous),
							new Feature("word_freq_parts", FeatureType.Continuous),
							new Feature("word_freq_pm", FeatureType.Continuous),
							new Feature("word_freq_direct", FeatureType.Continuous),
							new Feature("word_freq_cs", FeatureType.Continuous),
							new Feature("word_freq_meeting", FeatureType.Continuous),
							new Feature("word_freq_original", FeatureType.Continuous),
							new Feature("word_freq_project", FeatureType.Continuous),
							new Feature("word_freq_re", FeatureType.Continuous),
							new Feature("word_freq_edu", FeatureType.Continuous),
							new Feature("word_freq_table", FeatureType.Continuous),
							new Feature("word_freq_conference", FeatureType.Continuous),
							new Feature("char_freq_;", FeatureType.Continuous),
							new Feature("char_freq_(", FeatureType.Continuous),
							new Feature("char_freq_[", FeatureType.Continuous),
							new Feature("char_freq_!", FeatureType.Continuous),
							new Feature("char_freq_$", FeatureType.Continuous),
							new Feature("char_freq_#", FeatureType.Continuous),
							new Feature("capital_run_length_average", FeatureType.Continuous),
							new Feature("capital_run_length_longest", FeatureType.Continuous),
							new Feature("capital_run_length_total", FeatureType.Continuous)
					},
					new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
			KFoldCrossValidation spamCross = new KFoldCrossValidation(10, spamDataset);

			System.out.println("Testing Spam dataset");
			System.out.println("=====================");
			testDecisionTree(spamCross, new double[] {0.05, 0.10, 0.15, 0.20, 0.25}, false, new String[] {"0", "1"});

			Dataset mushroomDataset = new Dataset(
					"data/mushroom.csv",
					new Feature[] {
							new Feature("cap-shape", FeatureType.Nominal, new String[] {"b","c","x","f","k","s"}),
							new Feature("cap-surface", FeatureType.Nominal, new String[] {"f","g","y","s"}),
							new Feature("cap-color", FeatureType.Nominal, new String[] {"n","b","c","g","r","p","u","e","w","y"}),
							new Feature("bruises?", FeatureType.Nominal, new String[] {"t","f"}),
							new Feature("odor", FeatureType.Nominal, new String[] {"a","l","c","y","f","m","n","p","s"}),
							new Feature("gill-attachment", FeatureType.Nominal, new String[] {"a","d","f","n"}),
							new Feature("gill-spacing", FeatureType.Nominal, new String[] {"c","w","d"}),
							new Feature("gill-size", FeatureType.Nominal, new String[] {"b","n"}),
							new Feature("gill-color", FeatureType.Nominal, new String[] {"k","n","b","h","g","r","o","p","u","e","w","y"}),
							new Feature("stalk-shape", FeatureType.Nominal, new String[] {"e","t"}),
//							new Feature("stalk-root", FeatureType.Nominal, new String[] {"b","c","u","e","z","r", "?"}),
							new Feature("stalk-surface-above-ring", FeatureType.Nominal, new String[] {"f","y","k","s"}),
							new Feature("stalk-surface-below-ring", FeatureType.Nominal, new String[] {"f","y","k","s"}),
							new Feature("stalk-color-above-ring", FeatureType.Nominal, new String[] {"n","b","c","g","o","p","e","w","y"}),
							new Feature("stalk-color-below-ring", FeatureType.Nominal, new String[] {"n","b","c","g","o","p","e","w","y"}),
							new Feature("veil-type", FeatureType.Nominal, new String[] {"p","u"}),
							new Feature("veil-color", FeatureType.Nominal, new String[] {"n","o","w","y"}),
							new Feature("ring-number", FeatureType.Nominal, new String[] {"n","o","t"}),
							new Feature("ring-type", FeatureType.Nominal, new String[] {"c","e","f","l","n","p","s","z"}),
							new Feature("spore-print-color", FeatureType.Nominal, new String[] {"k","n","b","h","r","o","u","w","y"}),
							new Feature("population", FeatureType.Nominal, new String[] {"a","c","n","s","v","y"}),
							new Feature("habitat", FeatureType.Nominal, new String[] {"g","l","m","p","u","w","d"})
					},
					new ArrayList<String>(Arrays.asList(new String[] {"e", "p"})));
			KFoldCrossValidation mushroomCross = new KFoldCrossValidation(10, mushroomDataset);

			System.out.println("Testing Mushroom dataset (multiway splits)");
			System.out.println("===========================================");
			testDecisionTree(mushroomCross, new double[] {0.05, 0.10, 0.15}, false, new String[] {"e", "p"});

			mushroomDataset.convertNominalFeaturesToBinary();
			KFoldCrossValidation binaryMushroomCross = new KFoldCrossValidation(10, mushroomDataset);

			System.out.println("Testing Mushroom dataset (binary splits)");
			System.out.println("===========================================");
			testDecisionTree(binaryMushroomCross, new double[] {0.05, 0.10, 0.15}, false, new String[] {"e", "p"});

			Dataset housingDataset = new Dataset(
					"data/housing.csv",
					new Feature[] {
							new Feature("CRIM", FeatureType.Continuous),
							new Feature("ZN", FeatureType.Continuous),
							new Feature("INDUS", FeatureType.Continuous),
							new Feature("CHAS", FeatureType.Binary),
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

			System.out.println("Testing Housing dataset (regression)");
			System.out.println("====================================");
			testDecisionTree(housingCross, new double[] {0.05, 0.10, 0.15, 0.20}, true, null);

			System.out.println("DONE!!!");
			TASK_EXECUTOR.shutdown();
			System.exit(0);
		} catch (NotImplementedException e) {
			System.out.println("ERROR! " + e.getMessage());
		}
	}

	/**
	 * Builds and validates decision trees for the passed list of nMinRatios.
	 *
	 * @param cross The k folds to use for cross validation
	 * @param nMinRatios The list of ratios to use as the stopping threshold for growing trees
	 * @param isRegression True if this is a regression problem, false for classification
	 * @param classValues Possible values if this is a classification problem
	 * @throws NotImplementedException
	 */
	private static void testDecisionTree(KFoldCrossValidation cross, double[] nMinRatios, boolean isRegression,
			String[] classValues) throws NotImplementedException {
		// Initialize the confusion matrix
		HashMap<String, HashMap<String, Integer>> confusionMatrix = new HashMap<String, HashMap<String, Integer>>();

		// Test with each of the given ratios for early stopping instances
		for(double nMinRatio : nMinRatios) {
			// Clear the confusion matrix
			if(!isRegression) {
				for(String c : classValues) {
					HashMap<String, Integer> row = new HashMap<String, Integer>();
					for(String d : classValues) {
						row.put(d, 0);
					}
					confusionMatrix.put(c, row);
				}
			}

			System.out.println("Testing nMin ratio: " + nMinRatio);
			System.out.println("-----------------------");

			// Grow a decision tree for each of the training-validation set folds
			ArrayList<Double> foldTrainingResults = new ArrayList<Double>();
			ArrayList<Double> foldTestResults = new ArrayList<Double>();
			int foldCount = 0;
			for(TrainingValidationSet fold : cross.getFolds()) {
				// Rescale the folds continuous data
				fold.rescaleContinuousData();
				System.out.println("*** Fold " + ++foldCount + " ***");
				int nMin = (int)Math.round(fold.getTrainingSet().size() * nMinRatio);
				int correctTrainingClassifications = 0;
				int correctTestClassifications = 0;

				// Build the tree
				Node tree = TreeBuilder.growDecisionTree(
		 				fold.getTrainingSet(),
						new ArrayList<Feature>(Arrays.asList(fold.getFeatures())),
						nMin,
						isRegression);

				// Get accuracy for training data
				if(!isRegression) {
					for(Instance trainingInstance : fold.getTrainingSet()) {
						double predictedClass = tree.runTreeForInstance(trainingInstance);
						double actualClass = trainingInstance.getInstanceClass();
						if(predictedClass == actualClass) {
							correctTrainingClassifications++;
						}
					}
					int trainingSize = fold.getTrainingSet().size();
					double percentTrainingCorrect = (double)correctTrainingClassifications / (double)trainingSize * 100;
					foldTrainingResults.add(percentTrainingCorrect);
					System.out.println("TRAINING -- Correct classifications: " + correctTrainingClassifications + "/" +
							trainingSize + ", % correct: " + percentTrainingCorrect);

					// Get accuracy for test data
					for(Instance testInstance : fold.getTestSet()) {
						double predictedClass = tree.runTreeForInstance(testInstance);
						double actualClass = testInstance.getInstanceClass();
						if(predictedClass == actualClass) {
							correctTestClassifications++;
						}
						// Store the result in the confusion matrix. The row (first index) is the predicted
						// class and the column (second index) is the actual class
						HashMap<String, Integer> row = confusionMatrix.get(classValues[(int)predictedClass]);
						row.put(classValues[(int)actualClass], row.get(classValues[(int)actualClass]) + 1);
					}
					int testSize = fold.getTestSet().size();
					double percentCorrect = (double)correctTestClassifications / (double)testSize * 100;
					foldTestResults.add(percentCorrect);
					System.out.println("TEST     -- Correct classifications: " + correctTestClassifications + "/" + testSize +
							", % correct: " + percentCorrect);
				} else {
					// Training results
					DescriptiveStatistics errors = new DescriptiveStatistics();
					for(Instance in : fold.getTrainingSet()) {
						errors.addValue(in.getInstanceClass() - tree.runTreeForInstance(in));
					}
					double sumOfSquaredErrors = errors.getSumsq();
					foldTrainingResults.add(sumOfSquaredErrors);
					System.out.println("TRAINING -- Sum of Squared Errors: " + sumOfSquaredErrors);

					// Test results
					errors.clear();
					for(Instance in : fold.getTestSet()) {
						errors.addValue(in.getInstanceClass() - tree.runTreeForInstance(in));
					}
					sumOfSquaredErrors = errors.getSumsq();
					foldTestResults.add(sumOfSquaredErrors);
					System.out.println("TEST     -- Sum of Squared Errors: " + sumOfSquaredErrors);
				}

			}

			// Get the summary statistics over all the folds
			DescriptiveStatistics resultStats = new DescriptiveStatistics();
			foldTrainingResults.forEach(r -> resultStats.addValue(r));
			System.out.println("\nMean TRAINING result: " + resultStats.getMean());
			System.out.println("Training SD         : " + resultStats.getStandardDeviation());

			resultStats.clear();
			foldTestResults.forEach(r -> resultStats.addValue(r));
			System.out.println("\nMean TEST result: " + resultStats.getMean());
			System.out.println("Test SD         : " + resultStats.getStandardDeviation());

			// Print the confusion matrix for classifications
			if(!isRegression) {
				System.out.println("Confusion matrix: ");
				System.out.print(String.format("%20s", " "));
				for(String c : classValues) {
					System.out.print(String.format("%20s", c));
				}
				System.out.println();
				for(String c : classValues) {
					System.out.print(String.format("%20s", c));
					for(String d : classValues) {
						System.out.print(String.format("%20s", confusionMatrix.get(c).get(d)));
					}
					System.out.println();
				}
				System.out.println();
			}
		}
	}

}
