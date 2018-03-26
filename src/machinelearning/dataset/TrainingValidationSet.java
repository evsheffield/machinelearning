package machinelearning.dataset;

import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * Contains a training and validation set from the same dataset to be used
 * for k-fold cross-validation.
 *
 * @author evanc
 *
 */
public class TrainingValidationSet {
	/** The data to be used for training */
	private ArrayList<Instance> trainingSet;
	/** The data to be used for testing built classifiers/regressors */
	private ArrayList<Instance> testSet;
	/** The set of features from the origin dataset */
	private Feature[] features;

	public TrainingValidationSet(ArrayList<Instance> trainingSet, ArrayList<Instance> testSet,
			Feature[] features) {
		super();
		this.trainingSet = trainingSet;
		this.testSet = testSet;
		this.features = features;
	}

	public ArrayList<Instance> getTrainingSet() {
		return trainingSet;
	}

	public ArrayList<Instance> getTestSet() {
		return testSet;
	}

	public Feature[] getFeatures() {
		return features;
	}

	/**
	 * Returns a further partitioning of the training set into a nested
	 * layer of training validation sets.
	 *
	 * @param m The number of folds to create from the training set
	 * @return The nested set of m folds
	 */
	public ArrayList<TrainingValidationSet> getTrainingSetFolds(int m) {
		// Partition all of the training data
		ArrayList<ArrayList<Instance>> partitions = new ArrayList<ArrayList<Instance>>();
		for(int i = 0; i < m; i++) {
			partitions.add(new ArrayList<Instance>());
		}
		for(int i = 0; i < trainingSet.size(); i++) {
			partitions.get(i % m).add(trainingSet.get(i).clone());
		}

		ArrayList<TrainingValidationSet> subsets = new ArrayList<TrainingValidationSet>();
		for(int i = 0; i < m; i++) {
			ArrayList<Instance> training = new ArrayList<Instance>();
			for(int j = 0; j < m; j++) {
				if(j != i) {
					for(Instance in : partitions.get(j)) {
						training.add(in.clone());
					}
				}
			}
			ArrayList<Instance> test = new ArrayList<Instance>();
			for(Instance in : partitions.get(i)) {
				test.add(in.clone());
			}

			subsets.add(new TrainingValidationSet(training, test, features));
		}

		return subsets;
	}

	/**
	 * Normalizes all the continuous features of the training and test datasets to
	 * their z-score.
	 *
	 * z-score of an individual value x is defined as:
	 * (x - mean) / sd
	 * Where mean and sd refer to the mean and standard deviation of the population
	 * (i.e. training set values)
	 */
	public void zScoreNormalizeContinuousFeatures() {
		for(Feature f : features) {
			if(f.isContinuous()) {
				String featureKey = f.getName();

				// Get the mean and SD of the feature value
				DescriptiveStatistics stats = new DescriptiveStatistics();
				for(Instance in : trainingSet) {
					stats.addValue((Double)in.getFeatureValue(featureKey));
				}
				double mean = stats.getMean();
				double standardDeviation = stats.getStandardDeviation();

				// Normalize the value of this feature for all instances based on the summary stats
				// of the training set
				for(Instance i : trainingSet) {
					double normalizedValue = ((double)i.getFeatureValue(featureKey) - mean) /standardDeviation;
					i.setFeatureValue(featureKey, normalizedValue);
				}
				for(Instance i : testSet) {
					double normalizedValue = ((double)i.getFeatureValue(featureKey) - mean) /standardDeviation;
					i.setFeatureValue(featureKey, normalizedValue);
				}
			}
		}
	}

	/**
	 * Normalizes all the continuous features of the training and test datasets according
	 * to the max and min value of each feature in the training data.
	 *
	 * Feature values are normalized on a scale from 0 to 1 inclusive using the formula:
	 * xnorm = (x - min(X)) / (max(X) - min(X))
	 */
	public void rescaleContinuousData() {
		for(Feature f : features) {
			if(f.isContinuous()) {
				String featureKey = f.getName();

				// Find the minimum and maximum values of the feature in all the
				// training data instances
				double max = getMaxContinuousFeatureValue(f);
				double min = getMinContinuousFeatureValue(f);

				// Normalize the value of this feature for all instances based on the max and min values
				for(Instance i : trainingSet) {
					double normalizedValue = ((double)i.getFeatureValue(featureKey) - min) / (max - min);
					i.setFeatureValue(featureKey, normalizedValue);
				}
				for(Instance i : testSet) {
					double normalizedValue = ((double)i.getFeatureValue(featureKey) - min) / (max - min);
					i.setFeatureValue(featureKey, normalizedValue);
				}
			}
		}
	}

	/**
	 * Gets the maximum value of a continuous feature from the training set.
	 *
	 * @param f The feature to get the maximum value of
	 * @return The maximum value of the feature
	 */
	private double getMaxContinuousFeatureValue(Feature f) {
		return trainingSet.stream().map(s -> (Double)s.getFeatureValue(f.getName()))
			.max((f1, f2) -> Double.compare(f1, f2)).get();
	}

	/**
	 * Gets the minimum value of a continuous feature from the training set.
	 *
	 * @param f The feature to get the minimum value of
	 * @return The minimum value of the feature
	 */
	private double getMinContinuousFeatureValue(Feature f) {
		return trainingSet.stream().map(s -> (Double)s.getFeatureValue(f.getName()))
			.min((f1, f2) -> Double.compare(f1, f2)).get();
	}

}
