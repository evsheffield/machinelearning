package machinelearning.validation;

import java.util.ArrayList;

import machinelearning.dataset.Dataset;
import machinelearning.dataset.Instance;
import machinelearning.dataset.TrainingValidationSet;

/**
 * Used for performing k-fold cross-validation of classifiers/regressors.
 *
 * Creating an instance of this class from a dataset will automatically
 * partition the data into k-folds.
 *
 * @author evanc
 *
 */
public class KFoldCrossValidation {
	private int k;
	private ArrayList<TrainingValidationSet> folds;
	private String[] classValues;

	/**
	 * Create a new k-fold cross-validation set.
	 *
	 * @param k The number of folds to create
	 * @param dataset The dataset to validate
	 */
	public KFoldCrossValidation(int k, Dataset dataset) {
		this.k = k;
		folds = new ArrayList<TrainingValidationSet>();
		// Break the dataset into k partitions
		ArrayList<ArrayList<Instance>> partitions = dataset.createKSets(k);

		// Use the k partitions to create k "folds". In each fold, one
		// partition is held back as the test set and the rest is used
		// as training data
		for(int i = 0; i < k; i++) {
			// Create a fold which holds back the ith partition
			ArrayList<Instance> training = new ArrayList<Instance>();
			for(int j = 0; j < k; j++) {
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
			TrainingValidationSet tvs = new TrainingValidationSet(training, test, dataset.getFeatures());
			folds.add(tvs);
		}
	}

	public int getK() {
		return k;
	}

	public ArrayList<TrainingValidationSet> getFolds() {
		return folds;
	}
}
