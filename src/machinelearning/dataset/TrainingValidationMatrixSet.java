package machinelearning.dataset;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * Contains a matrix-based representation of both a training
 * and validation set for use in k-fold cross validation.
 *
 * @author evanc
 */
public class TrainingValidationMatrixSet {
	DatasetMatrices trainingSet;
	DatasetMatrices testSet;
	boolean addedConstantFeature;

	public TrainingValidationMatrixSet(DatasetMatrices trainingSet, DatasetMatrices testSet,
			boolean addedConstantFeature) {
		super();
		this.trainingSet = trainingSet;
		this.testSet = testSet;
		this.addedConstantFeature = addedConstantFeature;
	}

	public TrainingValidationMatrixSet(TrainingValidationSet tvSet, boolean addConstantFeature) {
		trainingSet = new DatasetMatrices(tvSet.getFeatures(), tvSet.getTrainingSet(), addConstantFeature);
		testSet = new DatasetMatrices(tvSet.getFeatures(), tvSet.getTestSet(), addConstantFeature);
		addedConstantFeature = addConstantFeature;
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
		double[][] trainingDesignMatrix = trainingSet.getDesignMatrix();
		double[][] testDesignMatrix = testSet.getDesignMatrix();
		// For each column
		int startingColumn = addedConstantFeature ? 1 : 0;
		for(int col = startingColumn; col < trainingSet.getM(); col++) {
			// Get the training mean and SD of the column
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for(int row = 0; row < trainingSet.getN(); row++) {
				stats.addValue(trainingDesignMatrix[row][col]);
			}
			double mean = stats.getMean();
			double standardDeviation = stats.getStandardDeviation();

			// Normalize the value of this feature for all instances of training
			// and test data based on the training stats
			for(int row = 0; row < trainingSet.getN(); row++) {
				trainingDesignMatrix[row][col] = (trainingDesignMatrix[row][col] - mean) / standardDeviation;
			}
			for(int row = 0; row < testSet.getN(); row++) {
				testDesignMatrix[row][col] = (testDesignMatrix[row][col] - mean) / standardDeviation;
			}
		}
		trainingSet.setDesignMatrix(trainingDesignMatrix);
		testSet.setDesignMatrix(testDesignMatrix);
	}

	/**
	 * Returns a new TrainingValidationMatrixSet where both the training
	 * and validation set have had their feature set expanded to the
	 * indicated polynomial level.
	 *
	 * @param maxP The polynomial level to expand to (1=linear, 2=quadratic, 3=cubic, etc.)
	 * @return A new training validation set with both training and validation
	 * sets having additional polynomial features.
	 */
	public TrainingValidationMatrixSet getPolynomialExpansion(int maxP) {
		return new TrainingValidationMatrixSet(
				trainingSet.getPolynomialExpansion(maxP),
				testSet.getPolynomialExpansion(maxP),
				addedConstantFeature);
	}

	public DatasetMatrices getTrainingSet() {
		return trainingSet;
	}

	public DatasetMatrices getTestSet() {
		return testSet;
	}

}
