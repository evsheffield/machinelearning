package machinelearning.validation;

/**
 * Accuracy, precision, and recall statistics for a binary
 * classification task.
 *
 * @author evanc
 */
public class BinaryAPRStatistics {
	private double accuracy;
	private double precision;
	private double recall;

	public BinaryAPRStatistics(double accuracy, double precision, double recall) {
		super();
		this.accuracy = accuracy;
		this.precision = precision;
		this.recall = recall;
	}

	/**
	 * Calculates accuracy, percentage, and recall, from classification results
	 *
	 * @param tp True positives
	 * @param tn True negatives
	 * @param fp False positives
	 * @param fn False negatives
	 */
	public BinaryAPRStatistics(int tp, int tn, int fp, int fn) {
		accuracy = (double)(tp + tn) / (tp + tn + fp + fn) * 100;
		precision = (double)tp / (tp + fp) * 100;
		recall = (double)tp / (tp + fn) * 100;
	}

	public double getAccuracy() {
		return accuracy;
	}

	public double getPrecision() {
		return precision;
	}

	public double getRecall() {
		return recall;
	}
}
