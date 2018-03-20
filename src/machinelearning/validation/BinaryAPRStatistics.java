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
