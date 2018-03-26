package machinelearning.classification;

/**
 * The output of an SVM classifier on a single instance.
 * Includes the actual label and the probability predicted
 * for the positive class.
 *
 * @author evanc
 */
public class SVMResult {
	public double actualLabel;
	public double probability;
	public SVMResult(double actualLabel, double probability) {
		super();
		this.actualLabel = actualLabel;
		this.probability = probability;
	}
}