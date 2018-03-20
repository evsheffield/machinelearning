package machinelearning.validation;

import java.util.ArrayList;

/**
 * A grouping of accuracy, precision, and recall for
 * a dataset with multiple classes.
 *
 * Also stores a class confusion matrix.
 * @author evanc
 *
 */
public class APRStatistics {
	private double accuracy;
	private ArrayList<Integer> classLabels;
	private ArrayList<Double> precisions;
	private ArrayList<Double> recalls;
	private int[][] confusionMatrix;

	public APRStatistics(double accuracy, ArrayList<Integer> classLabels, ArrayList<Double> precisions,
			ArrayList<Double> recalls, int[][] confusionMatrix) {
		super();
		this.accuracy = accuracy;
		this.classLabels = classLabels;
		this.precisions = precisions;
		this.recalls = recalls;
		this.confusionMatrix = confusionMatrix;
	}

	public double getAccuracy() {
		return accuracy;
	}

	public ArrayList<Integer> getClassLabels() {
		return classLabels;
	}

	public ArrayList<Double> getPrecisions() {
		return precisions;
	}

	public ArrayList<Double> getRecalls() {
		return recalls;
	}

	public int[][] getConfusionMatrix() {
		return confusionMatrix;
	}

}
