package machinelearning.decisiontree;

import java.util.ArrayList;

import machinelearning.dataset.Feature;
import machinelearning.dataset.Instance;

/**
 * A binary split on a continuous attribute.
 *
 * This split operates on a threshold value. Instances with values less than
 * the threshold value will be directed to the left and instances with values
 * greater than or equal to the threshold will be directed to the right.
 *
 * @author evanc
 */
public class BinaryContinuousSplit extends Split {
	private double threshold;

	public BinaryContinuousSplit(Feature feature, double threshold) {
		super();
		this.feature = feature;
		this.threshold = threshold;
	}

	public boolean belowThreshold(Instance instance) {
		return (double) instance.getFeatureValue(feature.getName()) < threshold;
	}

	@Override
	public int chooseChild(Instance instance) {
		return belowThreshold(instance) ? 0 : 1;
	}

	@Override
	public ArrayList<ArrayList<Instance>> getPartitions(ArrayList<Instance> instances) {
		ArrayList<ArrayList<Instance>> partitions = new ArrayList<ArrayList<Instance>>();
		partitions.add(new ArrayList<Instance>());
		partitions.add(new ArrayList<Instance>());

		for(Instance instance : instances) {
			partitions.get(chooseChild(instance)).add(instance);
		}

		return partitions;
	}

	public double getThreshold() {
		return threshold;
	}
}
