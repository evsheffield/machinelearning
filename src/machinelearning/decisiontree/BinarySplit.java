package machinelearning.decisiontree;

import java.util.ArrayList;

import machinelearning.dataset.Feature;
import machinelearning.dataset.Instance;

/**
 * A binary split on a binary attribute.
 *
 * Falsy values will be bucketed to the left and truthy values to the right.
 *
 * @author evanc
 */
public class BinarySplit extends Split {
	public BinarySplit(Feature feature) {
		super();
		this.feature = feature;
	}

	@Override
	public  int chooseChild(Instance instance) {
		// 0 -> left, 1 -> right
		return (boolean) instance.getFeatureValue(feature.getName()) ? 1 : 0;
	}

	@Override
	public  ArrayList<ArrayList<Instance>> getPartitions(ArrayList<Instance> instances) {
		ArrayList<ArrayList<Instance>> partitions = new ArrayList<ArrayList<Instance>>();
		partitions.add(new ArrayList<Instance>());
		partitions.add(new ArrayList<Instance>());

		for(Instance instance : instances) {
			partitions.get(chooseChild(instance)).add(instance);
		}

		return partitions;
	}
}
