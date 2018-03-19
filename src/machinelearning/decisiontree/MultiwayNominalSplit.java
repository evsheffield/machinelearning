package machinelearning.decisiontree;

import java.util.ArrayList;

import machinelearning.dataset.Feature;
import machinelearning.dataset.Instance;

/**
 * A multiway split on a nominal feature.
 *
 * These splits are performed by matching the class of an instance
 * against the set of possible values of the feature.
 *
 * @author evanc
 */
public class MultiwayNominalSplit extends Split {

	public MultiwayNominalSplit(Feature feature) {
		super();
		this.feature = feature;
	}

	@Override
	public  int chooseChild(Instance instance) {
		// The index of the chosen child corresponds to the index of the class value
		// in the list of possible values of the feature.
		String[] featureValues = feature.getValues();
		for(int i = 0; i < featureValues.length; i++) {
			if(instance.getFeatureValue(feature.getName()).equals(featureValues[i])) {
				return i;
			}
		}

		throw new IllegalStateException(instance.getFeatureValue(feature.getName()) +
				" is not a valid value for nominal feature " + feature.getName());
	}

	@Override
	public  ArrayList<ArrayList<Instance>> getPartitions(ArrayList<Instance> instances) {
		ArrayList<ArrayList<Instance>> partitions = new ArrayList<ArrayList<Instance>>();
		for(int i = 0; i < feature.getValues().length; i++) {
			partitions.add(new ArrayList<Instance>());
		}

		for(Instance instance : instances) {
			partitions.get(chooseChild(instance)).add(instance);
		}

		return partitions;
	}
}
