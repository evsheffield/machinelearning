package machinelearning.decisiontree;

import java.util.ArrayList;

import machinelearning.dataset.Feature;
import machinelearning.dataset.Instance;

/**
 * An abstract class representing an n-way split in a decision tree.
 *
 * Implementing classes must define two methods which define how to partition
 * based on the split and select a child node during tree traversal.
 *
 * @author evanc
 */
public abstract class Split {
	/** The feature to split upon */
	Feature feature;

	public Feature getFeature() {
		return feature;
	}

	public void setFeature(Feature feature) {
		this.feature = feature;
	}

	/**
	 * Gets an integer corresponding to the index of the child node that should be chosen when running
	 * a tree on a test instance.
	 *
	 * @param instance The instance under test which we will test against our split to choose a child node
	 * @return The integer index of the child
	 */
	public abstract int chooseChild(Instance instance);

	/**
	 * Partitions a set of instances into lists based on the splitting criteria.
	 *
	 * @param instances The set of instances to partition
	 * @return A list of lists representing the partitioned sets of instances
	 */
	public abstract ArrayList<ArrayList<Instance>> getPartitions(ArrayList<Instance> instances);
}
