package machinelearning.decisiontree;

import java.util.ArrayList;

import machinelearning.dataset.Instance;

/**
 * A leaf node in the decision tree.
 *
 * Leaf nodes have a label (or result) which is used to produce predictions
 * for the test instances that reach them.
 *
 * @author evanc
 *
 */
public class LeafNode<T> extends Node {
	private double label;

	public LeafNode(ArrayList<Instance> instances, double label) {
		super();
		this.instances = instances;
		this.label = label;
	}

	public double getLabel() {
		return label;
	}
}
