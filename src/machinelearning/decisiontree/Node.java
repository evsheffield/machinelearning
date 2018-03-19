package machinelearning.decisiontree;

import java.util.ArrayList;

import machinelearning.dataset.Instance;

/**
 * An abstract class for tree nodes.
 *
 * All nodes contain a list of instances, which corresponds to the set of
 * instances that reached the node while building the tree. Although not
 * strictly necessary to store post tree creation, they can be useful in debugging
 * to see the distribution of training data in the tree.
 *
 * @author evanc
 *
 */
public abstract class Node {
	ArrayList<Instance> instances;

	/**
	 * A recursive algorithm to traverse the tree with this node at its root.
	 *
	 * Decisions at splits are made based on the passed instance.
	 *
	 * @param instance The test instance to run through the tree with this node as its root
	 * @return The predicted label/result for the given instance
	 */
	public double runTreeForInstance(Instance instance) {
		if(this instanceof LeafNode) {
			return ((LeafNode)this).getLabel();
		}
		// Choose a child node
		int childIx = ((InternalNode)this).getSplit().chooseChild(instance);
		return ((InternalNode)this).getChildren().get(childIx).runTreeForInstance(instance);
	}
}
