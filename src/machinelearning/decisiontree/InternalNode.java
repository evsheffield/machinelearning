package machinelearning.decisiontree;

import java.util.ArrayList;

import machinelearning.dataset.Instance;

/**
 * An internal node in the decision tree.
 *
 * These nodes contain a split and a set of child nodes
 * @author evanc
 */
public class InternalNode extends Node {
	/** The children of the internal node */
	private ArrayList<Node> children;
	/** The split to use during tree traversal */
	private Split split;

	public InternalNode(ArrayList<Instance> instances, ArrayList<Node> children, Split split) {
		super();
		this.instances = instances;
		this.children = children;
		this.split = split;
	}

	public Split getSplit() {
		return split;
	}

	public ArrayList<Node> getChildren() {
		return children;
	}
}
