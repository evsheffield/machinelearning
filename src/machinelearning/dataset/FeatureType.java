package machinelearning.dataset;

/**
 * An enum of the types of features that the decision tree
 * builder supports.
 *
 * Currently three types of features are supported
 * - Continuous (i.e. numeric)
 * - Nominal (i.e. categorical, taking on a value from one or more possible values)
 * - Binary (i.e. true/false, 0/1)
 *
 * @author evanc
 */
public enum FeatureType {
	Continuous, Nominal, Binary
}
