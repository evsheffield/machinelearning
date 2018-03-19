package machinelearning.dataset;

import java.util.HashMap;

/**
 * A class representing a single instance in a dataset, i.e. a single row
 * of data.
 *
 * For classification problems, the result is encoded as a double based on
 * the index into the list of possible classes.
 *
 * @author evanc
 */
public class Instance {

	/** The class or result value of the instance */
	private double instanceClass;

	/**
	 * A hash of the various features and their values.
	 *
	 * Keys correspond to feature names and should match up with the name
	 * property of the Feature class instances. Values are generic object classes
	 * for the value of the feature. These could be strings, doubles, etc.
	 */
	private HashMap<String, Object> featureValues;

	public Instance(double instanceClass, HashMap<String, Object> featureValues) {
		super();
		this.instanceClass = instanceClass;
		this.featureValues = featureValues;
	}

	public double getInstanceClass() {
		return instanceClass;
	}
	public HashMap<String, Object> getFeatureValues() {
		return featureValues;
	}

	public Object getFeatureValue(String featureKey) {
		return featureValues.get(featureKey);
	}

	public void setFeatureValue(String featureKey, Object featureValue) {
		featureValues.put(featureKey, featureValue);
	}

	@Override
	/**
	 * Clones an Instance, producing a new one with identical class and feature values but new references
	 * so that modifying the clone should not impact the original.
	 */
	public Instance clone() {
		Instance clone = new Instance(instanceClass, new HashMap<String, Object>(featureValues));
		return clone;
	}



}
