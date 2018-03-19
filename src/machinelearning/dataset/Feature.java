package machinelearning.dataset;

/**
 * A class representing a feature of an instance, i.e. a value in a row
 * other than the class value.
 *
 * @author evanc
 */
public class Feature {

	/** The name of the feature */
	private String name;
	/** The type of the feature */
	private FeatureType featureType;
	/** For nominal features, the list of possible values the feature can take */
	private String[] values;

	public Feature(String name, FeatureType featureType) {
		super();
		this.name = name;
		this.featureType = featureType;
		this.values = new String[0];
	}

	public Feature(String name, FeatureType featureType, String[] values) {
		super();
		this.name = name;
		this.featureType = featureType;
		this.values = values;
	}

	public String getName() {
		return name;
	}

	/**
	 * Gets whether or not the feature is continuous
	 *
	 * @return True iff the feature is continuous
	 */
	public boolean isContinuous() {
		return featureType == FeatureType.Continuous;
	}

	/**
	 * Gets whether or not the feature is nominal (categorical)
	 *
	 * @return True iff the feature is nominal
	 */
	public boolean isNominal() {
		return featureType == FeatureType.Nominal;
	}

	public FeatureType getFeatureType() {
		return featureType;
	}

	public String[] getValues() {
		return values;
	}
}
