package machinelearning.dataset;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import machinelearning.dataset.utils.DatasetReader;

/**
 * A class representing a data set. Includes information about its feature set as well
 * as a set of instances.
 *
 * @author evanc
 *
 */
public class Dataset {
	private ArrayList<Instance> instances;
	private Feature[] features;
	private ArrayList<String> classValues;

	public Dataset(ArrayList<Instance> instances, Feature[] features) {
		super();
		this.instances = instances;
		this.features = features;
		this.classValues = null;
	}

	/**
	 * Constructs a dataset from a CSV file and an array of feature specifications.
	 * Normalizes all continuous features in the dataset.
	 *
	 * @param datasetFile The CSV file to read instances form
	 * @param features The specification for all features in the dataset
	 * @param classValues For classification datasets, the list of possible labels the result can take
	 */
	public Dataset(String datasetFile, Feature[] features, ArrayList<String> classValues) {
		this.features = features;
		try {
			instances = DatasetReader.readDatasetFromFile(datasetFile, features, classValues);
		} catch (IOException e) {
			System.out.println("Error reading dataset from file: " + datasetFile);
			System.exit(1);
		}
	}

	/**
	 * Randomly shuffles all of the instances in the data set into k equal sized sets.
	 *
	 * @param k The number of sets to create
	 * @return The list of sets
	 */
	public ArrayList<ArrayList<Instance>> createKSets(int k) {
		ArrayList<ArrayList<Instance>> sets = new ArrayList<ArrayList<Instance>>();
		for(int i = 0; i < k; i++) {
			sets.add(new ArrayList<Instance>());
		}
		Collections.shuffle(instances);
		for(int j = 0; j < instances.size(); j++) {
			int destIx = j % k;
			sets.get(destIx).add(instances.get(j));
		}

		ArrayList<ArrayList<Instance>> subsets = new ArrayList<ArrayList<Instance>>();
		for(ArrayList<Instance> s : sets) {
			subsets.add(s);
		}

		return subsets;
	}

	/**
	 * Converts all nominal features in the dataset to a set of binary features.
	 *
	 * For example, if we had a nominal feature "element" with values {fire, water, earth, air},
	 * this would result in four new features: "element=fire", "element=water", etc.
	 */
	public void convertNominalFeaturesToBinary() {
		ArrayList<Feature> newFeatures = new ArrayList<Feature>();
		for(Feature f : features) {
			if(f.isNominal()) {
				// Create a new feature for each possible value
				for(String fVal : f.getValues()) {
					String newFeatureName = f.getName() + "=" + fVal;

					// Update all the instances and set the new feature on them
					for(Instance in : instances) {
						if(in.getFeatureValue(f.getName()).equals(fVal)) {
							in.setFeatureValue(newFeatureName, true);
						} else {
							in.setFeatureValue(newFeatureName, false);
						}
					}
					newFeatures.add(new Feature(newFeatureName, FeatureType.Binary));
				}
			} else {
				newFeatures.add(f);
			}
		}

		// Overwrite the old feature set with the new ones
		Feature[] newFeats = new Feature[newFeatures.size()];
		newFeatures.toArray(newFeats);
		features = newFeats;
	}

	public Feature[] getFeatures() {
		return features;
	}

	public ArrayList<Instance> getInstances() {
		return instances;
	}
}
