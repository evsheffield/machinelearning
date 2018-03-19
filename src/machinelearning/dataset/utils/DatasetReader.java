package machinelearning.dataset.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import machinelearning.dataset.Feature;
import machinelearning.dataset.Instance;

/**
 * A utility class for reading and normalizing datasets from CSV files.
 * @author evanc
 *
 */
public class DatasetReader {

	/**
	 * Reads a dataset from a CSV file into a set of instance objects.
	 *
	 * @param filepath The path to the CSV file containing the data
	 * @param features The features available in the dataset
	 * @param isClassContinuous Whether the class/result should be interpreted as continuous
	 * @return The list of rows in the dataset.
	 * @throws IOException
	 */
	public static ArrayList<Instance> readDatasetFromFile(String filepath, Feature[] features, ArrayList<String> classValues) throws IOException {
		FileReader file = new FileReader(filepath);
		BufferedReader br = new BufferedReader(file);

		String line = "";
		ArrayList<Instance> dataRows = new ArrayList<Instance>();
		while((line = br.readLine()) != null) {
			// Skip empty lines
			if(line.matches("^\\s*$")) {
				continue;
			}

			String[] values = line.split(",");

			HashMap<String, Object> featureValues = new HashMap<String, Object>();
			for (int i = 0; i < values.length - 1; i++) {
				Feature f = features[i];
				switch(f.getFeatureType()) {
					case Continuous:
						featureValues.put(f.getName(), Double.parseDouble(values[i]));
						break;
					case Binary:
						featureValues.put(f.getName(), values[i].equals("1"));
						break;
					case Nominal:
					default:
						featureValues.put(f.getName(), values[i]);
						break;
				}
			}
			Instance dataRow;
			if(classValues == null) {
				dataRow = new Instance((Double.parseDouble(values[values.length - 1])), featureValues);
			} else {
				// Normalize the class to a number which is its index into the list of class values
				double normalizedClass = classValues.indexOf(values[values.length - 1]);
				dataRow = new Instance(normalizedClass, featureValues);
			}

			dataRows.add(dataRow);
		}

		return dataRows;
	}

	public static ArrayList<Instance> readDatasetFromFile(String filepath, Feature[] features) throws IOException {
		return readDatasetFromFile(filepath, features, null);
	}
}
