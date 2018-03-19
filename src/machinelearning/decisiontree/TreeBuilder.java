package machinelearning.decisiontree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import machinelearning.DTExecutor;
import machinelearning.dataset.Feature;
import machinelearning.dataset.Instance;
import machinelearning.exception.NotImplementedException;

/**
 * A library of static methods for building decision trees.
 *
 * The only public method is `growDecisionTree`, which can be used to grow
 * a decision tree for a set of training data.
 * @author evanc
 *
 */
public class TreeBuilder {

	/**
	 * Trains a decision tree based on the passed training data set.
	 *
	 * @param data The training data to grow the tree with
	 * @param features The set of features in the dataset
	 * @param minSplittingThreshold The minimum number of instances that a node can contain
	 * for it to be split. Otherwise, it will be created as a leaf node
	 * @param isRegression True if the tree is being used for regression, false if it is being
	 * used for classification
	 * @return The root node of the trained decision tree
	 * @throws NotImplementedException
	 */
	public static Node growDecisionTree(ArrayList<Instance> data, ArrayList<Feature> features, int minSplittingThreshold, boolean isRegression)
			throws NotImplementedException {
		if ((!isRegression && areInstancesSameClass(data)) ||
				features.size() == 0 ||
				data.size() <= minSplittingThreshold) {
			return new LeafNode(data, getNodelLabel(data, isRegression));
		}

		// Select the best feature to use
		Split bestSplit = getBestFeatureSplit(features, data, isRegression);

		// Partition the instances based on the chosen split
		ArrayList<ArrayList<Instance>> partitions = bestSplit.getPartitions(data);

		// It's possible for a split to result in only a single partition.
		// In that case, just create a leaf node.
		int emptyPartitions = 0;
		for(ArrayList<Instance> part : partitions) {
			if(part.size() == 0) {
				emptyPartitions++;
			}
		}
		if(emptyPartitions == (partitions.size() - 1)) {
			return new LeafNode(data, getNodelLabel(data, isRegression));
		}

		// Recursively build the subtree for each partition
		ArrayList<Node> children = new ArrayList<Node>();
		if(!bestSplit.getFeature().isContinuous()) {
			features.remove(bestSplit.getFeature());
		}
		for(ArrayList<Instance> part : partitions) {
			children.add(growDecisionTree(part, features, minSplittingThreshold, isRegression));
		}

		return new InternalNode(data, children, bestSplit);
	}

	/**
	 * Gets whether or not a set of instances all have the same class.
	 *
	 * Note that this is effectively the same as saying the given instances have zero entropy.
	 *
	 * @param instances The set of instances to examine
	 * @return True iff all passed instances are of the same class, i.e. have zero entropy
	 */
	private static  boolean areInstancesSameClass(ArrayList<Instance> instances) {
		if (instances.size() <= 1) {
			return true;
		}
		double firstClass = instances.get(0).getInstanceClass();
		for (int i = 1; i < instances.size(); i++) {
			if(instances.get(i).getInstanceClass() != firstClass) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Gets the class with the most instances in the given set.
	 *
	 * For regressions, returns the average result value of the given instances
	 *
	 * Used as reference: https://stackoverflow.com/questions/15725370/write-a-mode-method-in-java-to-find-the-most-frequently-occurring-element-in-an
	 * @param instances The list of instances
	 * @param isRegression Whether or not the tree is being used for regression
	 * @return The label/estimated value to apply to the leaf node
	 */
	private static double getNodelLabel(ArrayList<Instance> instances, boolean isRegression) {
		if(instances.size() == 0) {
			return -1;
		} else if(instances.size() == 1) {
			return instances.get(0).getInstanceClass();
		} else if(!isRegression) {
			HashMap<Double, Integer> classCounts = new HashMap<Double, Integer>();
			int max = 0;
			Double best = null;

			for(int i = 0; i < instances.size(); i++) {
				Double currClass = instances.get(i).getInstanceClass();
				if(classCounts.containsKey(currClass)) {
					int count = classCounts.get(currClass) + 1;
					classCounts.put(currClass, count);

					if(count > max) {
						max = count;
						best = currClass;
					}
				} else {
					classCounts.put(currClass, 1);
					if(1 > max) {
						max = 1;
						best = currClass;
					}
				}
			}
			return best;
		} else {
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for(Instance instance : instances) {
				stats.addValue(instance.getInstanceClass());
			}
			Double gm = stats.getMean();
			return gm;
		}
	}

	/**
	 * Get the entropy of a set of instances in a node.
	 *
	 * @param instances The instances in the node
	 * @return The entropy
	 */
	private static  double getEntropy(ArrayList<Instance> instances) {
		// Get the counts of each class
		HashMap<Double, Integer> classCounts = new HashMap<Double, Integer>();
		for(int i = 0; i < instances.size(); i++) {
			Double currClass = instances.get(i).getInstanceClass();
			if(classCounts.containsKey(currClass)) {
				int count = classCounts.get(currClass) + 1;
				classCounts.put(currClass, count);
			} else {
				classCounts.put(currClass, 1);
			}
		}

		// Convert to probabilities by dividing by the total instances
		HashMap<Double, Double> probabilities = new HashMap<Double, Double>();
		double count = instances.size();
		for(Double key : classCounts.keySet()) {
			probabilities.put(key, (double)classCounts.get(key) / count);
		}

		double entropy = 0;
		for(Double key : probabilities.keySet()) {
			double prob = probabilities.get(key);
			entropy -= (prob * log2(prob));
		}
		return entropy;
	}

	/**
	 * Gets the sum of squared errors for a set of instances (i.e. the instances in a node)
	 *
	 * This is only valid when the output/response variable of our data is continuous.
	 *
	 * @param instances The instances in the node
	 * @return The sum of squared errors
	 */
	private static  double getSumOfSquaredErrors(ArrayList<Instance> instances) {
		DescriptiveStatistics stats = new DescriptiveStatistics();
		for(Instance instance : instances) {
			stats.addValue(instance.getInstanceClass());
		}
		double gm = stats.getMean();

		double sumOfSquaredErrors = 0;
		for(Instance instance : instances) {
			sumOfSquaredErrors += Math.pow(instance.getInstanceClass() - gm, 2);
		}
		return sumOfSquaredErrors;
	}

	/**
	 * Gets the information gain resulting from a partition.
	 *
	 * @param root The instances that make up the root
	 * @param partitions The partitioning of the instance values to test
	 * @return The information gain
	 */
	private static  double getInformationGain(ArrayList<Instance> root, ArrayList<ArrayList<Instance>> partitions) {
		double rootInstances = root.size();

		double ig = getEntropy(root);
		for(ArrayList<Instance> node : partitions) {
			ig -= ((node.size() / rootInstances) * getEntropy(node));
		}

		return ig;
	}

	/**
	 * Gets the drop in error resulting from a partition
	 *
	 * @param root The instances that make up the root
	 * @param partitions The partitioning of the instance values to test
	 * @return The drop in error from applying the partition
	 */
	private static  double getErrorDrop(ArrayList<Instance> root, ArrayList<ArrayList<Instance>> partitions) {
		double rootSize = root.size();
		double partitionsError = getSumOfSquaredErrors(root);
		for(ArrayList<Instance> partition : partitions) {
			partitionsError -= ((partition.size() / rootSize) * getSumOfSquaredErrors(partition));
		}
		return partitionsError;
	}

	/**
	 * Gets the log base 2 of a number.
	 *
	 * @param x The number
	 * @return log2(x)
	 */
	private static double log2(double x) {
		return Math.log(x) / Math.log(2);
	}

	/**
	 * Gets the best split to use for a continuous feature on a given set of instances.
	 *
	 * @param f The continuous feature to split on
	 * @param instances The instances to create the split for
	 * @return The binary continuous split with the best possible threshold value
	 */
	private static  Split getContinuousFeatureSplit(Feature f, ArrayList<Instance> instances, boolean isRegression) {
		if(!f.isContinuous()) {
			throw new IllegalArgumentException("Can't get continuous split for a non-continuous feature");
		}

		String featureKey = f.getName();
		// Sort all the instances by the given feature
		instances.sort((i1, i2) -> ((Double) i1.getFeatureValue(featureKey)).compareTo((Double) i2.getFeatureValue(featureKey)));

		// Find all the threshold values where the instance class changes.
		ArrayList<Split> splits = new ArrayList<Split>();
		ArrayList<Double> thresholds = new ArrayList<Double>();
		for(int i = 1; i < instances.size(); i++) {
			Instance i1 = instances.get(i);
			Instance i2 = instances.get(i - 1);
			// For classification tasks, we can reduce the instances we need to consider by only considering the midpoints
			// where the class changes.
			// For regression, since there are no class values we just consider all midpoints
			if(isRegression || i1.getInstanceClass() != i2.getInstanceClass()) {
				double threshold = ((double) i1.getFeatureValue(featureKey) + (double) i2.getFeatureValue(featureKey)) / 2;
				// Try to avoid creating duplicate splits
				if(!thresholds.contains(threshold)) {
					splits.add(new BinaryContinuousSplit(f, threshold));
					thresholds.add(threshold);
				}
			}
		}

		// Try partitioning by each one of the thresholds and select the one with the maximum information gain or
		// drop in error
		return getBestSplit(splits, instances, isRegression);
	}

	/**
	 * Selects the best feature to split upon given the current set of instances in the node.
	 *
	 * @param features The features to choose from
	 * @param instances The instances to split the feature on
	 * @param isRegression True if the output of a instance is a continuous value
	 * @return The best split to use at this node
	 * @throws NotImplementedException Thrown if the feature type is not supported
	 */
	private static  Split getBestFeatureSplit(ArrayList<Feature> features, ArrayList<Instance> instances, boolean isRegression)
			throws NotImplementedException {
		ArrayList<Split> splits = new ArrayList<Split>();
		for(Feature f : features) {
			switch(f.getFeatureType()) {
				case Continuous:
					splits.add(getContinuousFeatureSplit(f, instances, isRegression));
					break;
				case Nominal:
					splits.add(new MultiwayNominalSplit(f));
					break;
				case Binary:
					splits.add(new BinarySplit(f));
					break;
				default:
					throw new NotImplementedException("That feature type is not implemented");
			}
		}

		return getBestSplit(splits, instances, isRegression);
	}

	/**
	 * Given a set of splits and a set of instances, finds the best split to use at
	 * this point in time.
	 *
	 * For classification tasks, the best split is the one that maximizes information gain.
	 *
	 * For regression, the best split is the one that maximizes the drop in error.
	 *
	 * Splits are tested using parallel processing to improve performance.
	 *
	 * @param splits The splits to choose from
	 * @param instances The nodes in the instance
	 * @param isRegression True if this is a regression, false if it is classification
	 * @return The best split to use for those instances.
	 */
	private static  Split getBestSplit(ArrayList<Split> splits, ArrayList<Instance> instances, boolean isRegression) {
		if(splits.size() == 0) {
			return null;
		}
		if(splits.size() == 1) {
			return splits.get(0);
		}

		class SplitResult {
			public double value;
			public Split split;

			public SplitResult(double value, Split split) {
				this.value = value;
				this.split = split;
			}
		}

		ArrayList<Callable<SplitResult>> tasks = new ArrayList<Callable<SplitResult>>();
		for (final Split split : splits) {
		    Callable<SplitResult> c = new Callable<SplitResult>() {
		        @Override
		        public SplitResult call() throws Exception {
		        	ArrayList<ArrayList<Instance>> parts = split.getPartitions(instances);

					double ig;
					if (isRegression) {
						ig = getErrorDrop(instances,  parts);
					} else {
						ig = getInformationGain(instances, parts);
					}

					return new SplitResult(ig, split);
		        }
		    };
		    tasks.add(c);
		}
		try {
			List<Future<SplitResult>> results = DTExecutor.TASK_EXECUTOR.invokeAll(tasks);
			SplitResult best = results.stream()
					.map(f -> {
						try {
							return f.get();
						} catch (InterruptedException | ExecutionException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
							return new SplitResult(0, null);
						}
					})
					.max((r1, r2) -> Double.compare(r1.value, r2.value)).get();
			return best.split;
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.out.println("Multi-threading error!");
			return null;
		}
	}
}

