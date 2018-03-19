package machinelearning.classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.function.IntUnaryOperator;
import java.util.stream.Collectors;

import machinelearning.ClassificationExecutor;
import machinelearning.dataset.Document;
import machinelearning.dataset.DocumentDataset;
import machinelearning.validation.APRStatistics;

/**
 * A model for classifying documents using a Naive
 * Bayes unigram model.
 *
 * @author evanc
 */
public class DocumentClassification {
	/** The documents the classifier is trained on */
	private DocumentDataset trainingDocumentDataset;
	/** Vocabulary of words to consider for the model */
	private ArrayList<String> vocabulary;
	/** The priors for each distinct class (pi_k values) */
	private double[] priors;
	/** The likelihood for each vocabulary word given a class, theta_jk */
	private double[][] likelihoods;
	/** The set of discrete class labels that documents can take */
	private ArrayList<Integer> classLabels;
	/** The size of the vocabulary */
	private int vocabularySize;
	/** The number of documents in the training data */
	private int trainingDocumentCount;


	public static final int VOCAB_SIZE_ALL = -1;

	public DocumentClassification(DocumentDataset trainingData) {
		this(trainingData, VOCAB_SIZE_ALL);
	}

	public DocumentClassification(DocumentDataset trainingData, int vocabSize) {
		trainingDocumentDataset = trainingData;

		// Extract the vocabulary from the training data
		vocabulary = new ArrayList<String>(trainingData.getWordFrequencies().keySet());
		if(vocabSize != VOCAB_SIZE_ALL && vocabSize < trainingData.getVocabularyWords().size()) {
			// Just extract the top N vocab words (N = vocabSize)
			vocabulary = new ArrayList<String>(vocabulary.subList(0, vocabSize));
		}
		vocabularySize = vocabulary.size();
		trainingDocumentCount = trainingData.getDocuments().size();

		classLabels = trainingData.getDistinctClassLabels();
		priors = new double[classLabels.size()];
		likelihoods = new double[vocabularySize][classLabels.size()];
	}

	/**
	 * Trains the document classifier using a multivariate Bernoulli model for data
	 * likelihoods and a multinomial for class priors using maximum likelihood estimates.
	 */
	public void trainMultivariateBernoulliModel() {
		calculatePriors();
		calculateMultivariateLikelihoodsAddOneSmoothing();
	}

	/**
	 * Trains the document classifier using multinomial models for data likelihoods
	 * and class priors using maximum likelihood estimates.
	 */
	public void trainMultinomialEventModel() {
		calculatePriors();
		calculateMultinomialLikelihoodsAddOneSmoothing();
	}

	/**
	 * Trains the document classifier using a multivariate Bernoulli model for data
	 * likelihoods and a multinomial for class priors using MAP estimates.
	 */
	public void trainMultivariateBernoulliModelMap(int alpha, int beta) {
		calculatePriors();
		calculateMultivariateLikelihoodsMap(alpha, beta);
	}

	/**
	 * Trains the document classifier using multinomial models for data likelihoods
	 * and class priors using MAP estimates.
	 */
	public void trainMultinomialEventModelMap(ArrayList<Integer> alphas) {
		calculatePriors();
		calculateMultinomialLikelihoodsMap(alphas);
	}

	/**
	 * Evaluates the accuracy of the model and the precision and recall for
	 * each class.
	 *
	 * @param modelType The type of model that we trained for
	 * @param testSet The test set to evaluate
	 * @return An object containing accuracy, precision, and recall information
	 */
	public APRStatistics getClassifierPerformance(DocumentClassificationModelType modelType, DocumentDataset testSet) {
		// Get a prediction for every document in the test set
		ArrayList<Document> testDocuments = testSet.getDocuments();

		ArrayList<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>();
		for(final Document testDocument : testDocuments) {
			Callable<Integer> c = new Callable<Integer>() {
				@Override
				public Integer call() throws Exception {
					switch(modelType) {
						case Multinomial:
							return predictDocumentLabelMultinomial(testDocument);
						case Bernoulli:
						default:
							return predictDocumentLabelBernoulli(testDocument);
					}
				}
			};
			tasks.add(c);
		}

		List<Future<Integer>> results;
		try {
			results = ClassificationExecutor.TASK_EXECUTOR.invokeAll(tasks);
		} catch (InterruptedException e1) {
			System.out.println("multithreading error!");
			return null;
		}
		ArrayList<Integer> predictions = results.stream()
				.map(f -> {
					try {
						return f.get();
					} catch (InterruptedException | ExecutionException e) {
						e.printStackTrace();
						return -1;
					}
				})
				.collect(Collectors.toCollection(ArrayList::new));

		// Initialize a confusion matrix
		int[][] confusionMatrix = new int[classLabels.size()][classLabels.size()];
		for(int row = 0; row < classLabels.size(); row++)
			for(int col = 0; col < classLabels.size(); col++)
				confusionMatrix[row][col] = 0;

		// Calculate accuracy
		int correctPredictions = 0;
		for(int i = 0; i < testDocuments.size(); i++) {
			int prediction = predictions.get(i);
			int actual = testDocuments.get(i).getClassLabel();
			confusionMatrix[prediction - 1][actual - 1]++;
			if(actual == prediction) {
				correctPredictions++;
			}
		}
		double accuracy = (double)correctPredictions / (double)testDocuments.size() * 100;

		// Calculate precision and recall for each class
		ArrayList<Double> precisions = new ArrayList<Double>();
		ArrayList<Double> recalls = new ArrayList<Double>();
		for(int classLabel : classLabels) {
			int tp = 0, fp = 0, fn = 0;
			for(int i = 0; i < testDocuments.size(); i++) {
				int label = testDocuments.get(i).getClassLabel();
				int prediction = predictions.get(i);

				if(label == classLabel && prediction == classLabel) {
					tp++;
				} else if (prediction == classLabel && label != classLabel) {
					fp++;
				} else if (label == classLabel && prediction != classLabel) {
					fn++;
				}
			}
			precisions.add((double)tp / (tp + fp) * 100);
			recalls.add((double)tp / (tp + fn) * 100);
		}

		return new APRStatistics(accuracy, classLabels, precisions, recalls, confusionMatrix);
	}

	/**
	 * Predicts the label of an unseen document.
	 *
	 * Precondition is that the classifier has been trained with a multivariate
	 * Bernoulli model.
	 *
	 * @param testDocument The document to evaluate
	 * @return The predicted class label
	 */
	public int predictDocumentLabelBernoulli(Document testDocument) {
		// Try evaluating each of the classes and choose the one with the higher
		// estimate
		int bestLabel = -1;
		double bestEstimate = Double.NEGATIVE_INFINITY;
		Map<String, Integer> wordFrequencies = testDocument.getWordFrequencies();
		for(int k = 0; k < classLabels.size(); k++) {
			int classLabel = classLabels.get(k);
			double estimate = Math.log(priors[k]);
			// Iterate over each word in the vocabulary. If the word is present in the
			// test document, multiply by the theta for it. If it is not present, multiply
			// by 1 - theta
			for(int j = 0; j < vocabularySize; j++) {
				String word = vocabulary.get(j);
				estimate += (wordFrequencies.containsKey(word)
						? Math.log(likelihoods[j][k])
						: Math.log(1 - likelihoods[j][k]));
			}
			if(estimate > bestEstimate) {
				bestLabel = classLabel;
				bestEstimate = estimate;
			}
		}
		return bestLabel;
	}

	/**
	 * Predicts the label of an unseen document.
	 *
	 * Precondition is that the classifier has been trained with a multinomial
	 * event model.
	 *
	 * @param testDocument The document to evaluate
	 * @return The predicted class label
	 */
	public int predictDocumentLabelMultinomial(Document testDocument) {
		// Try evaluating each of the classes and choose the one with the higher
		// estimate
		int bestLabel = -1;
		double bestEstimate = Double.NEGATIVE_INFINITY;
		Map<String, Integer> wordFrequencies = testDocument.getWordFrequencies();
		Map<String, Integer> relevantWordFrequencies = new HashMap<String, Integer>();

		// Remove all words not in the vocabulary
		for(String word : wordFrequencies.keySet()) {
			if(vocabulary.contains(word)) {
				relevantWordFrequencies.put(word, wordFrequencies.get(word));
			}
		}

		for(int k = 0; k < classLabels.size(); k++) {
			int classLabel = classLabels.get(k);
			double estimate = Math.log(priors[k]);
			// Iterate over all the words in the test document. Multiply the thetas
			// raised to the power of word frequency
			for(Entry<String, Integer> entry : relevantWordFrequencies.entrySet()) {
				// Find the index (j value) for the word
				int j = vocabulary.indexOf(entry.getKey());
				estimate += (Math.log(likelihoods[j][k]) * entry.getValue());
			}
			if(estimate > bestEstimate) {
				bestLabel = classLabel;
				bestEstimate = estimate;
			}
		}
		return bestLabel;
	}

	/**
	 * Calculates the priors (pi values) for each class from
	 * our training data.
	 *
	 * For a given class, k, pi_k is defined as the number of
	 * training documents with class k divided by the total
	 * number of training documents.
	 */
	private void calculatePriors() {
		for(int k = 0; k < classLabels.size(); k++) {
			int classLabel = classLabels.get(k);
			long classInstanceCount = trainingDocumentDataset.getDocuments().stream()
					.filter(d -> d.getClassLabel() == classLabel).count();
			priors[k] = (double)classInstanceCount / (double)trainingDocumentCount;
		}
	}

	/**
	 * Calculates the data likelihoods (theta values) for each vocabulary
	 * word and class from the training data, assuming a multivariate Bernoulli
	 * model.
	 *
	 * Performs smoothing by adding the given counts to the numerator or denominator
	 *
	 * @param numeratorDefaultCount Default count to add to num. for each likelihood
	 * @param denominatorDefaultCount Default count to add to denom. for each likelihood
	 */
	private void calculateMultivariateLikelihoods(int numeratorDefaultCount, int denominatorDefaultCount) {
		for(int k = 0; k < classLabels.size(); k++) {
			int classLabel = classLabels.get(k);
			// Get the number of times the jth word appears in the documents with the kth class.
			// For the multivariate model we don't care how many times the word appears, just whether
			// it is present or not. The way the data is structured, the word frequency map
			// should only contain words that occur at least once
			ArrayList<Document> classDocuments = trainingDocumentDataset.getDocuments().stream()
					.filter(d -> d.getClassLabel() == classLabel)
					.collect(Collectors.toCollection(ArrayList::new));
			for(int j = 0; j < vocabularySize; j++) {
				String vocabWord = vocabulary.get(j);

				long observationCount = classDocuments.stream()
						.filter(d -> d.getWordFrequencies().containsKey(vocabWord))
						.count();
				long classCount = classDocuments.size();

				// Use Laplace smoothing to prevent zero probabilities for unseen words
				likelihoods[j][k] = (double)(observationCount + numeratorDefaultCount) / (double)(classCount + denominatorDefaultCount);
			}
		}
	}

	/**
	 * Calculates the data likelihoods (theta values) for each vocabulary
	 * word and class from the training data, assuming a multivariate Bernoulli
	 * model.
	 *
	 * Uses Laplace smoothing to prevent zero probabilities in model predictions.
	 */
	private void calculateMultivariateLikelihoodsAddOneSmoothing() {
		calculateMultivariateLikelihoods(1, 2);
	}

	/**
	 * Calculates the data likelihoods (theta values) for each vocabulary
	 * word and class from the training data, assuming a multivariate Bernoulli
	 * model.
	 *
	 * Uses MAP estimates based on a Beta prior.
	 *
	 * @param alpha The alpha parameter to the Beta distribution
	 * @param beta The beta parameter to the Beta distribution
	 */
	private void calculateMultivariateLikelihoodsMap(int alpha, int beta) {
		calculateMultivariateLikelihoods(alpha - 1, alpha - 1 + beta - 1);
	}

	/**
	 * Calculates the data likelihoods (theta values) for each vocabulary
	 * word and class from the training data, assuming a multinomial model.
	 *
	 * Performs smoothing by adding the given counts to the numerator or denominator
	 *
	 * @param getNumeratorCount Gets the default count to add to num. for each likelihood given word index j
	 * @param denominatorDefaultCount Default count to add to denom. for each likelihood
	 */
	private void calculateMultinomialLikelihoods(IntUnaryOperator getNumeratorCount, int denominatorDefaultCount) {
		for(int k = 0; k < classLabels.size(); k++) {
			int classLabel = classLabels.get(k);
			// Get all the documents with the kth class
			ArrayList<Document> classDocuments = trainingDocumentDataset.getDocuments().stream()
					.filter(d -> d.getClassLabel() == classLabel)
					.collect(Collectors.toCollection(ArrayList::new));
			for(int j = 0; j < vocabularySize; j++) {
				String vocabWord = vocabulary.get(j);
				// Get the count of all occurrences of the jth word in documents of the kth class
				long observationCount = classDocuments.stream()
						.mapToInt(d -> d.getWordFrequencies().containsKey(vocabWord)
								? (int)d.getWordFrequencies().get(vocabWord)
								: 0).sum();
				long classDocumentsWordCount = classDocuments.stream()
						.mapToInt(d -> d.getWordCount()).sum();

				// Use Laplace smoothing to prevent zero probabilities for unseen words
				likelihoods[j][k] = (double)(observationCount + getNumeratorCount.applyAsInt(j)) / (double)(classDocumentsWordCount + denominatorDefaultCount);
			}
		}
	}

	/**
	 * Calculates the data likelihoods (theta values) for each vocabulary
	 * word and class from the training data, assuming a multinomial model.
	 *
	 * Uses Laplace smoothing to prevent zero probabilities in model predictions.
	 */
	private void calculateMultinomialLikelihoodsAddOneSmoothing() {
		calculateMultinomialLikelihoods((j) -> 1, vocabularySize);
	}

	/**
	 * Calculates the data likelihoods (theta values) for each vocabulary
	 * word and class from the training data, assuming a multinomial model.
	 *
	 * Uses MAP estimates based on a Dirichilet prior.
	 */
	private void calculateMultinomialLikelihoodsMap(ArrayList<Integer> alphas) {
		int alphasSum = alphas.stream().mapToInt(Integer::intValue).sum();
		calculateMultinomialLikelihoods((j) -> alphas.get(j) - 1, alphasSum - vocabularySize);
	}

	public int getVocabularySize() {
		return vocabularySize;
	}

}
