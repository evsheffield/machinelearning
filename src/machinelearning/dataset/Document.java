package machinelearning.dataset;

import java.util.Map;

/**
 * Represents a text document with a class label and set of word
 * frequencies.
 * @author evanc
 *
 */
public class Document {
	private int id;
	private Map<String, Integer> wordFrequencies;
	private int classLabel;
	private int wordCount;

	public Document(int id, Map<String, Integer> wordFrequencies, int classLabel) {
		super();
		this.id = id;
		this.wordFrequencies = wordFrequencies;
		this.classLabel = classLabel;
		// Compute the total count of all words in the document from the word frequencies
		this.wordCount = wordFrequencies.values().stream().mapToInt(Integer::intValue).sum();
	}

	public int getId() {
		return id;
	}

	public Map<String, Integer> getWordFrequencies() {
		return wordFrequencies;
	}

	public int getClassLabel() {
		return classLabel;
	}

	public int getWordCount() {
		return wordCount;
	}
}
