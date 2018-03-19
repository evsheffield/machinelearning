package machinelearning.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

/**
 * A data set consisting of documents for use in document
 * classification tasks.
 *
 * @author evanc
 */
public class DocumentDataset {
	private ArrayList<String> vocabularyWords;
	private ArrayList<String> stopWords;
	private ArrayList<Document> documents;
	/** A mapping of words to frequencies for the entire dataset, sorted highest to lowest */
	private Map<String, Integer> wordFrequencies;
	private ArrayList<Integer> distinctClassLabels;

	public DocumentDataset(String vocabFilePath, String dataFilePath, String labelFilePath, boolean removeStopWords) {
		vocabularyWords = new ArrayList<String>();
		stopWords = new ArrayList<String>();
		documents = new ArrayList<Document>();
		distinctClassLabels = new ArrayList<Integer>();
		try {
			readVocabularyFromFile(vocabFilePath);
		} catch(IOException e) {
			System.out.println("Unable to read vocab file : " + vocabFilePath);
		}
		try {
			readDocumentsFromDataFiles(dataFilePath, labelFilePath);
		} catch(IOException e) {
			System.out.println("Unable to read data or label file : " + dataFilePath + ", " + labelFilePath);
		}
		if(removeStopWords) {
			try {
				readStopWordsFromFile();
				vocabularyWords = (ArrayList<String>)vocabularyWords.stream().filter(word -> !stopWords.contains(word)).collect(Collectors.toList());
			} catch(IOException e) {
				System.out.println("Unable to read stop words file");
			}
		}
		populateWordFrequenciesList();
	}

	public DocumentDataset(String vocabFilePath, String dataFilePath, String labelFilePath) {
		this(vocabFilePath, dataFilePath, labelFilePath, false);
	}

	/**
	 * Reads the vocabulary list from a file with a single vocabulary word per line.
	 *
	 * @param vocabFilePath The file to read the vocabulary from
	 * @throws IOException
	 */
	private void readVocabularyFromFile(String vocabFilePath) throws IOException {
		FileReader file = new FileReader(vocabFilePath);
		BufferedReader br = new BufferedReader(file);

		String line = "";
		while((line = br.readLine()) != null) {
			// Skip empty lines
			if(line.matches("^\\s*$")) {
				continue;
			}
			vocabularyWords.add(line);
		}
	}

	private void readStopWordsFromFile() throws IOException {
		FileReader file = new FileReader("data/stopwords.txt");
		BufferedReader br = new BufferedReader(file);

		String line = "";
		while((line = br.readLine()) != null) {
			// Skip empty lines
			if(line.matches("^\\s*$")) {
				continue;
			}
			stopWords.add(line);
		}
	}

	/**
	 * Reads documents from a data file describing the frequency of words in each
	 * document.
	 *
	 * @param dataFilePath The path to the data file
	 * @param labelFilePath The path to the label file with class labels
	 * for the documents.
	 * @throws IOException
	 */
	private void readDocumentsFromDataFiles(String dataFilePath, String labelFilePath) throws IOException {
		FileReader dataFile = new FileReader(dataFilePath);
		FileReader labelFile = new FileReader(labelFilePath);
		BufferedReader dataBr = new BufferedReader(dataFile);
		BufferedReader labelBr = new BufferedReader(labelFile);

		int currentDocId = 1;

		String dataLine = "";
		HashMap<String, Integer> counts = new HashMap<String, Integer>();
		int docId, wordId, count;
		while((dataLine = dataBr.readLine()) != null) {
			String[] lineContents = dataLine.split("\\s+");
			docId = Integer.parseInt(lineContents[0]);
			wordId = Integer.parseInt(lineContents[1]);
			count = Integer.parseInt(lineContents[2]);

			if(docId != currentDocId) {
				// We've read all the word frequencies for a document!
				// Finish this document and move on to the next
				int nextLabel = Integer.parseInt(labelBr.readLine());
				if(!distinctClassLabels.contains(nextLabel)) {
					distinctClassLabels.add(nextLabel);
				}
				Document doc = new Document(currentDocId, counts, nextLabel);
				documents.add(doc);

				// Start building up the new document
				counts = new HashMap<String, Integer>();
				currentDocId = docId;
			}

			// Process the counts for this line
			String word = vocabularyWords.get(wordId - 1);
			counts.put(word, count);
		}

		// Process the final document
		int nextLabel = Integer.parseInt(labelBr.readLine());
		if(!distinctClassLabels.contains(nextLabel)) {
			distinctClassLabels.add(nextLabel);
		}
		Document doc = new Document(currentDocId, counts, nextLabel);
		documents.add(doc);
	}

	/**
	 * Populates a word frequencies list from all of the documents in the dataset.
	 */
	private void populateWordFrequenciesList() {
		HashMap<String, Integer> wordFreqs = new HashMap<String, Integer>();
		// Create an entry for every word in the vocabulary
		for(String word : vocabularyWords) {
			wordFreqs.put(word, 0);
		}
		// Iterate over all the training documents and add their word
		// frequencies to the combined map
		for(Document doc : documents) {
			Map<String, Integer> docWordFreqs = doc.getWordFrequencies();
			for(Entry<String, Integer> entry : docWordFreqs.entrySet()) {
				if(wordFreqs.containsKey(entry.getKey())) {
					int newVal = wordFreqs.get(entry.getKey()) + entry.getValue();
					wordFreqs.put(entry.getKey(), newVal);
				}
			}
		}

		// Sort the map in order by decreasing word frequency
		// Reference: https://stackoverflow.com/questions/29567575/sort-map-by-value-using-java-8
		wordFrequencies = wordFreqs.entrySet().stream()
				.sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
				//.sorted((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
				.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
	}

	public ArrayList<String> getVocabularyWords() {
		return vocabularyWords;
	}

	public ArrayList<Document> getDocuments() {
		return documents;
	}

	public Map<String, Integer> getWordFrequencies() {
		return wordFrequencies;
	}

	public ArrayList<Integer> getDistinctClassLabels() {
		return distinctClassLabels;
	}

}
