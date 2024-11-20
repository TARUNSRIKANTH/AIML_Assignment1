import math
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize the stemmer and stop words
ps = PorterStemmer()
stopWords = set(stopwords.words("english"))

def _create_frequency_matrix(sentences):
    """Calculate the frequency of words in each sentence."""
    frequency_matrix = {}
    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue
            freq_table[word] = freq_table.get(word, 0) + 1
        frequency_matrix[sent] = freq_table  # Store full sentence as key
    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    """Calculate Term Frequency (TF) and generate a matrix."""
    tf_matrix = {}
    for sent, f_table in freq_matrix.items():
        count_words_in_sentence = sum(f_table.values())
        tf_table = {word: count / count_words_in_sentence for word, count in f_table.items()}
        tf_matrix[sent] = tf_table
    return tf_matrix


def _create_documents_per_words(freq_matrix):
    """Create a table for documents per word."""
    word_per_doc_table = {}
    for f_table in freq_matrix.values():
        for word in f_table.keys():
            word_per_doc_table[word] = word_per_doc_table.get(word, 0) + 1
    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    """Calculate Inverse Document Frequency (IDF) and generate a matrix."""
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {word: math.log10(total_documents / float(count_doc_per_words[word])) for word in f_table.keys()}
        idf_matrix[sent] = idf_table
    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    """Calculate TF-IDF and generate a matrix."""
    tf_idf_matrix = {}
    for sent in tf_matrix.keys():
        tf_idf_table = {word: tf_matrix[sent][word] * idf_matrix[sent][word] for word in tf_matrix[sent].keys()}
        tf_idf_matrix[sent] = tf_idf_table
    return tf_idf_matrix


def _score_sentences(tf_idf_matrix):
    """Score the sentences based on their TF-IDF values."""
    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = sum(f_table.values())
        sentenceValue[sent] = total_score_per_sentence  # Use total score directly
    return sentenceValue


def _find_average_score(sentenceValue):
    """Find the average score from the sentence value dictionary."""
    sumValues = sum(sentenceValue.values())
    average = sumValues / len(sentenceValue) if len(sentenceValue) > 0 else 0
    return average


def _generate_summary(sentences, sentenceValue, threshold) -> object:
    """Select sentences for summarization based on their scores."""
    summary = ' '.join([sentence for sentence in sentences if sentenceValue.get(sentence, 0) >= threshold])
    return summary


def summarize_text(text):
    """Main function to summarize the input text."""

    # Sentence tokenization
    sentences = sent_tokenize(text)

    # Create frequency matrix
    freq_matrix = _create_frequency_matrix(sentences)

    # Calculate Term Frequency (TF)
    tf_matrix = _create_tf_matrix(freq_matrix)

    # Create table of documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)

    # Calculate Inverse Document Frequency (IDF)
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, len(sentences))

    # Calculate TF-IDF matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)

    # Score sentences based on TF-IDF values
    sentence_scores = _score_sentences(tf_idf_matrix)

    # Find threshold score
    threshold = _find_average_score(sentence_scores)

    print("\n Sentence Scores:", sentence_scores)  # Debugging output
    print("Threshold:", threshold)  # Debugging output

    # Generate summary based on threshold with dynamic adjustment.

    summary_threshold_multiplier = 1.3  # You can adjust this multiplier to change summary length.

    # Adjusting threshold dynamically if no sentences are selected initially.

    summary_threshold_adjusted = summary_threshold_multiplier * threshold

    summary = _generate_summary(sentences, sentence_scores, summary_threshold_adjusted)

    # If no summary is generated with adjusted threshold, use a lower multiplier.
    if not summary:
        print("No summary generated with adjusted threshold. Trying with lower multiplier.")
        summary_threshold_adjusted *= 0.9  # Decrease multiplier by 10%
        summary = _generate_summary(sentences, sentence_scores, summary_threshold_adjusted)

        return summary


if __name__ == "__main__":
    # Input text from user or predefined text.
    text_input = """
  	Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is the strain of coronavirus 
  	that causes coronavirus disease 2019 (COVID-19), a respiratory illness. 
  	During the initial outbreak in Wuhan, China, the virus was commonly referred to as "coronavirus" or 
  	"Wuhan coronavirus". In March 2020, U.S. President Donald Trump referred to the virus as the 
  	"Chinese virus" in tweets, interviews, and White House press briefings. 
  	Based on whole genome sequence similarity, a pangolin coronavirus candidate strain was found to be less similar 
  	than RaTG13 but more similar than other bat coronaviruses to SARS-CoV-2. 
  	Arinjay Banerjee, a virologist at McMaster University, notes that "the SARS virus shared 99.8% of its genome 
  	with a civet coronavirus, which is why civets were considered the source." 
  	The virion then releases RNA into the cell and forces the cell to produce and disseminate copies of the virion.
  	"""

    # Generate and print summary.
    summary_output = summarize_text(text_input)

    print("\n Generated Summary:")
    print(summary_output)