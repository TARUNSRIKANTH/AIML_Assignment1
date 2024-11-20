import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import math

# Download necessary NLTK resources
nltk.download('punkt')


def calculate_tf(text):
    tf_scores = defaultdict(dict)
    words = word_tokenize(text.lower())
    total_words = len(words)

    for word in words:
        if word not in tf_scores:
            tf_scores[word] = words.count(word) / total_words

    return tf_scores


def calculate_idf(documents):
    idf_scores = {}
    total_documents = len(documents)

    # Count occurrences of each word across all documents
    word_document_count = defaultdict(int)

    for doc in documents:
        words = set(word_tokenize(doc.lower()))
        for word in words:
            word_document_count[word] += 1

    # Calculate IDF for each word
    for word, count in word_document_count.items():
        idf_scores[word] = math.log(total_documents / (1 + count))

    return idf_scores


def calculate_tfidf(documents):
    tfidf_scores = []
    idf_scores = calculate_idf(documents)

    for doc in documents:
        tf_scores = calculate_tf(doc)
        tfidf = {word: tf * idf_scores.get(word, 0) for word, tf in tf_scores.items()}
        tfidf_scores.append(tfidf)

    return tfidf_scores


def score_sentences(tfidf_scores):
    sentence_scores = defaultdict(float)

    for i, tfidf in enumerate(tfidf_scores):
        for word, score in tfidf.items():
            sentence_scores[i] += score

    return sentence_scores


def summarize(text, num_sentences=2):
    sentences = sent_tokenize(text)
    documents = sentences  # Each sentence is treated as a document

    # Calculate TF-IDF scores
    tfidf_scores = calculate_tfidf(documents)

    # Score sentences based on their TF-IDF scores
    sentence_scores = score_sentences(tfidf_scores)

    # Sort sentences by score and select the top ones
    ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top N sentences to form the summary
    top_sentences_indices = sorted([ranked_sentences[i][0] for i in range(num_sentences)])

    summary = ' '.join([sentences[i] for i in top_sentences_indices])

    return summary


# Example usage
if __name__ == "__main__":
    text = """
    Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is the strain of coronavirus 
    that causes coronavirus disease 2019 (COVID-19), a respiratory illness. 
    During the initial outbreak in Wuhan, China, the virus was commonly referred to as "coronavirus" or 
    "Wuhan coronavirus" . In March 2020, U.S. President Donald Trump referred to the virus as the 
    "Chinese virus" in tweets, interviews, and White House press briefings. 
    Based on whole genome sequence similarity, a pangolin coronavirus candidate strain was found to be less similar 
    than RaTG13, but more similar than other bat coronaviruses to SARS-CoV-2. 
    Arinjay Banerjee, a virologist at McMaster University, notes that "the SARS virus shared 99.8% of its genome 
    with a civet coronavirus, which is why civets were considered the source." 
    The virion then releases RNA into the cell and forces the cell to produce and disseminate copies of the vi
    """

    summary = summarize(text, num_sentences=2)

    print("Original Text:\n", text)
    print("\nSummary:\n", summary)