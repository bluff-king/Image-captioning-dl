import numpy as np
from collections import defaultdict, Counter
import nltk
from nltk.stem import WordNetLemmatizer
import spacy 
nlp = spacy.load('en_core_web_sm')

# Ensure NLTK packages are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def lemmatize_tokens(tokens):
        doc = nlp(" ".join(tokens))
        
        return [token.lemma_ for token in doc]

def n_grams(tokens, n):
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def compute_tf(sentence_tokens, n):
    ngram_counts = Counter(n_grams(sentence_tokens, n))
    total_ngrams = sum(ngram_counts.values())
    return {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}

def compute_idf(reference_corpus, n):
    idf_counts = defaultdict(int)
    num_docs = len(reference_corpus)
    
    for ref in reference_corpus:
        ref_tokens = set(n_grams(tokenize(ref), n))
        for token in ref_tokens:
            idf_counts[token] += 1
    
    idf = {token: np.log((num_docs + 1) / (idf_counts[token] + 1)) for token in idf_counts}
    return idf

def compute_cider(candidate, references, n=4):
    candidate_tokens = lemmatize_tokens(tokenize(candidate))
    reference_tokens = [lemmatize_tokens(tokenize(ref)) for ref in references]

    tf_candidate = [compute_tf(candidate_tokens, i) for i in range(1, n+1)]
    idf = [compute_idf(references, i) for i in range(1, n+1)]
    scores = []
    
    for i in range(n):
        ref_tf = [compute_tf(ref, i+1) for ref in reference_tokens]
        common_ngrams = set(tf_candidate[i].keys()).intersection(set(idf[i].keys()))
        candidate_vec = np.array([tf_candidate[i].get(ngram, 0) * idf[i].get(ngram, 0) for ngram in common_ngrams])
        reference_vecs = [np.array([tf.get(ngram, 0) * idf[i].get(ngram, 0) for ngram in common_ngrams]) for tf in ref_tf]
        
        sim_scores = []
        for ref_vec in reference_vecs:
            if np.linalg.norm(candidate_vec) != 0 and np.linalg.norm(ref_vec) != 0:
                sim_scores.append(np.dot(candidate_vec, ref_vec) / (np.linalg.norm(candidate_vec) * np.linalg.norm(ref_vec)))
            else:
                sim_scores.append(0)
        
        scores.append(np.mean(sim_scores))
    
    return np.mean(scores)

# Example usage:
candidate_sentence = "A person riding a horse on a beach."
reference_sentences = [
    "A man is riding a horse by the shore.",
    "A rider on a horse is moving along the beach.",
    "A person is horseback riding near the water."
]

cider_score = compute_cider(candidate_sentence, reference_sentences)
print("CIDEr Score:", cider_score)
