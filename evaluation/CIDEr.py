import numpy as np 
import spacy 
from collections import Counter, defaultdict
import nltk

nlp = spacy.load('en_core_web_sm')

class CIDEr:
    """
    A class to calculate the CIDEr (Consensus-based Image Description Evaluation) metric.
    """
    
    def __init__(self, references, candidates):
        """
        Initialize a CIDEr object with reference and candidate sentences.

        Parameters:
        references (list of list of str): A list of reference sentences for each image. Each reference sentence is a list of words.
        candidates (list of str): A list of candidate sentences for each image. Each candidate sentence is a string.
        """
        # Store the provided reference and candidate sentences.
        self.references = references
        self.candidates = candidates
    
    def tokenize(self, caption):
        """
        Tokenize the given caption into words.
        
        Parameters:
        caption (str): The input sentence to be tokenized.
        
        Returns:
        list: A list of lowercase word tokens.
        """
        return nltk.word_tokenize(caption.lower())
    
    def lemmatization(self, caption):
        """
        Lemmatize the given caption to obtain base forms of words.
        
        Parameters:
        caption (list): A list of word tokens.
        
        Returns:
        list: A list of lemmatized word tokens.
        """
        doc = nlp(" ".join(caption))
        
        return [token.lemma_ for token in doc]
    
    def n_grams(self, tokens, n):
        """
        Generate n-grams from the given list of tokens.
        
        Parameters:
        tokens (list): A list of word tokens.
        n (int): The n-gram size.
        
        Returns:
        list: A list of n-gram strings.
        """
        return [' '.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    
    def compute_tf(self, sentence_tokens, n):
        """
        Compute the term frequency (TF) for n-grams in the sentence.
        
        Parameters:
        sentence_tokens (list): A list of word tokens.
        n (int): The n-gram size.
        
        Returns:
        dict: A dictionary with n-grams as keys and their TF values as values.
        """
        # Count the occurrences of each n-gram.
        ngram_counts = Counter(self.n_grams(sentence_tokens, n))
        total_ngrams = sum(ngram_counts.values())
        
        # Calculate and return the term frequency (TF).
        return {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    
    def compute_idf(self, reference_corpus, n):
        """
        Compute the inverse document frequency (IDF) for n-grams in the reference corpus.
        
        Parameters:
        reference_corpus (list): A list of reference sentences.
        n (int): The n-gram size.
        
        Returns:
        dict: A dictionary with n-grams as keys and their IDF values as values.
        """
        idf_counts = defaultdict(int)
        num_docs = len(reference_corpus)
        
        # Count document frequency for each unique n-gram across the reference corpus.
        for ref in reference_corpus:
            ref_tokens = set(self.n_grams(self.tokenize(ref), n))
            for token in ref_tokens: 
                idf_counts[token] += 1
        
        # Calculate IDF values.
        idf = {token: np.log((num_docs) / (idf_counts[token])) for token in idf_counts}
        return idf
    
    def single_cider(self, reference_corpus, candidate, n=4):
        """
        Calculate the CIDEr score for a single candidate sentence against a set of references.
        
        Parameters:
        reference_corpus (list): A list of reference sentences.
        candidate (str): The candidate sentence.
        n (int): The maximum n-gram size (default is 4).
        
        Returns:
        float: The CIDEr score for the candidate sentence.
        """
        # Tokenize and lemmatize the candidate and references.
        candidate_tokens = self.lemmatization(self.tokenize(candidate))
        reference_tokens = [self.lemmatization(self.tokenize(ref)) for ref in reference_corpus]

        # Compute term frequency (TF) for the candidate.
        tf_candidate = [self.compute_tf(candidate_tokens, i) for i in range(1, n+1)]
        
        # Compute inverse document frequency (IDF) for the reference corpus.
        idf = [self.compute_idf(reference_corpus, i) for i in range(1, n+1)]
        
        scores = []
        
        # Calculate similarity scores for each n-gram size.
        for i in range(n):
            ref_tf = [self.compute_tf(ref, i+1) for ref in reference_tokens]
            common_ngrams = set(tf_candidate[i].keys()).intersection(set(idf[i].keys()))
            
            # Create vectors for the candidate and references.
            candidate_vec = np.array([tf_candidate[i].get(ngram, 0) * idf[i].get(ngram, 0) for ngram in common_ngrams])
            reference_vecs = [np.array([tf.get(ngram, 0) * idf[i].get(ngram, 0) for ngram in common_ngrams]) for tf in ref_tf]
            
            sim_scores = []
            # Calculate cosine similarity between candidate vector and each reference vector.
            for ref_vec in reference_vecs:
                if np.linalg.norm(candidate_vec) != 0 and np.linalg.norm(ref_vec) != 0:
                    sim_scores.append((candidate_vec @ ref_vec) / (np.linalg.norm(candidate_vec) * np.linalg.norm(ref_vec)))
                else:
                    sim_scores.append(0)
            
            scores.append(np.mean(sim_scores))
        
        # Return the average score over all n-gram sizes.
        return np.sum(scores) / n   # Combine scores uniformly
            
    def cider(self):
        """
        Calculate the average CIDEr score for all candidate sentences.
        
        Returns:
        float: The average CIDEr score for all candidate sentences.
        """
        cider_scores = []
        
        # Compute the CIDEr score for each candidate-reference pair.
        for reference_corpus, candidate in zip(self.references, self.candidates):
            cider_scores.append(self.single_cider(reference_corpus, candidate))
        
        # Return the average CIDEr score across all candidates and references.
        return np.mean(cider_scores)
