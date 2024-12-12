from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class BLEU: 
    
    """
    A class to calculate BLEU (Bilingual Evaluation Understudy) score.
    """
    
    def __init__(self, references, candidates):
        """
        Initialize a BLEU object with reference and candidate sentences.

        Parameters:
        references (list of lists of str): A list of reference sentences. Each reference sentence is a list of words.
        candidates (list of str): A list of candidate sentences. Each candidate sentence is a string of words.
        """
        self.references = [[reference.lower().split() for reference in reference_corpus] for reference_corpus in references]
        self.candidates = [candidate.lower().split() for candidate in candidates]


        
    def bleu1(self):
        # Calculates the BLEU-1 score, which evaluates the precision of unigrams (single words) in the candidate sentence compared to the reference sentences.
        return corpus_bleu(self.references, self.candidates, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)


    def bleu2(self):
        # Calculates the BLEU-2 score, which evaluates the precision of both unigrams eand bigrams (two-word sequences) in the candidate sentence compared to the reference sentences.
        return corpus_bleu(self.references, self.candidates, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method4)


    def bleu3(self):
        # Calculates the BLEU-3 score, which evaluates the precision of unigrams, bigrams, and trigrams (three-word sequences) in the candidate sentence compared to the reference sentences.
        return corpus_bleu(self.references, self.candidates, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method4)


    def bleu4(self):
        # Calculates the BLEU-4 score, which evaluates the precision of unigrams, bigrams, trigrams, and 4-grams (four-word sequences) in the candidate sentence compared to the reference sentences.
        return corpus_bleu(self.references, self.candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method4)         