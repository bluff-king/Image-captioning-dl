from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class BLEU: 
    
    """
    A class to calculate BLEU (Bilingual Evaluation Understudy) score.
    """
    
    def __init__(self, references, candidate):
        """
        Initializes the BLEU class with reference sentences and candidate sentences for evaluating with corpus-level BLEU scores.

        Parameters:
        references (list of list of list of str): A list of reference sentences for each candidate, where each reference is a list of sentences, and each sentence is a list of words (tokens).
        candidate (list of list of str): A list of candidate sentences, where each candidate sentence is a list of words (tokens).

        Example:
        --------
        references = [
            [["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]],
            [["A", "dog", "barks", "at", "the", "mailman"], ["The", "dog", "barks", "at", "a", "mailman"]]
        ]
        candidate = [
            ["The", "cat", "sat", "on", "the", "mat"],
            ["A", "dog", "barks", "loudly"]
        ]

        bleu = BLEU(references, candidate)
        """
        self.references = references
        self.candidate = candidate

        
    def bleu1(self):
        # Calculates the BLEU-1 score, which evaluates the precision of unigrams (single words) in the candidate sentence compared to the reference sentences.
        return corpus_bleu(self.references, self.candidate, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)


    def bleu2(self):
        # Calculates the BLEU-2 score, which evaluates the precision of both unigrams and bigrams (two-word sequences) in the candidate sentence compared to the reference sentences.
        return corpus_bleu(self.references, self.candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method4)


    def bleu3(self):
        # Calculates the BLEU-3 score, which evaluates the precision of unigrams, bigrams, and trigrams (three-word sequences) in the candidate sentence compared to the reference sentences.
        return corpus_bleu(self.references, self.candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method4)


    def bleu4(self):
        # Calculates the BLEU-4 score, which evaluates the precision of unigrams, bigrams, trigrams, and 4-grams (four-word sequences) in the candidate sentence compared to the reference sentences.
        return corpus_bleu(self.references, self.candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method4)

    
    


    
    
    
