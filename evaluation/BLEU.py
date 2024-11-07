from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BLEU: 
    
    """
    A class to calculate BLEU (Bilingual Evaluation Understudy) score.
    """
    
    def __init__(self, references, candidate):
        """
        Initialize a BLEU object with references and candidate sentences.

        Parameters:
        references (list(list(str))): A list of reference sentences. Each reference sentence is a list of words.
        candidate (list(str)): A candidate sentence as a list of words.
        
        """
        self.references = references
        self.candidate = candidate
        
    def bleu1(self):
        # Calculates the BLEU-1 score, which evaluates the precision of unigrams (single words) in the candidate sentence compared to the reference sentences.
        return sentence_bleu(self.references, self.candidate, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)


    def bleu2(self):
        # Calculates the BLEU-2 score, which evaluates the precision of both unigrams and bigrams (two-word sequences) in the candidate sentence compared to the reference sentences.
        return sentence_bleu(self.references, self.candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method4)


    def bleu3(self):
        # Calculates the BLEU-3 score, which evaluates the precision of unigrams, bigrams, and trigrams (three-word sequences) in the candidate sentence compared to the reference sentences.
        return sentence_bleu(self.references, self.candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method4)


    def bleu4(self):
        # Calculates the BLEU-4 score, which evaluates the precision of unigrams, bigrams, trigrams, and 4-grams (four-word sequences) in the candidate sentence compared to the reference sentences.
        return sentence_bleu(self.references, self.candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method4)

    
    


    
    
    
