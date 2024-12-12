from nltk.translate.meteor_score import meteor_score

class METEOR:
    """
    A class to calculate METEOR (Metric for Evaluation of Translation with Explicit ORdering) score.
    """

    def __init__(self, references, candidates):
        """
        Initialize the METEOR class with reference and candidate sentences.

        Parameters:
        references (list of lists of str): A list of reference sentences. Each reference sentence is a list of words.
        candidates (list of str): A list of candidate sentences. Each candidate sentence is a string.
        """
        
        self.references = [[reference.lower().split() for reference in reference_corpus] for reference_corpus in references]
        self.candidates = [candidate.lower().split() for candidate in candidates]

        
    def meteor(self):
        """
        Calculate the average METEOR score for all candidate sentences.

        Returns:
        float: The average METEOR score for the provided candidate sentences.
        """
        
        total_score = 0
        for reference, candidate in zip(self.references, self.candidates):
            total_score += meteor_score(reference, candidate)
        
        return total_score / len(self.candidates)