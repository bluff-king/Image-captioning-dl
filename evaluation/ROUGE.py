from rouge_score import rouge_scorer, scoring

class ROUGE_L: 
    
    def __init__(self, references, candidates):
        self.references = [[reference.lower() for reference in reference_corpus] for reference_corpus in references]
        self.candidates = [candidate.lower() for candidate in candidates]
    
    def rouge_l(self):
        total = 0
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        for references, candidate in zip(self.references, self.candidates):
            total += sum(scorer.score(reference, candidate)['rougeL'].fmeasure for reference in references)/len(references)
        
        return total/len(self.references)