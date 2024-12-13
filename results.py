from models import TfidfSearch, BooleanSearch, FaissSearch
from common import abstract_texts, abstract_ids, query, evaluate, ground_truth

class Results:
    def __init__(self, abstracts, abstract_ids, ground_truth):
        self.abstracts = abstracts
        self.abstract_ids = abstract_ids
        self.ground_truth = ground_truth
    
    def tfidf_search(self, query, threshold=0.20):
        tfidf_search = TfidfSearch(self.abstracts)
        
        tfidf_results = tfidf_search.search(query, self.abstract_ids, threshold)
        tfidf_precision, tfidf_recall = evaluate(tfidf_results, self.ground_truth)

        print(f"TF-IDF Precision: {tfidf_precision:.4f}, Recall: {tfidf_recall:.4f}")
        return tfidf_precision, tfidf_recall
    
    def boolean_search(self, query):
        boolean_search = BooleanSearch(self.abstracts, self.abstract_ids)

        boolean_results = boolean_search.search(query, self.abstract_ids)
        boolean_precision, boolean_recall = evaluate(boolean_results, self.ground_truth)

        print(f"Boolean Precision: {boolean_precision:.4f}, Recall: {boolean_recall:.4f}")
        return boolean_precision, boolean_recall

    def faiss_search(self, query, threshold_high=0.62, threshold_low=0.35):
        faiss_search = FaissSearch(self.abstracts)
        
        faiss_high_results = faiss_search.search(query, self.abstracts, self.abstract_ids, threshold_high)
        faiss_low_results = faiss_search.search(query, self.abstracts, self.abstract_ids, threshold_low)
        
        high_precision, high_recall = evaluate(faiss_high_results, self.ground_truth)
        low_precision, low_recall = evaluate(faiss_low_results, self.ground_truth)

        print(f"Faiss High Threshold Precision: {high_precision:.4f}, Recall: {high_recall:.4f}")
        print(f"Faiss Low Threshold Precision: {low_precision:.4f}, Recall: {low_recall:.4f}")
        
        return (high_precision, high_recall), (low_precision, low_recall)
    
    def run_all_searches(self, query):
        print("****** Running TF-IDF Search ******")
        self.tfidf_search(query)

        print("\n****** Running Boolean Search ******")
        self.boolean_search(query)

        print("\n****** Running Faiss Search ******")
        self.faiss_search(query)

if __name__ == "__main__":
    
    results = Results(abstract_texts, abstract_ids, ground_truth)
    results.run_all_searches(query)
