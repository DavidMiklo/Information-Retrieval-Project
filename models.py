from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from collections import defaultdict

class TfidfSearch:
    def __init__(self, abstracts):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(abstracts)

    def search(self, query, abstract_ids, threshold=0.20):
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = self.tfidf_matrix.dot(query_vec.T).toarray().flatten()
        filtered_indices = np.where(tfidf_scores > threshold)[0]
        return [{'id': abstract_ids[i]} for i in filtered_indices]

class BooleanSearch:
    def __init__(self, abstracts, abstract_ids):
        self.inverted_index = self._build_inverted_index(abstracts, abstract_ids)

    def _tokenize(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        return text.split()

    def _build_inverted_index(self, abstracts, abstract_ids):
        inverted_index = defaultdict(set)
        for abstract, abstract_id in zip(abstracts, abstract_ids):
            tokens = self._tokenize(abstract)
            for token in tokens:
                inverted_index[token].add(abstract_id)
        return {k: sorted(v) for k, v in inverted_index.items()}

    def search(self, query, all_ids):
        tokens = self._tokenize(query)
        result_set = set(all_ids)
        operation = "AND"
        for token in tokens:
            if token.upper() in {"AND", "OR", "NOT"}:
                operation = token.upper()
            else:
                token_ids = self.inverted_index.get(token, [])
                if operation == "AND":
                    result_set &= set(token_ids)
                elif operation == "OR":
                    result_set |= set(token_ids)
                elif operation == "NOT":
                    result_set -= set(token_ids)
        return [{'id': id_} for id_ in sorted(result_set)]

class FaissSearch:
    def __init__(self, abstracts):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(abstracts, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, abstracts, abstract_ids, threshold):
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, len(abstracts))
        results = []
        for i, d in zip(indices[0], distances[0]):
            if d >= threshold:
                results.append({'id': abstract_ids[i], 'distance': d, 'abstract': abstracts[i]})
        return results
