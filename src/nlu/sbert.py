from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NLUSBERT:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def find_most_similar_word(self, word_list, target_word):
        embeddings = self.model.encode(word_list)
        target_embedding = self.model.encode([target_word])
        
        similarities = cosine_similarity(target_embedding, embeddings)
        best_match_index = np.argmax(similarities)
        best_match_label = word_list[best_match_index]
        best_match_similarity = similarities[0][best_match_index]

        similarity_threshold = 0.7
        
        if best_match_similarity < similarity_threshold:
            return "No similar object found"
        else:
            return best_match_label

# example:
# nlu_sbert = NLUSBERT()
# result = nlu_sbert.find_most_similar_word(["person", "bicycle", "car"], "mountain")
# print(result)


