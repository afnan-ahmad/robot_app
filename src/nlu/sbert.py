from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from logging import getLogger

logger = getLogger("NLUBERT")

class NLUSBERT:
    def __init__(self, word_list, similarity_threshold=0.7, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

        self.word_list = word_list
        self.embeddings = self.model.encode(word_list)

        logger.info("Initialized SBERT with", model_name)

    def find_most_similar_word(self, target_word):
        logger.info(f"Target word is: {target_word}", "Finding most similar word...")

        target_embedding = self.model.encode([target_word])
        
        similarities = cosine_similarity(target_embedding, self.embeddings)

        best_match_index = np.argmax(similarities)
        best_match_label = self.word_list[best_match_index]
        best_match_similarity = similarities[0][best_match_index]

        if best_match_similarity < self.similarity_threshold:
            logger.info("Match similarity is below threshold. No best match found.")
            
            return None
        
        return best_match_label

# example:
# nlu_sbert = NLUSBERT()
# result = nlu_sbert.find_most_similar_word(["person", "bicycle", "car"], "mountain")
# print(result)


