import logging, numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.embeddings import Embeddings
from . import config
from .config import Element, MatchResult, MatchedPair

logger = logging.getLogger(__name__)

class TextMatcher:
    def __init__(self, embedding_model: Embeddings):
        if not hasattr(embedding_model, 'embed_documents'): raise TypeError("embedding_model must be a valid LangChain Embeddings instance.")
        self.embedding_model = embedding_model
    def match_text_elements(self, d_texts: List[Element], y_texts: List[Element]) -> MatchResult:
        if not d_texts or not y_texts: return [], d_texts, y_texts
        d_contents, y_contents = [str(e.get('content', {}).get('text', '')) for e in d_texts], [str(e.get('content', {}).get('text', '')) for e in y_texts]
        if not d_contents or not y_contents: return [], d_texts, y_texts
        try:
            vecs = self.embedding_model.embed_documents(d_contents + y_contents)
            d_vecs, y_vecs = vecs[:len(d_contents)], vecs[len(d_contents):]
        except Exception as e:
            logger.error(f"Failed to get embeddings for text: {e}"); return [], d_texts, y_texts
        sim_matrix = cosine_similarity(d_vecs, y_vecs)
        pairs, d_matched, y_matched = [], set(), set()
        for i in range(len(d_texts)):
            if not np.any(sim_matrix[i]): continue
            best_y = np.argmax(sim_matrix[i])
            if sim_matrix[i][best_y] >= config.SIMILARITY_THRESHOLD and best_y not in y_matched:
                pairs.append((d_texts[i], y_texts[best_y])); d_matched.add(i); y_matched.add(best_y)
                sim_matrix[:, best_y] = -1
        d_only, y_only = [e for i, e in enumerate(d_texts) if i not in d_matched], [e for j, e in enumerate(y_texts) if j not in y_matched]
        logger.info(f"Text matching: {len(pairs)} pairs, {len(d_only)} docling-only, {len(y_only)} docyolo-only.")
        return pairs, d_only, y_only

