import codecs as cs
from nltk.translate.bleu_score import corpus_bleu
from bert_score import score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import linalg
import numpy as np

#####Motion-to-Text Metrics
# refs: tokenized words (corpus_size, ref_size, word_size)
# candidate: tokenized words (corpus_size, word_size)
def calculate_bleu_score(refs, candidate):
    return corpus_bleu(refs, candidate, weights=(1, 0, 0, 0)),\
           corpus_bleu(refs, candidate, weights=(0.5, 0.5, 0, 0)), \
           corpus_bleu(refs, candidate, weights=(0.33, 0.33, 0.33, 0)), \
           corpus_bleu(refs, candidate, weights=(0.25, 0.25, 0.25, 0.25))


# refs: tokenized words (corpus_size, ref_size, word_size)
# candidate: tokenized words (corpus_size, word_size)
def calculate_bert_score(refs, candidate):
    return score(candidate, refs, lang='en', rescale_with_baseline=True, idf=True)[-1]


class CiderScore:
    def __init__(self, ref_txt_path):

        self.gram_i_vectorizers = []
        with cs.open(ref_txt_path, 'r') as f:
            all_refs = [line.strip() for line in f.readlines()]
        for i in range(4):
            gram_i_vectorizer = TfidfVectorizer(ngram_range=(i+1, i+1))
            gram_i_vectorizer.fit(all_refs)
            self.gram_i_vectorizers.append(gram_i_vectorizer)

    def calculate_cider_i_score(self, refs, candidate, i):
        # (b, vocab)
        ref_i_tf_idf_mat = self.gram_i_vectorizers[i].transform(refs)
        # (1, vocab)
        cand_i_tf_idf_mat = self.gram_i_vectorizers[i].transform(candidate)
        norm_ref = linalg.norm(ref_i_tf_idf_mat, axis=1)
        norm_cand = linalg.norm(cand_i_tf_idf_mat, axis=1)
        ref_i_tf_idf_mat = ref_i_tf_idf_mat / norm_ref
        cand_i_tf_idf_mat = cand_i_tf_idf_mat / norm_cand
        cosine_similarity = ref_i_tf_idf_mat.dot(cand_i_tf_idf_mat.T)
        avg_smilarity = cosine_similarity.mean()
        return avg_smilarity

    # # refs: tokenized words (corpus_size, ref_size)
    # # candidate: tokenized words (corpus_size)
    def calculate_cider_score(self, refs, candidate):
        cider_score = np.zeros(len(refs))
        for k, (ref, cand) in enumerate(zip(refs, candidate)):
            cider_k = 0
            for i in range(4):
                cider_k += self.calculate_cider_i_score(i)
            cider_k /= 4
            cider_score[k] = cider_k
        return cider_score


