# prepare_corpora.py

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from config import Config

cfg = Config()


def read_lines(path):
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield simple_preprocess(line)


def train_word2vec():

    print("Loading corpus...")

    sentences = read_lines(
        "./raw_corpora/raw_corpus.txt"
    )

    model = Word2Vec(
        sentences=sentences,
        vector_size=cfg.word_dim,
        window=5,
        min_count=2,
        workers=4,
        sg=1,          # skipgram
        negative=10,
        sample=1e-5,
        epochs=10
    )

    save_path = "./embeddings/raw_corpus_w2v.model"

    model.save(save_path)

    print("saved:", save_path)


if __name__ == "__main__":
    train_word2vec()
