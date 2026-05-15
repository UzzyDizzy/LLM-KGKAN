"""
Topic Extraction — Step 2 of the KETGM pipeline.

Extracts TWLDA topic words from source and target domain texts.
Paper Section III-B1.
"""
import time

from config import Config
from utils import extract_topics


def run(train_samples, target_samples, config=None):
    """
    Extract topic words from source (train) and target domains.

    Parameters
    ----------
    train_samples   : list[dict] — labeled source domain samples
    target_samples  : list[dict] — unlabeled target domain samples

    Returns
    -------
    list[str] — deduplicated topic words from both domains.
    """
    if config is None:
        config = Config

    t0 = time.time()

    source_texts = [s['text'] for s in train_samples]
    target_texts = [s['text'] for s in target_samples]

    topic_words_source = extract_topics(
        source_texts,
        num_topics=config.NUM_TOPICS,
        words_per_topic=config.WORDS_PER_TOPIC
    )

    topic_words_target = extract_topics(
        target_texts,
        num_topics=config.NUM_TOPICS,
        words_per_topic=config.WORDS_PER_TOPIC
    )

    topic_words = list(set(topic_words_source + topic_words_target))

    print(f'Extracted {len(topic_words)} topic words  ({time.time()-t0:.1f}s)')
    for i in range(0, len(topic_words), config.WORDS_PER_TOPIC):
        chunk = topic_words[i:i+config.WORDS_PER_TOPIC]
        print(f'  Topic {i//config.WORDS_PER_TOPIC}: {chunk}')

    return topic_words
