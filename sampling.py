# sampling.py

import random
from collections import defaultdict


def few_shot_sample(dataset, k):
    from collections import defaultdict
    import random

    label_groups = defaultdict(list)

    for i, (tokens, aspects, sentiments) in enumerate(dataset.data):
        label = sentiments[0]  # assume one primary sentiment per sentence
        label_groups[label].append(i)

    selected = set()

    per_class = max(1, k // len(label_groups))

    for label in label_groups:
        choices = random.sample(label_groups[label], min(per_class, len(label_groups[label])))
        selected.update(choices)

    dataset.data = [dataset.data[i] for i in selected]
    return dataset