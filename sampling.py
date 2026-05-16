# sampling.py

import random
from collections import defaultdict


def few_shot_sample(dataset, k):
    label_groups = defaultdict(list)

    for i, sample in enumerate(dataset.data):
        _, _, sentiments, _, _ = sample

        for label in set(sentiments):
            if isinstance(label, str) and label and label != "NULL":
                label_groups[label.lower()].append(i)

    selected = set()

    for label in sorted(label_groups):
        choices = random.sample(
            label_groups[label],
            min(k, len(label_groups[label]))
        )
        selected.update(choices)

    dataset.data = [dataset.data[i] for i in sorted(selected)]
    return dataset
