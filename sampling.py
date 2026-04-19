# sampling.py

import random
from collections import defaultdict


def few_shot_sample(dataset, k):
    from collections import defaultdict
    import random

    label_groups = defaultdict(list)

    for i, sample in enumerate(dataset.data):
        # unpack safely (NEW FORMAT)
        text, aspects, sentiments, starts, ends = sample

        if len(sentiments) == 0:
            continue

        # use first sentiment (same assumption as before)
        label = sentiments[0]
        label_groups[label].append(i)

    selected = set()

    per_class = max(1, k // max(len(label_groups), 1))

    for label in label_groups:
        choices = random.sample(
            label_groups[label],
            min(per_class, len(label_groups[label]))
        )
        selected.update(choices)

    dataset.data = [dataset.data[i] for i in selected]
    return dataset