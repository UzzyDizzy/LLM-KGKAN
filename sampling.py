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

    for label in label_groups:
        choices = random.sample(
            label_groups[label],
            min(k, len(label_groups[label]))
        )
        selected.update(choices)

    dataset.data = [dataset.data[i] for i in sorted(selected)]
    return dataset
