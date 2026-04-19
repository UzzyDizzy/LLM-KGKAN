# domains.py

import os

def validate_selected_domains(domains, source, low_resource, zero_shot):
    all_selected = source + low_resource + zero_shot

    # 1. check domain keys exist
    for d in all_selected:
        if d not in domains:
            raise ValueError(f"❌ Domain '{d}' not found in DOMAINS")

    # 2. check files exist
    for d in all_selected:
        path = domains[d]
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ File missing for domain '{d}': {path}")

    print("✅ Domain selection valid")

    return source, low_resource, zero_shot

DOMAINS = {
    "L": "data/laptops.csv",
    "R": "data/restaurants.csv",
    "D": "data/device.csv",
    "S": "data/service.csv",
    "A": "data/amazon.csv",
    "SH": "data/shoes.csv",
    "W": "data/water_purifier.csv",
    "U": "data/education.csv",
    "H": "data/healthcare.csv",
}

SOURCE_DOMAINS = ["L", "R"]
LOW_RESOURCE_TARGETS = ["A"]
ZERO_SHOT_TARGETS = ["U"]

SOURCE_DOMAINS, LOW_RESOURCE_TARGETS, ZERO_SHOT_TARGETS = validate_selected_domains(
    DOMAINS,
    SOURCE_DOMAINS,
    LOW_RESOURCE_TARGETS,
    ZERO_SHOT_TARGETS
)