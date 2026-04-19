# domains.py

DOMAINS = {
    "L": "data/laptop.csv",
    "R": "data/restaurant.csv",
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