"""Project constants."""

FLAVOR_LABELS = [
    "sweet",
    "bitter",
    "sour",
    "salty",
    "umami",
    "fruity",
    "floral",
    "spicy",
    "herbal",
    "fatty",
    "smoky",
    "nutty",
    "caramel",
    "vanilla",
    "citrus",
    "mint",
    "woody",
    "earthy",
    "sulfurous",
    "alcoholic",
    "astringent",
    "cooling",
    "metallic",
    "kokumi",
]

LABEL_TO_INDEX = {name: idx for idx, name in enumerate(FLAVOR_LABELS)}
INDEX_TO_LABEL = {idx: name for idx, name in enumerate(FLAVOR_LABELS)}

