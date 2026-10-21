from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Python decorators allow functions to be wrapped with additional behavior.",  # 1 - Python
    "FastAPI makes it easy to build high performance APIs with async support.",   # 2 - Python
    "Virtual environments isolate Python dependencies for cleaner project setups.",  # 3 - Python
    "Docker helps package Python applications into reproducible containers.",     # 4 - Python
    "Unit testing in Python is commonly done using the pytest framework.",        # 5 - Python

    "Searing meat at high heat creates a flavorful crust through the Maillard reaction.",  # 6 - Cooking
    "Simmering soups slowly helps deepen flavor and blend ingredients evenly.",   # 7 - Cooking
    "Fresh herbs should be added at the end of cooking to preserve their aroma.", # 8 - Cooking
    "Baking requires precise measurements because small changes affect texture.", # 9 - Cooking
    "Marinating chicken overnight improves tenderness and enhances seasoning.",   # 10 - Cooking

    "The International Space Station orbits Earth at roughly ninety minutes per rotation.",  # 11 - Space
    "Black holes form when massive stars collapse under their own gravity.",      # 12 - Space
    "Mars rovers collect soil samples to study the planet’s geological history.", # 13 - Space
    "The James Webb Space Telescope captures infrared images of distant galaxies.", # 14 - Space
    "Astronauts train extensively to adapt to microgravity environments."         # 15 - Space
]

embeddings = model.encode(sentences)

print("Find the Best Match")

queries = [
    "How do I deploy my Python app?",
    "What techniques improve flavor when cooking?",
    "How does the space station stay in orbit?",
    "What tools help manage Python project dependencies?",
    "What training do astronauts need?"
]

thresholds = [0.3, 0.5, 0.7]

for query in queries:
    query_emb = model.encode(query)
    scores = util.cos_sim(query_emb, embeddings)[0]

    print(f"\nQuery: \"{query}\"")

    for t in thresholds:
        # Filter results above threshold
        passing = [(i, float(scores[i])) for i in range(len(scores)) if scores[i] >= t]

        print(f"\n  Threshold {t}: {len(passing)} results")

        # Sort by score descending
        passing = sorted(passing, key=lambda x: -x[1])

        for idx, score in passing:
            print(f"    [{score:.4f}] {sentences[idx]}")
