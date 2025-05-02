import io, os, sys, json
import numpy as np
from collections import defaultdict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import easyocr
import re

# Get inputs (image path and predefined answer)
image_path = sys.argv[1]
predefined_answer = sys.argv[3]

# OCR with EasyOCR
reader = easyocr.Reader(['en'])  # Supports English
results = reader.readtext(image_path)

if not results:
    print("HELLO")
    print(json.dumps({"error": "No text detected"}))
    exit()

# Extract words with bounding box center coordinates (x, y)
text_data = []
for bbox, text, _ in results:
    x = np.mean([point[0] for point in bbox])
    y = np.mean([point[1] for point in bbox])
    text_data.append((text, x, y))

# Sort by y (line-wise), then x (left-to-right)
text_data.sort(key=lambda t: (t[2], t[1]))
line_threshold = 15
lines = defaultdict(list)
current_line = []
last_y = text_data[0][2]
for word, x, y in text_data:
    if abs(y - last_y) < line_threshold:
        current_line.append((word, x, y))
    else:
        lines[last_y].extend(sorted(current_line, key=lambda t: t[1]))
        current_line = [(word, x, y)]
    last_y = y
lines[last_y].extend(sorted(current_line, key=lambda t: t[1]))

# Load T5 model for correction
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def correct_line(text_line):
    input_ids = tokenizer.encode(f"correct: {text_line}", return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Reconstruct and correct text line-by-line
ordered_text = []
for line_y in sorted(lines.keys()):
    raw_line = " ".join([word for word, _, _ in lines[line_y]])
    corrected_line = correct_line(raw_line)
    ordered_text.append(corrected_line)

final_text = "\n".join(ordered_text)

# Semantic similarity with SBERT
sbert = SentenceTransformer('all-MiniLM-L6-v2')
corrected_emb = sbert.encode([final_text])
answer_emb = sbert.encode([predefined_answer])
similarity_score = cosine_similarity(corrected_emb, answer_emb)[0][0]

# Output JSON
print(json.dumps({
    "corrected_text": final_text,
    "similarity_score": round(similarity_score * 100, 2)
}))
