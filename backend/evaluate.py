import io, os, sys, json
import numpy as np
from google.cloud import vision
from collections import defaultdict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Manually set the credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:/Users/santh/Downloads/nimble-equator-456111-g9-8434372d9f3c.json"

# Get inputs (image path and predefined answer)
image_path = sys.argv[1]
predefined_answer = sys.argv[3]

# OCR with Google Vision
client = vision.ImageAnnotatorClient()
with io.open(image_path, "rb") as f:
    content = f.read()
image = vision.Image(content=content)
response = client.text_detection(image=image)
texts = response.text_annotations

if not texts:
    print(json.dumps({"error": "No text detected"}))
    exit()

# Extract words and sort lines
text_data = []
for text in texts[1:]:
    vertices = text.bounding_poly.vertices
    x = np.mean([v.x for v in vertices])
    y = np.mean([v.y for v in vertices])
    text_data.append((text.description, x, y))

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

ordered_text = []
for line_y in sorted(lines.keys()):
    ordered_text.append(" ".join([word for word, _, _ in lines[line_y]]))
final_text = "\n".join(ordered_text)

# Spell correction with T5
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
inputs = tokenizer.encode(f"{final_text}", return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Semantic similarity with SBERT
sbert = SentenceTransformer('all-MiniLM-L6-v2')
corrected_emb = sbert.encode([corrected_text])
answer_emb = sbert.encode([predefined_answer])
similarity_score = cosine_similarity(corrected_emb, answer_emb)[0][0]

# Output JSON
print(json.dumps({
    "corrected_text": corrected_text,
    "similarity_score": round(similarity_score * 100, 2)
}))
