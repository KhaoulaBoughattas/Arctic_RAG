from sentence_transformers import SentenceTransformer
import json, math, tqdm
import torch

# 1️⃣ Charger les chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2️⃣ Choisir device
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SentenceTransformer("BAAI/bge-m3", device=device)

# 3️⃣ Encodage par batch
batch_size = 32
all_embeddings = []

for i in tqdm.tqdm(range(0, len(data), batch_size), desc="Embedding"):
    batch_texts = [item["page_content"] for item in data[i:i+batch_size]]
    batch_embs  = model.encode(batch_texts,
                               normalize_embeddings=True,
                               batch_size=batch_size,
                               show_progress_bar=False)
    all_embeddings.extend(batch_embs)

# 4️⃣ Ajouter aux objets et sauvegarder
for item, emb in zip(data, all_embeddings):
    item["embedding"] = emb.tolist()

with open("data/chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ {len(data)} embeddings sauvegardés.")
