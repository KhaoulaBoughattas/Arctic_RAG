from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  
import json
import os

# Read the cleaned text file
with open("data/texte_nettoye.txt", "r", encoding="utf-8") as f:
    texte = f.read()

# Initialize the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Create document chunks
documents = splitter.create_documents([texte])

# Save chunks to JSON
os.makedirs("data", exist_ok=True)
chunk_data = [
    {
        "page_content": doc.page_content,
        "metadata": {"chunk_id": i + 1, "source": "texte_nettoye.txt"}
    }
    for i, doc in enumerate(documents)
]

with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
print(f"✅ {len(chunk_data)} chunks saved to data/chunks.json")