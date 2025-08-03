# RAG Lorebook with Fandom Wiki and FAISS

## Setup

**1. Install requirements:**
```bash
pip install -r requirements.txt
```
**2. Configure `.env`:**
```python
BASE_URL=*your_fandom*.fandom.com
EMBEDDER_MODEL=your_model_from_FlagEmbedding
EMBEDDING_DIM=your_embedding_dim
EMBEDDINGS_DIR=your_directory_to_store_index
MAX_PAGES=max_pages_for_your_wiki
```
**3. Do parsing & indexing:**
```bash
python parsing_and_indexing.py
```
And wait for it to end.

**3. Inference command:**
```bash
python inference.py
```
