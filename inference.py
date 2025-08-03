import os
import numpy as np
import faiss
from FlagEmbedding import FlagModel
from dotenv import load_dotenv
import readline 
import logging
import torch
import warnings

warnings.filterwarnings('ignore')


load_dotenv()

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%d.%m.%Y %H:%M:%S',
    )

    logger = logging.getLogger(__name__)
    logger.info('Starting inference')

    embeddings_dir = os.getenv('EMBEDDINGS_DIR', default='data')
    embedder_model = os.getenv('EMBEDDER_MODEL', default='BAAI/bge-small-en')
    embed_dim = int(os.getenv('EMBEDDING_DIM', default=384))

    index_path = os.path.join(embeddings_dir, 'faiss.index')
    texts_path = os.path.join(embeddings_dir, 'texts.npy')

    try:
        index = faiss.read_index(index_path)
        texts = np.load(texts_path, allow_pickle=True)
        logging.info(f'FAISS index and texts loaded')
    except Exception as e:
        logging.info(f'Was not able to load texts of faiss index: {e}')


    embedder = FlagModel(embedder_model,
                         query_instruction_for_retrieval='Represent this sentence for searching relevant passages:',
                         normalize_embeddings=True)
    logging.info(f'Embedder model {embedder_model} loaded')

    if torch.cuda.is_available():
        embedder.model = embedder.model.to('cuda')

    print("\nEnter your query:")
    try:
        while True:
            query = input('> ').strip()
            if not query:
                continue

            embedding = embedder.encode([query])[0]

            D, I = index.search(embedding[np.newaxis, :], k=1)

            score = D[0][0]
            best_idx = I[0][0]

            print('\nMost similar text:')
            print(texts[int(best_idx)])
            print(f'\nDistance: {score}')

    except KeyboardInterrupt:
        print('\nInference stopped')
