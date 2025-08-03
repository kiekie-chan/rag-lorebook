import logging
from scrape import WikiParser
from faiss_embed import FaissEmbed
import os
from FlagEmbedding import FlagModel 
from dotenv import load_dotenv
import os
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
    logger.info('Starting execution')

    try:
        fandom_url = os.getenv('BASE_URL')
        embedder_model = os.getenv('EMBEDDER_MODEL', default='BAAI/bge-small-en')
        embed_dim = os.getenv('EMBEDDING_DIM', default=384)
        embeddings_dir = os.getenv('EMBEDDINGS_DIR', default='data')
        max_pages = os.getenv('MAX_PAGES', default=100000)
    except Exception as e:
        logging.info('Env is not configured or configured wrong')
    
    database = FaissEmbed(dim=int(embed_dim), embeddings_dir=embeddings_dir)
    logger.info('Vector database created')

    embedder = FlagModel(embedder_model, 
                        query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')
    
    if torch.cuda.is_available():
        embedder.model = embedder.model.to('cuda')
    logger.info('Embedder model created')

    parser = WikiParser(base_url=fandom_url, 
                        db=database, 
                        embedder=embedder, 
                        max_pages=int(max_pages))
    logger.info('Parsing finished')
    

