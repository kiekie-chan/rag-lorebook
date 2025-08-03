from typing import Optional, List
from mediawiki import MediaWiki
import mwclient
import wikitextparser as wtp
import re
from tqdm import tqdm
from FlagEmbedding import FlagModel 
import logging
import os
import numpy as np

from faiss_embed import FaissEmbed

class WikiParser:

    def __init__(self, base_url: str, db: FaissEmbed, embedder: FlagModel, max_pages: int):
        
        try:
            self.wiki = MediaWiki(url=f'https://{base_url}/api.php')
        except Exception as e:
            logging.info(f'It seems there is something wrong with the Wiki API: {e}')

        self.db = db
        self.embedder = embedder

        self.texts = []
       
        site = mwclient.Site(base_url, path='/')
        
        total_pages = sum(1 for _ in site.allpages()) 
        total_pages = min(max_pages, total_pages)
        count = 0

        for page in tqdm(site.allpages(), total=total_pages, desc='Parsing pages', unit='pages'):

            try:
                scrapped_page = self.scrap_page(str(page), page.name.split('/')[0].strip('"'))
                if scrapped_page:                 
                    self.index_page(scrapped_page)
                    
            except Exception as e:
                logging.info(f'Cannot parse the page: {e}')
            
            count += 1
            if count >= total_pages:
                break
            
        
        self.db.save()
        self.save_texts()

    def scrap_page(self, page: str, page_name: str) -> Optional[List[str]]:
        '''Selects text, cleans it and returns dictionary with subtags and context'''
        
        page = self.wiki.page(page)

        parsed = wtp.parse(page.wikitext)
        text = parsed.plain_text() 

        chapters = re.split(r'==+([^=]+)==+', text.strip())

        scrapped_page = []
        for subtag, content in zip(chapters[1::2], chapters[2::2]):
            content = content.strip()

            if content:
                subtag = page_name + ' ' + subtag.strip()
                content = ' '.join(re.sub(r'[\n*]+', ' ', content).split())
                scrapped_page.append(f'{subtag} : {content}')

        return scrapped_page if scrapped_page else None
    
    def index_page(self, scrapped_page: list) -> None:
        ''' Indexes scrapped page into faiss by chapters'''

        for chapter in scrapped_page:
            embedding = self.embedder.encode(chapter)
            self.db.add(embedding)
            self.texts.append(chapter)
    
    def save_texts(self):
        
        embeddings_dir = os.path.dirname(self.db.index_path)
        os.makedirs(embeddings_dir, exist_ok=True)
        texts_path = os.path.join(embeddings_dir, 'texts.npy')
        np.save(texts_path, np.array(self.texts))
