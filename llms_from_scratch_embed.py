"""
    Word Embeddings with word2vec from Scratch in Python
    https://medium.com/@bradneysmith/word-embeddings-with-word2vec-from-scratch-in-python-eb9326c6ab7c
    
    
    osobny venv ze starymi bibliotekami
    
    $ python3 -m venv venvTransformers
    $ source source venvTransformers/bin/activate
    $ pip install scipy==1.10.1
    $ pip install gensim
"""

import gensim
import gensim.downloader as api

google_cbow = api.load('word2vec-google-news-300')

# api.load('word2vec-google-news-300', return_path=True)
# ~/gensim-data/word2vec-google-news-300/

king = google_cbow['king']
man = google_cbow['man']
woman = google_cbow['woman']

google_cbow.most_similar(king, topn=2)
king_result = google_cbow.most_similar(king-man+woman, topn=2)
print(king_result)


paris = google_cbow['paris']
france = google_cbow['france']
germany = google_cbow['germany']

paris_result = google_cbow.most_similar(paris-france+germany, topn=3)
print(paris_result)
