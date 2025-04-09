from enum import Enum
from importlib import reload
import os
from pathlib import Path
import gensim
import h5py
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
from gensim.models import Word2Vec, FastText
from .base import Embedding
from scipy.sparse import save_npz, load_npz
from tqdm import tqdm

reload(gensim)

class EmbeddingAlgorithm(Enum):
    WORD2VEC = "word2vec"
    TF_IDF = "tfidf"
    FASTTEXT = "fasttext"

class TrainableEmbedding(Embedding):
    """
    A configurable embedding model that can be trained on a corpus using different algorithms.
    
    Supported algorithms:
    - Word2Vec
    - FastText
    - TF-IDF
    
    Usage:
        embedding_model = TrainableEmbedding(
            tokenized_corpus=texts,
            algorithm="fasttext", 
            vector_size=100, 
            window=5,
            min_count=2
        )
        embedding_model.train()
        embeddings = embedding_model.embed(text)
    """
    
    def __init__(self,
                 tokenized_corpus: List[List[str]],  # Each document is a list of tokens
                 algorithm: str,
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 2,
                 special_tokens: Optional[List[str]] = None):
        """
        Initialize the trainable embedding model.
        
        Args:
            tokenized_corpus: List of documents, where each document is a list of tokens.
            algorithm: The embedding algorithm to use ('word2vec', 'glove', 'fasttext', 'tfidf')
            vector_size: The dimensionality of the embeddings
            window: The window size for Word2Vec, GloVe, or FastText
            min_count: Minimum number of occurrences for a word to be considered
            special_tokens: List of special tokens for TF-IDF, if necessary
        """
        try:
            self.algorithm = algorithm.lower()
        except ValueError:
            raise ValueError(f"Algorithm must be one of: {[algo.value for algo in EmbeddingAlgorithm]}")



        # Create model directory
        self.model_dir = Path('trained_models/embeddings/trained')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.special_tokens = special_tokens or ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.aggregation_method = 'average'
        
        if isinstance(tokenized_corpus, list):
            self.tokenized_corpus = tokenized_corpus
        else:
            raise TypeError("corpus must be a list of tokenized documents")

        self.model = None
        self.vectorizer = None  # For TF-IDF
        self.trained = False

    def is_trained(self) -> bool:
        """Check if the model has been trained and saved"""
        if self.algorithm == EmbeddingAlgorithm.TF_IDF:
            path = f'trained_models/embeddings/trained/trained_{self.algorithm}.npz'
        else:
            path = f'trained_models/embeddings/trained/trained_{self.algorithm}.model'

        if os.path.exists(path):
            return True
        
        return False
        
    def train(self) -> None:
        """Train the embedding model on the corpus"""
        print(f'Training {self.algorithm} model')
        if self.algorithm == EmbeddingAlgorithm.TF_IDF.value:
            self.train_tfidf()
        elif self.algorithm == EmbeddingAlgorithm.WORD2VEC.value:
            self.train_word2vec()
        elif self.algorithm == EmbeddingAlgorithm.FASTTEXT.value:
            self.train_fasttext()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        self.trained = True
        self.save_model()

        
    def train_word2vec(self) -> None:
        """Train Word2Vec embeddings on the corpus"""
        # Initial model training on a subset of data
        self.model = Word2Vec(
            self.tokenized_corpus[:100000],  # Start with a smaller corpus
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=6,
        )
        self.model.train(self.tokenized_corpus[:100000], total_examples=100000, epochs=5)

        # Incrementally train on the remaining corpus
        for start in tqdm(
            range(100000, len(self.tokenized_corpus), 100000),
            'Incremental Training', 
            len(self.tokenized_corpus)/100000
            ):
            end = min(start + 100000, len(self.tokenized_corpus))
            self.model.build_vocab(self.tokenized_corpus[start:end], update=True)
            self.model.train(self.tokenized_corpus[start:end], total_examples=end-start, epochs=5)

        print(f"Word2Vec model type: {type(self.model)}")
        
    
    def train_fasttext(self) -> None:
        """Train FastText embeddings on the corpus"""
        initial_subset = self.tokenized_corpus[:100000]  # Use a smaller subset of data

        self.model = FastText(
            sentences=initial_subset,
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count,
            workers=6
        )

        # Build vocabulary first
        self.model.build_vocab(initial_subset)

        # Train on the first subset
        self.model.train(initial_subset, total_examples=len(initial_subset), epochs=5)

        # Incrementally train on the remaining data in chunks
        chunk_size = 100000
        for start in tqdm(
            range(100000, len(self.tokenized_corpus), chunk_size), 
            desc='Training incrementally',
            total=len(self.tokenized_corpus) // chunk_size
        ):
            end = min(start + chunk_size, len(self.tokenized_corpus))
            chunk = self.tokenized_corpus[start:end]

            # Update vocabulary and train on the new chunk
            self.model.build_vocab(chunk, update=True)
            self.model.train(chunk, total_examples=len(chunk), epochs=5)



    def train_tfidf(self, output_file='embeddings/tfidf_embeddings.h5'):
        """Train TF-IDF vectorizer on the corpus and store reduced embeddings"""
        self.vectorizer = TfidfVectorizer()
        batch_size = 1000
        svd_initialized = False
        svd = TruncatedSVD(n_components=self.vector_size)
    
        # Prepare the corpus for training (tokenized documents to string format)
        documents = [''.join(tokens) for tokens in self.tokenized_corpus]
        
        # Create HDF5 file to store embeddings
        with h5py.File(output_file, 'w') as hf:
            # Create an empty dataset in HDF5 to store the reduced embeddings
            embeddings_dataset = hf.create_dataset(
                'tfidf_embeddings', 
                shape=(len(documents), 100),  # 100 dimensions after SVD
                dtype='float32',
                chunks=(batch_size, 100),  # Chunks for efficient storage
                compression="gzip"  # Optional: Compress data to save space
            )
            
            start = 0
            while start < len(documents):
                end = min(start + batch_size, len(documents))
                batch_docs = documents[start:end]
                
                # Fit the TF-IDF vectorizer on the current batch of documents
                batch_tfidf = self.vectorizer.fit_transform(batch_docs)
                
                # Apply TruncatedSVD (if not already initialized, fit it on the first batch)
                if not svd_initialized:
                    svd.fit(batch_tfidf)
                    svd_initialized = True
                else:
                    # If already initialized, just apply the transformation
                    batch_tfidf = svd.transform(batch_tfidf)
                
                # Save the reduced embeddings for this batch to the HDF5 dataset
                embeddings_dataset[start:end] = batch_tfidf
                
                # Update the starting point for the next batch
                start = end


    # def train_tfidf(self) -> None:
    #     """Train TF-IDF vectorizer on the corpus"""
    #     documents = [' '.join(tokens) for tokens in self.tokenized_corpus]
    #     # print(f'corpus: {self.tokenized_corpus}')
    #     # print(documents)
    #     self.vectorizer = TfidfVectorizer()
    #     self.model = self.vectorizer.fit_transform(documents)

        
    def save_model(self) -> None:
        """Save the trained model to disk"""
        if self.algorithm == EmbeddingAlgorithm.TF_IDF.value:
            path = self.model_dir / f'trained_{self.algorithm}.npz'
            save_npz(str(path), self.model)
            import pickle
            with open(self.model_dir / f'trained_{self.algorithm}_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
        else:
            path = self.model_dir / f'trained_{self.algorithm}.model'
            if self.model is None:
                raise ValueError("Model has not been trained yet, cannot save.")
            if isinstance(self.model, (Word2Vec, FastText)):  # Ensure the model type
                self.model.save(str(path))
            else:
                raise TypeError(f"Expected Word2Vec or FastText model, but got {type(self.model)}")
        print('Model saved')

    def load(self) -> None:
        """Load a trained model from disk"""
        if self.algorithm == EmbeddingAlgorithm.TF_IDF.value:
            path = self.model_dir / f'trained_{self.algorithm}.npz'
            self.model = load_npz(str(path))
            # Load the vectorizer
            import pickle
            with open(self.model_dir / f'trained_{self.algorithm}_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            path = self.model_dir / f'trained_{self.algorithm}.model'
            self.model = (gensim.models.FastText.load(str(path)) 
                         if self.algorithm == EmbeddingAlgorithm.FASTTEXT.value 
                         else gensim.models.Word2Vec.load(str(path)))


    def embed(self, tokenized_corpus: List[List[str]], label_names: set =  None, create_doc_embedding: bool = False):
        """
        Generate embeddings for a list of documents in the tokenized corpus.

        Args:
            tokenized_corpus: List of documents where each document is a list of tokens.

        Returns:
            Embeddings of all documents, each represented by a vector.
        """
        if not self.trained:
            if not any(self.model_dir.glob(f'trained_{self.algorithm}.*')):
                self.train()
            else:
                self.load()
                self.trained = True


        if self.algorithm == EmbeddingAlgorithm.TF_IDF.value:
            documents = [''.join(tokens) for tokens in tokenized_corpus]
            embeddings = self.vectorizer.transform(documents)
            return embeddings.toarray()
        
        else:
            abbr_idx = 0
            abbr_embedding = []
            all_embeddings = []
            # for doc in tqdm(tokenized_corpus, 'Vectorizing', len(tokenized_corpus)):
            for doc in tokenized_corpus:
                doc_embeddings = []
                for idx, token in enumerate(doc):
                    if not isinstance(token, str):
                        token = token.as_py()

                    if token in label_names:
                        abbr_idx = idx
                        if token in self.model.wv:
                            abbr_embedding = self.model.wv[token]
                        else:
                            print('Abbreviation embedding not found!')
                            abbr_embedding = np.zeros(self.vector_size)

                    if token in self.model.wv:
                        doc_embeddings.append(self.model.wv[token])
                    else:
                        doc_embeddings.append(np.zeros(self.vector_size))
                
                if create_doc_embedding:
                    if doc_embeddings:
                        doc_embedding = np.mean(doc_embeddings, axis=0)
                    else:
                        doc_embedding = np.zeros(self.vector_size)
                    all_embeddings.append(doc_embedding)

                else: 
                    all_embeddings.append(doc_embeddings)


            context_window = 50
            start = abbr_idx - context_window // 2
            end = abbr_idx + context_window // 2
            if start < 0:
                start = 0
            
            if end > len(all_embeddings[0]):
                end = len(all_embeddings[0])
            
            context_embedding = all_embeddings[0][start:end].copy()
            
            return all_embeddings[0], abbr_embedding, context_embedding