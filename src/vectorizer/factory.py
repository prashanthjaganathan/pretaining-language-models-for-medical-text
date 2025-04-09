from enum import Enum
from .base import Embedding
from .bert import BERTModel
from .robert import RoBERTaModel
from .bio_wordvec import BioWordVecModel
from .google_news_word2vec import GoogleNewsModel
from .trainable import TrainableEmbedding
from .bio_bert import BioBERTModel

class EmbeddingType(Enum):
    BERT = "bert"
    ROBERTA = "roberta"
    BIO_WORDVEC = "bio_wordvec"
    GOOGLE_NEWS = "google_news"
    TRAINABLE = "trainable"
    BIO_BERT = "bio_bert"

class EmbeddingFactory:
    """Factory class that maps embedding types to specific Embedding classes."""
    
    embedding_map = {
        EmbeddingType.BERT: BERTModel,
        EmbeddingType.ROBERTA: RoBERTaModel,
        EmbeddingType.BIO_WORDVEC: BioWordVecModel,
        EmbeddingType.GOOGLE_NEWS: GoogleNewsModel,
        EmbeddingType.TRAINABLE: TrainableEmbedding,
        EmbeddingType.BIO_BERT: BioBERTModel
    }

    @staticmethod
    def get_embedding(embedding_type: str, **kwargs) -> Embedding:
        """
        Returns an instance of the appropriate embedding model.

        Parameters:
        - embedding_type: Type of embedding to use
            Options: 'bert', 'roberta', 'bio_wordvec', 'google_news', 'trainable', 'bio_bert'

        Kwargs:
            For BERT/RoBERTa:
                - model_name (str): Name of the pretrained model
                    Options: 'bert-base-uncased', 'dmis-lab/biobert-v1.1',
                            'roberta-base', 'allenai/biomed_roberta_base'
                
            For BioWordVec:
                - model_path (str): Path to pretrained BioWordVec model

            For BioBERTModel:
                - model_name (str): Eg: dmis-lab/biobert-base-cased-v1.1
                
            For GoogleNews Word2Vec:
                - model_path (str): Path to pretrained Word2Vec model
                
            For Trainable:
                - tokenized_corpus (List[List[str]]): Training corpus of tokenized texts
                - algorithm (str): Algorithm type ('word2vec', 'fasttext', 'tfidf')
                - vector_size (int, optional): Embedding dimension. Default: 100
                - window (int, optional): Context window size. Default: 5
                - min_count (int, optional): Min word frequency. Default: 2

        Returns:
            Embedding: The corresponding embedding model
        """
        try:
            embedding_type = EmbeddingType(embedding_type.lower())
        except ValueError:
            raise ValueError(f"Unknown embedding type: {embedding_type}. "
                           f"Available types: {[e.value for e in EmbeddingType]}")

        embedding_class = EmbeddingFactory.embedding_map.get(embedding_type)
        if not embedding_class:
            raise ValueError(f"No implementation for embedding type: {embedding_type}")

        # Validate required parameters
        if embedding_type in [EmbeddingType.BERT, EmbeddingType.ROBERTA, EmbeddingType.BIO_BERT]:
            required_params = {'model_name'}
            
        elif embedding_type in [EmbeddingType.BIO_WORDVEC, EmbeddingType.GOOGLE_NEWS]:
            required_params = {'model_path'}
            
        elif embedding_type == EmbeddingType.TRAINABLE:
            required_params = {'tokenized_corpus', 'algorithm'}
            
            kwargs.setdefault('vector_size', 100)
            kwargs.setdefault('window', 5)
            kwargs.setdefault('min_count', 2)

        missing_params = required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters for {embedding_type}: {missing_params}")

        return embedding_class(**kwargs)