import re
import shutil
from typing import Dict, List, Tuple, Union, Any
import torch.nn as nn

from ..tokenizer.factory import TokenizerFactory
from ..models.embedding.pretrained_embedding import PretrainedEmbedding
from ..models.embedding.custom_embedding import CustomEmbedding
from ..vectorizer.factory import EmbeddingFactory
# TODO: Add ruff pre-commit hooks
import spacy
from .base import BaseDataset
import kagglehub
import os
from env import ProjectPaths
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.corpus import stopwords
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

nltk.download('stopwords')


class MeDALSubset(BaseDataset):
    """
    Full MeDAL dataset is extremely large ~ 14GB.
    This class includes a subset of the MeDAL dataset ~ 5M entries.
    """
    def __init__(self, name):
        super().__init__(name)
        print(f'MeDAL dataset initialized with name: {self.name}')


    def load_dataset(self) -> Tuple[pd.DataFrame]:
        """
        Loads the MeDAL dataset from Kaggle
        """
        
        downloaded_path = kagglehub.dataset_download("xhlulu/medal-emnlp")
        print("Dataset downloaded to:", downloaded_path)

        medal_dir = ProjectPaths.DATASET_DIR.value / 'medal'

        if not medal_dir.exists() or not medal_dir.is_dir():
            items = os.listdir(downloaded_path)

            for item in tqdm(items, desc='Moving dataset to project dir', unit='item'):
                item_path = os.path.join(downloaded_path, item)
                if os.path.isdir(item_path):
                    # Move folders
                    shutil.move(item_path, medal_dir / item)
                else:
                    # Move files
                    shutil.move(item_path, medal_dir)

        self.path = medal_dir / 'pretrain_subset' # Points to pretrain_subset
        print(f"Dataset moved to: {ProjectPaths.DATASET_DIR.value}")

        self.train_data = pd.read_csv(self.path / 'train.csv')
        self.val_data = pd.read_csv(self.path / 'valid.csv')
        self.test_data = pd.read_csv(self.path / 'test.csv')
        self.data = pd.concat([self.train_data, self.val_data, self.test_data], ignore_index=True)
        self.class_to_idx = self.convert_class_to_idx(self.data)
        print(f'Total number of classes: {len(self.class_to_idx)}')

        return self.data, self.train_data, self.val_data, self.test_data

    def convert_class_to_idx(self, dataset: pd.DataFrame):
        """
        Creates a dict of class name: index
        """
        labels: set = set(dataset['LABEL'])
        class_to_idx: Dict[str, int] = {}

        for idx, label in enumerate(labels):
            class_to_idx[label] = idx

        return class_to_idx
    

    def split_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the dataset into an `DataFrame` objects and parses it into train, val, test sets.

        Returns:
        train_data, val_data, test_data
        """
        return self.train_data, self.val_data, self.test_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data) or idx < 0:
            raise ValueError('Index out of bounds')
        
        return self.data.iloc[idx]
    

    def _preprocess_split(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic pre-processing on the dataset:
        - Renames the LOCATION column to ABBREVIATION.
        - Uses the integer stored in the ABBREVIATION column (now preserved as 'abbr_index') 
            to replace the corresponding word in the TEXT with its abbreviation.
        - Converts all other words to lowercase.
        - Performs lemmatization and stop-word removal.
        
        This version is optimized for large datasets (e.g., 5M rows) by:
        1. Vectorizing what we can.
        2. Using pandarallel to process rows in parallel.
        """

        # Rename the column LOCATION to ABBREVIATION, and preserve the original abbreviation index
        data = data.rename(columns={'LOCATION': 'ABBREVIATION'})
        # Preserve the original abbreviation index in a new column, if not already present
        if 'abbr_index' not in data.columns:
            data['abbr_index'] = data['ABBREVIATION']

        # Define a function to preprocess a single row
        def preprocess_row(row: pd.Series) -> pd.Series:
            text: str = row['TEXT']
            words = text.split()
            # Use the saved abbreviation index
            abbr_index = int(row['abbr_index'])
            # Get the abbreviation from the words list
            try:
                abbreviation = words[abbr_index]
            except IndexError:
                abbreviation = ""
            # Overwrite the 'ABBREVIATION' column with the actual abbreviation
            row['ABBREVIATION'] = abbreviation

            # Build a new text by replacing the word at abbr_index with the abbreviation (as-is)
            # and lowercasing all other words.
            tmp_words = []
            for i, word in enumerate(words):
                if i == abbr_index:
                    tmp_words.append(abbreviation)
                else:
                    tmp_words.append(word.lower())
            tmp = " ".join(tmp_words).strip()
            
            # Remove punctuation using regex
            tmp = re.sub(r'[^\w\s]', '', tmp)

            # Apply lemmatization and stop-word removal (assumed to be static methods on MeDALSubset)
            processed = BaseDataset.lemmatizer(tmp)
            processed = BaseDataset.remove_stop_words(processed)

            row['TEXT'] = processed
            return row

        # Use pandarallel to apply the row function in parallel
        data = data.parallel_apply(preprocess_row, axis=1)
        return data 


    def preprocess(self, splits=['train']) -> Union[pd.DataFrame, 
                                                     Tuple[pd.DataFrame, pd.DataFrame], 
                                                     Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Performs basic pre-processing based on the splits passed.

        Parameters:
        splits (list): A list of dataset splits to pre-process. It can contain any of the following:
            'train', 'valid', 'test'. The specified splits will be pre-processed accordingly.
            By default, it processes the 'train' split.

        Example:
        >>> preprocess(['train', 'valid']) will pre-process both the 'train' and 'valid' datasets.
        """

        processed_splits = []

        if len(splits) > 3 or len(splits) < 1:
            raise ValueError('Invalid number of splits passed!')

        for split in splits:
            if split == 'train':
                data = self.train_data
                processed_split = self._preprocess_split(data)
                self.train_data = processed_split  

            elif split == 'valid':
                data = self.val_data
                processed_split = self._preprocess_split(data)
                self.val_data = processed_split

            elif split == 'test':
                data = self.test_data
                processed_split = self._preprocess_split(data)
                self.test_data = processed_split

            else:
                raise ValueError('Invalid split passed. Refer to func. documentation.')

            processed_splits.append(processed_split)

        if len(splits) == 1:
            return processed_splits[0]
        return tuple(processed_splits)
        

    def tokenize(self, tokenizer_type: str, splits = ['train'], **kwargs) -> Union[List[str], 
                                                                                         Tuple[List[str], List[str]], 
                                                                                         Tuple[List[str], List[str], List[str]],
                                                                                         ]:
        """
        Tokenizes the dataset based on the specified tokenizer type and the given splits.

        Parameters:
        - tokenizer_type (str): The type of tokenization to apply. Choices are:
            - 'whitespace': Tokenizes based on whitespace
            - 'characters': Tokenizes into individual characters
            - 'nltk': Uses NLTK's word_tokenize
            - 'pretrained': Uses HuggingFace pretrained tokenizer
            - 'trainable': Uses trainable tokenizer (BPE or WordPiece)

        - splits (list): Dataset splits to tokenize ['train', 'valid', 'test']
        
        Kwargs for different tokenizer types:
            For pretrained:
                - pretrained_model (str): Name of the pretrained model
                    Options: 'google-bert/bert-base-uncased', 
                             'dmis-lab/biobert-v1.1', 
                             'neuml/pubmedbert-base-embeddings'
            
            For trainable:
                - corpus: Train data for the trainable algorithm ('bpe' or 'wordpeiece')
                - algorithm (str): Tokenization algorithm ('bpe' or 'wordpiece')
                - vocab_size (int, optional): Maximum vocabulary size. Default: 30000
                - min_frequency (int, optional): Minimum token frequency. Default: 2
                - special_tokens (List[str], optional): Special tokens to include

        Returns:
            Tokenized text for selected splits (single list or tuple of lists)
        """


        if len(splits) > 3 or len(splits) < 1:
            raise ValueError('Invalid number of splits passed!')

        tokenized_splits = []

        for split in splits:
            if split == 'train':
                data = self.train_data
            elif split == 'valid':
                data = self.val_data
            elif split == 'test':
                data = self.test_data
            else:
                raise ValueError('Invalid split passed. Refer to func. documentation.')

            text_data = data['TEXT']
            abbreviation = data['ABBREVIATION']

            tokenizer_instance = TokenizerFactory.get_tokenizer(
                tokenizer_type,
                **kwargs
            )

            if tokenizer_type == 'pretrained':
                return tokenizer_instance.tokenize()
            else:    
                # Tokenize each piece of text
                tokenized_data = text_data.parallel_apply(lambda text: tokenizer_instance.tokenize(text))
                
                # Zip tokenized text with the original abbreviation (string)
                tokenized_splits.append(tuple(zip(tokenized_data, abbreviation)))


        if len(splits) == 1:
            return tokenized_splits[0]
        else:
            return tuple(tokenized_splits)
        

    def embed(self, embedding_type: str, splits=['train'], tokenized_data=None, **kwargs):
        """
        Generate embeddings for the dataset using specified embedding type.
        
        Parameters:
        - embedding_type (str): Type of embedding to use
            Options: 'bert', 'roberta', 'bio_wordvec', 'google_news', 'trainable'
        - splits (list): Dataset splits to embed ['train', 'valid', 'test']
        - tokenized_data: Optional pre-tokenized data. If None, will use raw text.
        
        Kwargs:
            Passed to the embedding model. See EmbeddingFactory.get_embedding for details.
        
        Returns:
            Embeddings for selected splits (single array or tuple of arrays)
        """
        if len(splits) > 3 or len(splits) < 1:
            raise ValueError('Invalid number of splits passed!')

        embedded_splits = []
        
        # Get data from all required splits
        split_data = []
        split_abbr = []
        split_ids = []
        
        print('going to split data')
        for split in splits: 
            if tokenized_data is not None:
                if split == 'train':
                    data = tokenized_data
                    ids = self.train_data['ABSTRACT_ID']
                    abbr = self.train_data['ABBREVIATION']
                elif split == 'valid':
                    data = tokenized_data
                    abbr = self.val_data['ABBREVIATION']
                    ids = self.val_data['ABSTRACT_ID']
                elif split == 'test':
                    data = tokenized_data
                    ids = self.test_data['ABSTRACT_ID']
                    abbr = self.test_data['ABBREVIATION']
                else:
                    raise ValueError('Invalid split passed.')
                
            else:
                if split == 'train':
                    data = self.train_data['TEXT']
                    abbr = self.train_data['ABBREVIATION']
                    ids = self.train_data['ABSTRACT_ID']
                elif split == 'valid':
                    data = self.val_data['TEXT']
                    abbr = self.val_data['ABBREVIATION']
                    ids = self.val_data['ABSTRACT_ID']
                elif split == 'test':
                    data = self.test_data['TEXT']
                    ids = self.test_data['ABSTRACT_ID']
                    abbr = self.test_data['ABBREVIATION']
                else:
                    raise ValueError('Invalid split passed.')
                

            split_data.append(data.tolist())
            split_abbr.append(abbr.tolist())
            split_ids.append(ids.tolist())


        embedding_model = EmbeddingFactory.get_embedding(
            embedding_type=embedding_type,
            **kwargs
        )
        
        for data, abbr, id in zip(split_data, split_abbr, split_ids, splits):
            if embedding_type == 'bio_bert':
                embeddings = embedding_model.embed(data, abbr, id)
            elif tokenized_data is not None:
                print('received tokenized data')
                embeddings = embedding_model.embed(tokenized_data, abbr)
            else:
                embeddings = embedding_model.embed(data)
            embedded_splits.append(embeddings)

        self.save_embeddings(embedded_splits, splits)

        if len(splits) == 1:
            return embedded_splits[0]
            
        return tuple(embedded_splits)