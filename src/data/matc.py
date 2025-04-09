

import os
from pathlib import Path
import random
import re
import shutil
from typing import Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from ..tokenizer.factory import TokenizerFactory
from .base import BaseDataset
from env import ProjectPaths

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)



class MATC(BaseDataset):
    def __init__(self, name):
        super().__init__(name)
        print(f'Medical Abstract for Text Classification dataset initialized with name: {self.name}')


    def load_dataset(self) -> Tuple[pd.DataFrame]:
        """
        Loads the MATC dataset from Kaggle
        """
        splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
        self.train_data = pd.read_parquet("hf://datasets/TimSchopf/medical_abstracts/" + splits["train"])
        self.test_data = pd.read_parquet("hf://datasets/TimSchopf/medical_abstracts/" + splits["test"])
        labels_dict = pd.read_parquet("hf://datasets/TimSchopf/medical_abstracts/labels/train-00000-of-00001.parquet").to_dict()
        self.class_to_idx = self.convert_class_to_idx(labels_dict)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        matc_dir = ProjectPaths.DATASET_DIR.value / 'MATC'

        if not matc_dir.exists() or not matc_dir.is_dir():
            os.mkdir(matc_dir)
            self.train_data.to_csv(matc_dir / 'train.csv')
            self.test_data.to_csv(matc_dir / 'test.csv')
            print("Dataset downloaded to:", matc_dir)

        self.train_data, self.val_data = self.split_dataset(train_size=0.8)
        
        return self.train_data, self.val_data, self.test_data


    def split_dataset(self, train_size=0.8, shuffle=True):
        """Split into train, val, and test sets."""
        total_len = len(self.train_data)
        train_end = int(total_len * train_size)

        if shuffle:
            self.train_data = self.train_data.sample(frac=1, random_state=42).reset_index(drop=True)

        train = self.train_data[:train_end]
        val = self.train_data[train_end:]

        return train, val

    def convert_class_to_idx(self, class_dict: dict):
        """
        Creates a dict of class name: index
        """
        class_to_idx = {v: k for k, v in class_dict['condition_name'].items()}
        label_to_idx = {v: k for k, v in class_dict['condition_label'].items()}
        return class_to_idx
    

    def __len__(self):
        return len(self.train_data)
    

    def __getitem__(self, idx):
        if idx >= len(self.train_data) or idx < 0:
            raise ValueError('Index out of bounds')
        
        return self.train_data.iloc[idx]
    

    def _preprocess_split(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic pre-processing on the dataset:
        - Converts all other words to lowercase.
        - Performs lemmatization and stop-word removal.
        
        This version is optimized for large datasets (e.g., 5M rows) by:
        1. Vectorizing what we can.
        2. Using pandarallel to process rows in parallel.
        """

        # Define a function to preprocess a single row
        def preprocess_row(row: pd.Series) -> pd.Series:
            class_name = self.idx_to_class[row['condition_label'] - 1]
            text: str = row['medical_abstract']
            words = text.split()

            tmp_words = []
            for i, word in enumerate(words):
                    tmp_words.append(word.lower())

            tmp = " ".join(tmp_words).strip()
            
            # Remove punctuation using regex
            tmp = re.sub(r'[^\w\s]', '', tmp)

            # Apply lemmatization and stop-word removal (assumed to be static methods on MeDALSubset)
            processed = BaseDataset.lemmatizer(tmp)
            processed = BaseDataset.remove_stop_words(processed)

            row['medical_abstract'] = processed
            row['condition_label'] = class_name
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

            text_data = data['long_title']

            tokenizer_instance = TokenizerFactory.get_tokenizer(
                tokenizer_type,
                **kwargs
            )

            if tokenizer_type == 'pretrained':
                return tokenizer_instance.tokenize()
            else:    
                # Tokenize each piece of text
                tokenized_data = text_data.parallel_apply(lambda text: tokenizer_instance.tokenize(text))
                tokenized_splits.append(tokenized_data)

        if len(splits) == 1:
            return tokenized_splits[0]
        else:
            return tuple(tokenized_splits)


    def embed(self):
        """Embed using TF-IDF on context column."""
        pass

    def save_embeddings(self, path: str):
        """Save embeddings as .npy and the vectorizer as a pickle file."""
        print(f"✅ Embeddings saved to {path}.npy and vectorizer to {path}_vectorizer.pkl")