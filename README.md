# pertaining-language-models-for-medical-text
This repository focuses on pretraining language models on the MeDAL dataset for medical text, utilizing multiple architectures to enhance performance for medical NLP tasks.

Paper [Link](https://arxiv.org/pdf/2012.13978)

Here, we will be focusing on pretraining the MeDAL dataset using an LSTM + Self-Attention architecture.

## Getting Started
1. Clone the repository

2. To use the scispacy `en_core_sci_sm` model for lemmantizing as it is trained on biomedical data with the ~100k vocabulary. Run

```bash
python -m spacy download en_core_web_sm

```

3. How to use Embeddings
```python
# Initialize dataset
medal_dataset = MeDALSubset('MeDAL')
medal_dataset.load_dataset()

# Tokenize data (if needed)
tokenized_data = medal_dataset.tokenize('nltk', splits=['train'])

# Generate embeddings using different models:

# 1. Using BioBERT
bert_embeddings = medal_dataset.embed(
    'bio_bert', 
    splits=['train'],
    model_name='dmis-lab/biobert-base-cased-v1.1'
    )

# 2. Using BioWordVec
bio_embeddings = medal_dataset.embed(
    embedding_type='bio_wordvec',
    splits=['train'],
    model_path='trained_models/embeddings/pretrained/bio_wordvec.bin',
    tokenized_data=tokenized_data  # Pass tokenized data
)

# 3. Using trainable embeddings
trainable_embeddings = medal_dataset.embed(
    embedding_type='trainable',
    splits=['train'],
    tokenized_corpus=tokenized_data,
    algorithm='fasttext',
    vector_size=100
)
```