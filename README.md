# NLP Labs
## Description
This repository contains the Labs done during the course of Natural Language Processing

### Author: Ahmed Samady
### Supervised by: Lotfi El Aachak

## Get Started

To start, clone this branch of the repo onto your local machine:
```bash
git clone -b main --single-branch [https://github.com/Samashi47/NLP_Labs]
```
create a virtual environment in the repository by typing the followwing command:
```bash
python -m venv /path/to/repo/on/your/local/machine
```
After cloning the project and creating your venv, activate the venv by:

```bash
.venv\Scripts\activate
```
You can run the following command to install the dependencies:
```bash
pip3 install -r requirements.txt
```
## [Lab 1](https://github.com/Samashi47/NLP_Labs/tree/main/Lab1)
Lab 1 focuses on scraping a website with Arabic text, storing in a MongoDB database, NLP Pipeline (Tokenization, Normalization, stemming, Lemmatization and stopwords removal), PoS Tagging, and NER.
## [Lab 2](https://github.com/Samashi47/NLP_Labs/tree/main/Lab2)
Lab 2 focuses on two main areas. The first is on RegEx and rule-based NLP, where we attempted to generate a bill from a sentence with a specific pattern. The second part, on the other hand, centers around word embeddings. In this section, we explored various techniques, including One-Hot Encoding, Bag of Words, TF-IDF, Word2Vec (both CBoW and Skip Gram), FastText, and GloVe. Additionally, we visualized the encoded vectors using t-SNE for dimensionality reduction. This allowed us to gain insights into the differences between these methods and capture the semantic relationships among words in our corpus.
## [Lab 3](https://github.com/Samashi47/NLP_Labs/tree/main/Lab3)
Lab 3 focuses on language modeling, which includes regression for short answer grading and sentiment analysis classification. in this lab we established an NLP preprocessing pipeline, word embeddings and multiple models evaluation.
## [Lab 4](https://github.com/Samashi47/NLP_Labs/tree/main/Lab4)
Lab 4 focuses on classification regression and transformer (text generation) and BERT (text classification). in this lab we scraped articles from Wikipedia, established an NLP pipeline, embeddings, language modeling, and evaluated the models. We also generated text using GPT2 and classified text using BERT.