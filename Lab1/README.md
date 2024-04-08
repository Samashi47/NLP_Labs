# Lab 1
**Author :** ***Ahmed Samady***\
**Supervised by :** ***Pr. Lotfi El Aachak***\
**Course :** ***NLP***\
This directory will contain everything that concerns Lab 1 of the NLP course.
You can find the notebook [here](https://github.com/Samashi47/NLP_Labs/blob/main/Lab1/lab1.ipynb).
## 1. Introduction
Since we had the choice to scrape any website that we would like on the condition that it has to be in Arabic, I chose to scrape the Moroccan news website `hespress.com` for some articles in the economy category.
## 2. Scraping & storing to a MongoDB Database
For the scraping, I chose to use `BeautifulSoup` as an **HTML** parser. Upon retrieving the **HTML** page that contains the links to the articles, only 12 articles are loaded, if you need more articles you need to scroll down the page, that's why in this test drive (Lab) we'll be only scraping these 12 articles, as we'll only use one of the articles (1st one) in the [notebook](https://github.com/Samashi47/NLP_Labs/blob/main/Lab1/lab1.ipynb).\
As mentioned above we first retrieve the links and article titles from the economy category webpage that are placed in a certain `div` in the webpage, and store these links and titles in a `JSON` file, which I called [`hess_article_links.json`](https://github.com/Samashi47/NLP_Labs/blob/main/Lab1/data/hess_article_links.json). Furthermore, we send requests to each link in the `JSON` file to retrieve its **HTML** code, consequently retrieve the article contents which are also located in a certain `div` (we pinpoint these `div`s by using the inspect element feature in your preferred browser), after this we also store the article contents in a `JSON` file called [`hess_article_content.json`](https://github.com/Samashi47/NLP_Labs/blob/main/Lab1/data/hess_article_content.json).\
We finally store these two `JSON` documents in a local MongoDB Database called `NLP_lab1` and two collections in this DB are named after the two files to be uploaded.
## 3. NLP Pipeline
This section we'll discuss the choices made in the Arabic NLP piepline.
### 3.1 Tokenization
Tokenization is fairly straightforward in Arabic, as it doesn't differ from other languages (e.g. English).\
I chose to use a tokenizer from the `pyarabic` [1] library, there are two ways to tokenize your text, either by word tokens or sentence tokens, I tried both in the [notebook](https://github.com/Samashi47/NLP_Labs/blob/main/Lab1/lab1.ipynb) for comparison.\
We can take the following as an example:
```python
word tokenization : ['ومقارنها', 'بالسنة', 'الماضية', '،']
sentence tokenization : ['ومقارنها بالسنة الماضية،']
```
### 3.2 Removing punctuation
Removing punctuation is a crucial step in the NLP pipeline as it cleans the text, reducing unnecessary computation. However, Arabic has some special characters for punctuation, such as the comma `،` and quotes `”`, which is why I included an Arabic punctuation string to handle punctuation removal.
```python
ar_punct = ''')(+`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”،.”…“–ـ”.'''
```
Here's an example for punctuation removal:
```python
Tokenized text with punctuation : ['ومقارنها', 'بالسنة', 'الماضية', '،']
Tokenized text without punctuation: ['ومقارنها', 'بالسنة', 'الماضية']
Sentence tokenized text with punctuation: ['ومقارنها بالسنة الماضية،']
Sentence tokenized text without punctuation: ['ومقارنها بالسنة الماضية']
```
### 3.3 Removing stopwords
Stopwords reomval saves computation and keeping them doesn't provide much information in the NLP process, that's why it is recommended to eliminate them. Removing them is a simple process, simlified by the `NLTK` library which includes a corpus for Arabic stopwords.
Here's an example of the output:
```python
Tokenized text with stopwords: ['في', 'افتتاحية', 'النشرة', 'الفصلية', 'الأولى', 'ضمان', 'الاستثمار', 'لسنة', '2024']
Tokenized text without stopwords: ['افتتاحية', 'النشرة', 'الفصلية', 'الأولى', 'ضمان', 'الاستثمار', 'لسنة', '2024']
Sentence tokenized text with stopwords: 'في افتتاحية النشرة الفصلية الأولى ضمان الاستثمار لسنة 2024'
Sentence tokenized text without stopwords: [' افتتاحية النشرة الفصلية الأولى ضمان الاستثمار لسنة 2024']
```
### 3.4 Converting numbers to arabic words
Converting numbers to Arabic words is an optional step in this pipeline as they are considered as stopwords. Due to this, I deleted them from the token list even though I had converted them from numbers to Arabic word numbers.\
```python
number in token list : ['2023']
number in words in the token list : ['ألفان و ثلاث و عشرون']
```
### 3.5 Normalization
Normalization in Arabic differs from other languages it includes diacritization, stripping tatweel, normalizing Hamza (e.g. `أ` to `ء`), normalizing Lam Alef, normalizing Teh Marbuta and Alef Maksura.\
Here's an example of the output:
```python
Original token list: ['العربية', 'مؤشري']
Normalized token list: ['العربيه', 'مءشري']
```
## 4. Stemming
Stemming is an important step, it provides the root of a word but can sometimes give words without meaning (e.g. stem of `وكاله` is `ال`), this problem persists even in Arabic NLP. I chose to use an Arabic stemmer from the `tashaphyne`[2] library.\
Here's an example of the output:
```python
Origial token list: ['العربيه', 'مءشري', 'فيتش']
Stemmed token list: ['عربيه', 'مءشر', 'تش']
```
## 5. Lemmatization
Lemmatization is similar to stemming, but provides more meaningful words (e.g. lemma of `الشركات` is `شركة`) which may be useful in use cases where meaning is important. However lemmatizers are more computationally expensive than stemmers. I chose to use an Arabic lemmatizer from the `qalsadi`[3] library.\
Here's an example of the output:
```python
Origial token list: ['الشركات', 'متعدده', 'الجنسيات', 'ومءسسات']
Lemmatized token list: ['شركة', 'متعدد', 'جنس', 'ومءسسات']
```
## 6. Stemming and lemmatization comparison
As mentionned before, the key difference between Stemming and lemmatization is that lemmatizers provide more meaningful roots of words. However lemmatizers are computationally expensive, and we can see the difference even in our small use case (300-500 words/article). In addition, if we thoroughly search through the lemmatized token list, we can notice the difference in word meaning between stemming and lemmatization.
## 7. PoS Tagging
PoS tagging is a process where we tag word are given a particular part of speech (adverb, adjective, verb, etc.). Research done in English PoS tagging is quite advanced compared to Arabic, Arabic is quite a complex language and little research was done in this field, however we'll try to provide two approaches to solve this problem.
### 7.1 ML approach
Creating a model for PoS tagging is quite a difficult a task to do, espacially with the lack of datasets to train the model, and the huge number of words in the Arabic language, which is why I sticked to using the `Farasa` [4] library. This is a `Java` package in origin, but it has a Python wrapper. We'll stick to using this tagger on the original text because it takes 30+ minutes to use on the token list and we can see that the library is fairly heavy from the way it calls the jars each time.\
The PoS tagger seems quite accurate, let's see an example of the output:
```python
Tagged text: عن/PREP حلول/NOUN-FP ال+ مغرب/DET+NOUN-MS
```
### 7.2 RegEx approach
The Arabic language is quite a complex language with a huge number of patterns for verbs, nouns, adjectives, etc..., with some words not necessarily having a patterns and others change meaning if tashkeel changes[5]. However I tried to develop a small regex PoS tagger from the rules provided by Mr. Hjouj [6], the tagger is simple, as it only has NOUN and VERB classes. the arabic letters had to be converted to their ascii to be implemented in the regex pattern matcher, to extract the *Awzan* and patterns. The Tagger was given the lemmatized token list because of the advantages that lemmatization provides.\
In conclusion, after some evaluation the tagger seems to be not that accurate, however it does get some words correctly tagged, especially for th ones tha follow the patterns that it was supposed to handle.\
Here's an example of the output:
```python
Lemmatized token list: ['عالم', 'دول', 'العربيه', 'مءشري', 'فيتش']
Custom POS Tagged: ['NOUN', 'VERB', 'VERB', 'VERB', 'NOUN']
```
## 8. Named Entity Recognition
We finally applied the NER Tagger by `Ferasa`, I should note that the `Ferasa NER Tagger` is quite heavy on the machine.
However it seems to be accurate in detecting entities in the text.\
Here's an example of the output:
```python
NER Tagged text: 'أعلنت/O المؤسسة/B-ORG العربية/I-ORG لضمان/I-ORG الاستثمار'
```
## 9. What I learned
To conclude what I have learned during this Lab, is that Arabic is far more complex than its latin counterparts, tashkeel plays a huge role in determining the meaning of a word, also handling Arabic text seems to be quite tedious, in addition the lack of research done in the field of Arabic NLP, consequently, the libraries for Arabic NLP seem to be quite limited and small in number.
## 10. References
[1] Zerrouki, T., (2023). PyArabic: A Python package for Arabic text. Journal of Open Source Software, 8(84), 4886, https://doi.org/10.21105/joss.04886\
[2] Alkhatib, R. M., Zerrouki, T., Shquier, M. M. A., & Balla, A. (2023). Tashaphyne0.4: A new arabic light stemmer based on rhyzome modeling approach. Information Retrieval Journa, 26(14). doi: https://doi.org/10.1007/s10791-023-09429-y\
[3] T. Zerrouki, Qalsadi, Arabic mophological analyzer Library for python.,  https://pypi.python.org/pypi/qalsadi/\
[4] MagedSaeed. (n.d.). GitHub - MagedSaeed/farasapy: A Python implementation of Farasa toolkit. GitHub. https://github.com/MagedSaeed/farasapy?tab=readme-ov-file#want-to-cite\
[5] Sawalha, M., Atwell, E., & Abushariah, M. A. M. (2013). SALMA: Standard Arabic Language Morphological Analysis. 2013 1st International Conference on Communications, Signal Processing, and Their Applications (ICCSPA). https://doi.org/10.1109/iccspa.2013.6487311\
[6] Hjouj, M., Alarabeyyat, A., & Olab, I. (2016). Rule based approach for Arabic part of speech tagging and name entity recognition. International Journal of Advanced Computer Science and Applications, 7(6). https://doi.org/10.14569/ijacsa.2016.070642
