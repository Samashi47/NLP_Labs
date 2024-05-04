# Lab 2
**Author :** ***Ahmed Samady***\
**Supervised by :** ***Pr. Lotfi El Aachak***\
**Course :** ***NLP***\
This directory will contain everything that concerns Lab 2 of the NLP course.
You can find the notebook [here](https://github.com/Samashi47/NLP_Labs/blob/main/Lab2/lab2.ipynb).
## Part 1: Rule based NLP and RegEx
In the first part of this Lab we'll try to generate a bill from a sentence with a specific pattern, the following sentence represents this pattern:
```txt
I bought three Samsung smartphones 150 $ each, four kilos of fresh banana for 1,2 dollar a kilogram and one Hamburger with 4,5 dollar.
```
### Identifying the patterns
We can extract the following patterns from this sentence:
- The number of items of a product are always in plain English words, and it always comes before its name.
- The unite price of a product is always in digits, and it always comes before either the word `dollar` or the sign `$`.
- The name of the product is between the number of items and the unite price of this latter.
### Number of items of the product
We need to capture first the numbers in plain English words from a list that contains all the number names up to a trillion, which are not a lot compared to the Arabic language for example. Then, we need to add another capturing group with number names again plus the stopword `and` to make sure that we also capture composite numbers (e.g. `two million and five hundred thousand`).
The following RegEx represents this pattern:
```python
r"(?:\s)(?:" + "|".join(number_list) + r")(?:\s(?:" + "|".join(number_list) + r"|and))*"
```
### Product price
The price pattern is quite simple, we need to capture the digits that come before the word `dollar` or the sign `$`, we need to make sure that we can also deal with prices that have decimal points with either `,` or `.`, after this we convert the number from words to number using a library called `word2number`.
```python
r"(\d+(?:\,|\.)?\d*)\s*(?:\$|dollar)"
```
### product name
For the product name, we need to capture everything between the product, because we don't know how many words does the product name contain.
The following RegEx represents this pattern:
```python
r"(?:\b(?:" + "|".join(number_list) + r")\b)" + r"(.+?)" + r"(?:(?:\d*(?:\,|\.)?\d*)\s*(?:\$|dollar))"
```
To combat capturing words not related to the product name, we remove stopwords, `kilogram` word variants and number names in a rule after capturing the name.
```python
product_matches[i] = ' '.join([word for word in word_tokenize(product_matches[i]) if word not in stopwords.words('english')])
product_matches[i] = ' '.join([word for word in word_tokenize(product_matches[i]) if word not in kg_variants])
product_matches[i] = ' '.join([word for word in word_tokenize(product_matches[i]) if word not in number_list])
```
After Applying these regex patterns on our sentence we get the following bill:
```txt
| Product                | Quantity |   Unit Price |          Total Price |
---------------------------------------------------------------------------
| Samsung smartphones    |        3 |          150 |                450.0 |
| fresh banana           |        4 |          1.2 |                  4.8 |
| Hamburger              |        1 |          4.5 |                  4.5 |
---------------------------------------------------------------------------
```
## Part 2: Word Embeddings
In the second part of this Lab, we'll use the data collected during the [Lab1](https://github.com/Samashi47/NLP_Labs/blob/main/Lab2) to apply on it different word embedding technics, and finally visualize the word vectors using t-SNE.
### Tokenizing and creating a set of unique words of the corpus
Before encoding the words in our corpus, we need first to tokenize our raw text into sentences, which we'll be calling documents. Then tokenizing the text into words and create a set of unique words in our corpus. After that we need clean our set by removing punctuations, and numbers from our set.
### 1.1. One-Hot Encoding
To One-Hot encode, we'll create an array of zeroes for each word with the size of the unique words set, then put the number 1 in the position where our word occurs in our raw text. This approach creates an array with size 110 for each word in the unique word set.
#### Example:
```txt
One-hot Encoded vector for the word 'العملات' :
[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```
### 1.2. Bag of Words
The Bag of words approach creates vectors for documents instead of words, which means it creates a vector of zeroes for each document with length 144, then it puts adds one each time the word appears in the document without taking into consideration the position of the word.
#### Example
```txt
Document 5: 	[0	1	1	0	1	0	0	0	0	0	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	1	0	0	0	0	1	0	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	1	1	0	0	0	0	1	0	0	0	1	1	1	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0	1	0	1	0	0	0	0	0	0	2	0	1	0	0	0	0	0	0	0	0	0	0	0	1	0	0]
```
### 1.3. TF-IDF
The same goes for TF-IDF, the value given to each word is in relation with the document it occurs in, which means we'll also be creating vectors for sentences instead of words, by calculating the TF-IDF value according to the following formula:
$$TF = \frac {\textrm{number of times the term appears in the document} }{ \textrm{total number of terms in the document}}$$
$$IDF = log (\frac {\textrm{number of the documents in the corpus}} {\textrm{number of documents in the corpus contain the term}})$$
And finally:
$$\textit{TF-IDF} = TF * IDF$$
#### Example
```txt
Document 5:	[0.0	0.17841660962639477	0.2038368891367365	0.0	0.2038368891367365	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.2038368891367365	0.17841660962639477	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.17841660962639477	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.2038368891367365	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.17841660962639477	0.0	0.0	0.0	0.0	0.0	0.0	0.2038368891367365	0.0	0.0	0.0	0.0	0.17841660962639477	0.0	0.0	0.0	0.0	0.0	0.0	0.17841660962639477	0.2038368891367365	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.2038368891367365	0.17841660962639477	0.0	0.0	0.0	0.0	0.2038368891367365	0.0	0.0	0.0	0.10293459009673701	0.2038368891367365	0.2038368891367365	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.2038368891367365	0.0	0.2038368891367365	0.0	0.0	0.0	0.0	0.17841660962639477	0.0	0.2038368891367365	0.0	0.0	0.0	0.0	0.0	0.0	0.35683321925278955	0.0	0.16038063428449728	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.2038368891367365	0.0	0.0]
```
### 2. Word2Vec
For the Word2Vec we'll be using the `gensim` library to train the models on our corpus, but before that we need to create a list of lists of words for each document in our corpus.
#### 2.1. CBoW
For CBoW, we train the model on the tokenized corpus for 1000 epochs with the following params:
```python
cbow_w2v_model = word2vec.Word2Vec(tokenized_corpus, vector_size=100, window=30,min_count=1, sample=1e-3, sg=0, epochs=1000)
```
- vector_size: The output vectors will be in size 100.
- window: Maximum distance between the current and predicted word within a document is set to 30.
- min_count: Set to 1, to not Ignore any words including those with frequency equal to 1 since our corpus is small.
- sample: setting the threshold for configuring which higher-frequency words are randomly downsampled to $10^{-3}$.
After successfully training our CBoW model, We can see which words are the most similar to each other, the following example represents the tp 5 most similar words to the word `بنسبة` according to the CBoW model:
```txt
['بالمئة', 'وعلى', 'المحللون', 'ارتفاعه', 'أساس'] : 'بنسبة'
```
##### **Example**
```txt
Vector for the word 'الحالي' using CBoW:
[ 0.40585384  0.32509694  0.22022803  0.29177958  0.48195058 -0.6955288
  0.60727715  1.0006884  -0.16462882 -0.98436636 -0.04244173 -0.91498667
  0.5704375   0.73498315 -0.84547573  0.46672276  0.286757   -0.35923517
 -0.5054921  -1.0979508   0.67788446  0.10715085  0.19086702  0.1860761
  0.46020535  0.73452985 -0.20113394 -0.12013378 -0.26016775 -0.54818803
  0.0186763   0.35564584  0.27218705 -0.9666781   0.43027267  0.21815307
  0.7488732  -0.5986262  -0.2980162  -0.30958772  0.01217927 -0.48528177
 -0.6167618  -0.45601204  0.08364289 -0.19954176 -0.12483604 -0.683708
  0.25369087  0.39801896  0.5395854  -0.38150787 -0.4052238   0.48062095
 -0.14117202  0.4635832   0.07261931 -0.14047082 -0.53966504  0.00846349
  0.15194567 -0.5561599  -0.0581234   0.9189117  -0.03927541  0.6039816
 -0.02695844 -0.0502873  -0.4994488   1.1964487  -0.12635693  0.4027613
  0.16340312 -0.61614096 -0.02511098 -0.35206544  0.2003625   0.28168663
  0.40613893  0.6483951  -0.6059088   0.10292934 -0.20696087  0.23929845
 -0.8706727   0.17895703  0.6637303   0.31540412  0.25318483 -0.11749188
  1.0395292   0.07300752  0.27297366  0.34289142  0.11322439 -0.04124933
 -0.13191976 -0.1626137  -0.14869364  0.60889995]
```
#### 2.1. Skip Gram
We pursue the same steps we did in CBoW for Skip Gram, including the params set in CBoW.
```python
sg_w2v_model = word2vec.Word2Vec(tokenized_corpus, vector_size=100, window=30,min_count=1, sample=1e-3, sg=1, epochs=1000)
```
The following example represents the tp 5 most similar words to the word `بنسبة` according to the Skip Gram model:
```txt
['بالمئة', 'الشهر', 'شهريا', 'المحللون', 'ارتفعت'] : 'بنسبة'
```
##### **Example**
```txt
Vector for the word 'الحالي' using Skip Gram:
[ 0.351596    0.11718044  0.12637074  0.31565505  0.13480297 -0.46318123
  0.3316163   0.72249305 -0.04884658 -0.635933   -0.06255357 -0.5499263
  0.32690346  0.42105365 -0.459817    0.3308721   0.08732525 -0.2651812
 -0.41773757 -0.7221311   0.41768947  0.08542081  0.11429168  0.2575409
  0.17747745  0.51699615 -0.14309505 -0.07245834 -0.28154376 -0.29573435
  0.09176683  0.15002267  0.4425892  -0.7556945   0.160434    0.3132306
  0.30820101 -0.26995087 -0.32891345 -0.3092285   0.07381533 -0.3913162
 -0.5034647  -0.33627924  0.10959452 -0.154013   -0.16939299 -0.41897428
  0.09075999  0.39628726  0.4646336  -0.2859987  -0.24887535  0.16225596
 -0.17590246  0.4373922  -0.03602189 -0.06720302 -0.5498436  -0.11821971
  0.11999602 -0.24190864 -0.08767218  0.70117855 -0.06968234  0.5284474
  0.05285387  0.02619917 -0.24548586  0.84020305 -0.04866069  0.27681342
  0.0822886  -0.4161338   0.34246632 -0.20079285  0.22625217  0.18045907
  0.28693095  0.43843457 -0.41984242 -0.0082708  -0.05928691  0.14469407
 -0.7065391   0.10465859  0.44125286  0.1331416   0.0965855  -0.09510706
  0.83116615 -0.01111831  0.35314786  0.33043167 -0.01366086  0.00699723
 -0.12238304 -0.10693885 -0.36175126  0.40693915]
```
### 3. FastText and GloVe
#### 3.1. FastText
For FastText using CBoW we'll also be using `gensim` library to train a model on our corpus. We also pursue the same steps we did in CBoW and Skip Gram to train the FastText, we train the model with the following params:
```python
fasttext_model = fasttext.FastText(tokenized_corpus, vector_size=100, window=30,min_count=1, sample=1e-3, sg=0, epochs=1000)
```
The following example represents the tp 5 most similar words to the word `بنسبة` according to the FastText model:
```txt
['بالمئة', 'سنوي', 'وعلى', 'أساس', 'يتوقعون'] : 'بنسبة'
```
##### Example
```txt
Vector for the word 'الحالي' using FastText :
[-0.40731886 -0.31146854  0.81020516  0.66898406 -0.52810407 -0.09366013
 -0.06773606  0.13596728 -0.30394506 -0.10587464 -0.280977   -0.5401727
 -0.0314507   0.2769033   0.08481572  1.0180722  -0.08037109 -0.49991283
  0.41156042 -1.0127972   0.24638991 -0.01124164 -0.15263402 -0.83982295
  0.23412077 -0.34710303  0.14548664  0.2342649  -0.36383757  0.18283173
  0.49757963  0.06439958 -0.26919374  0.24766779  0.12325227  0.3118563
  0.41452795 -0.19806488  0.04893633 -0.06481551  0.22802117  0.54331255
 -0.30992422  0.20416823  0.06943946  0.72058505 -0.4426385  -0.60023373
 -0.9508384  -0.02861012  0.4676601  -0.03480138  0.8184698   0.9758465
  0.17863071  0.053195    0.42390653  0.433571   -0.20044725  0.34744048
 -1.0864998   0.5439439   0.63057524  0.7016845   0.07221132  0.76783025
 -0.02533424  0.7956822   0.8608498   0.77650046  0.8219391   0.68097275
  0.38662454  0.2103829   0.03143413 -0.42955193  0.15455462  0.00594598
 -0.64253587 -0.0819249  -0.83429754  0.12272646  0.5350089   0.18250081
 -0.42760637 -0.13992354 -0.65360874 -0.16630192 -0.37904644 -0.18750203
 -0.6259723   0.34792718  0.02784466 -0.46881574 -0.2538496  -0.2918771
  0.7175975  -0.28689224 -0.2408298   0.83918816]
```
#### 3.2. GloVe
For GloVe we'll be using a pre-trained model of Arabic vectors from the [internet archive](https://archive.org/details/arabic_corpus), and then use the `gensim` library with the KeyedVectors class to load the model.
```python
glove_model = KeyedVectors.load_word2vec_format('.vector_cache/vectors.txt', encoding='utf-8', no_header=True)
```
The vectors are 256 in length, and some words in our corpus are not available in the model's vocabulary, so we won't be treating those.
Here's a list of words from our corpus not available in its vocabulary:
```txt
['أسعار', 'الأمريكية', 'الأصفر', 'للأوقية', 'إعلان', 'الأساسي', 'أيام', 'أساس', 'وأظهرت', 'الأشد', 'أنهت', 'الآجلة', 'الأربعاء', 'بأكثر']
```
##### Example
I won't be including an example for a vector here since they are lengthy compared to the others I included here. But as an example, we can test the cosine similarity between the two words `الحالي` and `الماضي` = \
0.650802493095398
### 4. Plotting the vectors using t-SNE
To plot the vectors, we'll be using `matplotlib` for the plots, and the TSNE class from the `scikit-learn` library.in addition we use two libraries called `arabic_reshaper` and `bidi` to correctly display the Arabic words on the plots.
For all the models's vectors we'll be transforming them to 2d vectors using t-SNE with the following params:
```python
tsne = TSNE(n_components=2, random_state=0, n_iter=5000, n_iter_without_progress=500)
vectors_2d = tsne.fit_transform(vectors)
```
#### 4.1. One-Hot Encoding
For the One-Hot encoded 2d vectors we get the following scatter plot:
<img src="t-sne-plots\One-Hot.png"></img>\
We can notice that the vectors do not represent any semantic meaning of a word in relation to the corpus, as the majority of the words are arounnd (3,-3) position, this is due to the nature of how One-Hot encoding works.

#### 4.2. CBoW
For the CBoW 2d vectors we get the following scatter plot:\
<img src="t-sne-plots\cbow.png"></img>\
The CBoW model gives more scattered words, and some word clusters that the model thinks are the most similar. This approach provides a better way to represent the semantic rlationship between words in our corpus compared to One-hot encoding.

#### 4.3. Skip Gram
For the Skip Gram  2d vectors we get the following scatter plot:\
<img src="t-sne-plots\skipgram.png"></img>\
The same goes for Skip Gram. However, the words vectors seem to be more sparse in comparison with CBoW, this can be due to the fact that skip gram unlike CBoW, it tries to predict the context of a word instead of word in a certain context.

#### 4.4. FastText
For the FastText with CBoW 2d vectors we get the following scatter plot:\
<img src="t-sne-plots\fasttext.png"></img>\
We can see in the plot that the clusters generated using FastText are clearly well-separated, and we can notice that words such as `ارتفاع` ,`ارتفع` , and `ارتفعت` are close to each other, which means they are semantically related.

#### 4.5. GloVe
For the Arabic pre-trained GloVe model 2d vectors we get the following scatter plot:\
<img src="t-sne-plots\glove.png"></img>\
For the pre-trained Arabic GloVe model we can notice that the vectors are sparse, and that there are no clusters that we can identify visually. This can be due to the fact that this model is trained on a huge number of words, which means we're discarding the relationship of the words in our corpus but rather on a their relationship with other words on a much larger corpus, which might affect how the plot represents the semantic relation between the words of our small corpus.

### Conclusion

At the end of this Lab we can come to the following conclusions:
- Each word embedding technique offers unique advantages and is suited for different tasks and types of corpora.
- For tasks requiring semantic understanding and capturing contextual relationships between words, advanced techniques like Word2Vec (Skip Gram and CBoW) and FastText are preferable.
- When working with a small corpus, custom training Word2Vec or FastText models on the specific dataset may yield more tailored and informative word embeddings compared to pre-trained models like GloVe.
- Visualization using t-SNE helps to understand the distribution and relationships of word vectors in a lower-dimensional space, providing insights into the semantic structure of the corpus.

## References
- Kal. (2022, January 4). Word embedding using FastText - KAL - Medium. Medium. https://medium.com/@93Kryptonian/word-embedding-using-fasttext-62beb0209db9
- Educative. One-hot encoding of text data in natural language processing. https://www.educative.io/answers/one-hot-encoding-of-text-data-in-natural-language-processing
- Imran, R. (2023, June 12). Comparing text preprocessing techniques: One-Hot encoding, Bag of Words, TF-IDF, and Word2VEC for sentiment analysis. Medium. https://medium.com/@rayanimran307/comparing-text-preprocessing-techniques-one-hot-encoding-bag-of-words-tf-idf-and-word2vec-for-5850c0c117f1
- Kal. (2022b, January 4). Word embedding using FastText - KAL - Medium. Medium. https://medium.com/@93Kryptonian/word-embedding-using-fasttext-62beb0209db9
- Alshargi. (2024, March 1). Building Skipgram and CBOW Models for Arabic FastText Word Embeddings from Scratch. Medium. https://medium.com/@alshargi.usa/building-skipgram-and-cbow-models-for-arabic-fasttext-word-embeddings-from-scratch-c54860ccd7e7
- Arabic Corpus and Trained GloVe word2vec Vectors : Compiled by: Tarek Eldeeb : Free Download, Borrow, and Streaming : Internet Archive. (2018, June 14). Internet Archive. https://archive.org/details/arabic_corpus
- Alshargi. (2023, July 7). Plotting Arabic Words using Word2Vec Technique: Fixing Right-to-Left Character Appearance Issue. Medium. https://medium.com/@alshargi.usa/plotting-arabic-words-using-word2vec-technique-fixing-right-to-left-character-appearance-issue-5951f17ed592