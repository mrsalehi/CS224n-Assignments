# CS224n-Assignments
These are my solutions to the practical assignments of [CS224n (Natural Language Processing with Deep Learning)](http://web.stanford.edu/class/cs224n/) offered by Stanford University at Winter 2019.

There are five assignments in total. Here is a brief description of each one of these assignments:


## Assignment 1. Word Embeddings
This assignments has two parts which are about 

### 1. Count-Based Word Vectors
In this part, you have to use the co-occurence matrices to develop dense vectors for words. A co-occurence matrix counts how often different terms co-occur in different documents. To derive a co-occurence matrix, we use a window with a fixed size _w_, and then slide this window over all of the documents. Then, we count how many times two different words v_i and v_j occurs with each other in a window, and put this number in the (i, j) entry of the matrix.<br/>
Then, we have to run dimensionality reduction on the co-occurence matrix using singular value decomposition. We then select the top _r_ components after the decomposition and thus, derive r-dimensional embeddings for words.



### 2. Prediction(or Maximum Likelihood)-Based Word Vectors: Word2Vec
In this part, you will work with the pretrained word2vec embeddings of [gensim](https://radimrehurek.com/gensim/) package. There are lots of tasks in this part. At first, you have to reduce the dimensionality of word vectors using SVD from 300 to 2 so as to be able to visualize the vectors and analyze this visualization. Then you will find the closest word vectors to a given word vector. You will get to know words with several meanings (Polysemous words). You will get to know the analogy task, mentioed for the first time in the original paper of word2vec [(Mikolov et al. 2013)](https://arxiv.org/pdf/1301.3781.pdf%5D). The task is simple: given words x, y, and z, you have to find a word w such that the following relationship holds: x to y is like z to w. For example, Rome to Italy is like D.C. to the United Stats. You will find that solving this task with word2vec vectors is easy and is just a simple addition and subtraction of vectors, which is a nice feature of word2vec.




