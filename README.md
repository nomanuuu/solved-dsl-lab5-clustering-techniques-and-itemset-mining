Download Link: https://assignmentchef.com/product/solved-dsl-lab5-clustering-techniques-and-itemset-mining
<br>
The main objective of this laboratory is to put into practice what you have learned on clustering techniques and itemset mining. You will mainly work on textual data, a domain where the data preparation phase is crucial to any subsequent task. Specifically, you will try to detect topics out of a set of real-world news data. Then, you will describe each cluster through frequent itemset mining.

<strong>Important note. </strong>For what concerns this laboratory, you are encouraged to upload your results to our online verification platform, even though the submission will not count on your final exam mark. Doing so, you can practice with the same system that you will use for the final exam. Reference Section 3 to read more about it.

<h1>1          Preliminary steps</h1>

<h2>1.1         Useful libraries</h2>

As you may have already understood, the Python language comes with many handy functions and thirdparty libraries that you need to master to avoid boilerplate code. In many cases, you should leverage them to focus on the analysis process rather than its implementation.

That said, we listed a series of libraries you can make use of in this laboratory:

<ul>

 <li><a href="https://numpy.org/">NumPy</a></li>

 <li><a href="https://scikit-learn.org/stable/">scikit-learn</a></li>

 <li><a href="https://www.nltk.org/">Natural Language Toolkit</a></li>

 <li><a href="https://www.scipy.org/">SciPy</a></li>

</ul>

We will point out their functions and classes when needed. In many cases, their full understanding decreases significantly your programming effort: take your time to explore their respective documentations.

<strong>Warning: </strong>we have noticed from previous laboratories that in some cases copying snippets of code directly from the PDF file leaded to wrong behaviours in Jupyter notebooks. Please consider to write them down by yourself.

<h2>1.2         wordcloud</h2>

Make sure you have this library installed. As usual, if not available, you need to install it with pip install wordcloud (or any other package manager you may be using). The wordcloud library is a word cloud generator. You can read more about it on its <a href="https://amueller.github.io/word_cloud/">official website</a><a href="https://amueller.github.io/word_cloud/">.</a>

<h2>1.3         Datasets</h2>

For this laboratory, a single real-world dataset will be used.

<h3>1.3.1         20 Newsgroups</h3>

The 20 Newsgroups dataset was originally collected in Lang 1995. It includes approximately 20,000 documents, partitioned across 20 different newsgroups, each corresponding to a different topic.

For the sake of this laboratory, we chose <em>T </em>≤20 topics and sampled uniformly only documents belonging to them. As a consequence, you have <em>K </em>≤20<em>,</em>000 documents uniformly distributed across <em>T </em>different topics. You can download the dataset at: https://github.com/dbdmg/data-science-lab/blob/master/datasets/T-newsgroups.zip?raw=true Each document is located in a different file, which contains the raw text of the news. The name of the file in an integer number and corresponds to its ID.

<h1>2          Exercises</h1>

Note that exercises marked with a (*) are optional, you should focus on completing the other ones first.

<h2>2.1         Newsgroups clustering</h2>

In this exercise you will build your first complete data analytics pipeline. More specifically, you will load, analyze and prepare the newsgroups dataset to finally identify possible clusters based on topics. Then, you will evaluate your process through any clustering quality measure.

<ol>

 <li>Load the dataset from the root folder. Here the Python’s <a href="https://docs.python.org/3/library/os.html#module-os">os</a> module comes to your help. You can use the os.listdir function to list files in a directory.</li>

 <li>Focus now on the data preparation step. As you have learned in laboratory 2, textual data needs to be processed to obtain a numerical representation of each document. This is typically achieved via the application of a weighting schema.</li>

</ol>

Choose now one among the weighting schema that you know and transform each news into a numerical representation. The Python implementation of a simple <em>TFIDF </em>weighting schema is provided in section 2.1.1, you can use it as starting point.

This preprocessing phase is likely going to influence the quality of your results the most. Pay enough attention to it. You could try to answer the following questions:

<ul>

 <li>Which weigthing schema have you used?</li>

 <li>Have you tried to remove stopwords?</li>

 <li>More generally, have you ignored words with a document frequency lower than or higher than a given threshold?</li>

 <li>Have you applied any dimensionality reduction strategy? This is not mandatory, but in some cases it can improve your results. You can find more details in Appendix 3.</li>

</ul>

<ol start="3">

 <li>Once you have your vector representation, choose one clustering algorithm of those you know and apply it to your data.</li>

 <li>You can now evaluate the quality of the cluster partitioning you obtained. There exists many metrics based on distances between points (e.g. the Silhouette or the Sum of Squared Errors (SSE)) that you can explore. Choose one of those that you known and test your results on your computer.</li>

 <li>Consider now that our online system will evaluate your cluster quality based on the real cluster labels (a.k.a. the <em>ground truth</em>, that you do not have). Consequently, it could happen that a cluster subdivision achieves an high Silhouette value (i.e. <em>geometrically </em>close points were assigned to the same cluster) while the matching with the real labels gives a poor score (i.e. real labels are heterogeneous within your clusters).</li>

</ol>

In order to understand how close you came to the real news subdivision, upload your results to our online verification system (you can perform as many submission as you want for this laboratory, the only limitation being a time limit of 5 minutes between submissions). Head to Section 3 to learn more about it.

<h3>2.1.1         A basic TFIDF implementation</h3>

The transformation from texts to vector can be simplified by means of ad-hoc libraries like Natural Language Toolkit and scikit-learn (from now on, nltk and sklearn). If you plan to use the <em>TFIDF </em>weighting schema, you might want to use the sklearn’s <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html">TfidfVectorizer</a> class. Then you can use its fit_transform method to obtain the <em>TFIDF </em>representation for each document. Specifically, the method returns a SciPy <a href="https://docs.scipy.org/doc/scipy/reference/sparse.html">sparse matrix</a><a href="https://docs.scipy.org/doc/scipy/reference/sparse.html">.</a> You are encouraged to exhaustively analyze Tfidf Vectorizer’s constructor parameters since they can significantly impact the results. Note for now that you can specify a custom tokenizer object and a set of stopwords to be used.

For the sake of simplicity, we are providing you with a simple tokenizer class. Note that the TfidfTokenizer’s tokenizer argument requires a callable object. Python’s callable objects are instances of classes that implement the __call__ method. The class makes use of two nltk functionalities: <a href="https://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize">word_tokenize </a>and the class <a href="https://www.nltk.org/_modules/nltk/stem/wordnet.html">WordNetLemmatizer</a><a href="https://www.nltk.org/_modules/nltk/stem/wordnet.html">.</a> The latter is used to lemmatize your words after the tokenization. The <em>lemmatization </em>process leverages a morphological analysis of the words in the corpus with the aim to remove the grammatical inflections that characterize a word in different contexts, returning its base or dictionary form (e.g. {am, are, is} ⇒ be; {car, cars, car’s, cars’} ⇒ car).

For what concerns the stop words, you can use again a nltk already-available function: <a href="https://www.nltk.org/_modules/nltk/corpus.html">stopwords</a><a href="https://www.nltk.org/_modules/nltk/corpus.html">. </a>The following is a snippet of code including everything you need to get to a basic <em>TFIDF </em>representation:

from sklearn.feature_extraction.text import TfidfVectorizer from nltk.tokenize import word_tokenize from nltk.stem.wordnet import WordNetLemmatizer from nltk.corpus import stopwords as sw

class LemmaTokenizer(object):

def __init__(self):

self.lemmatizer = WordNetLemmatizer()

def __call__(self, document):

lemmas = [] for t in word_tokenize(document): t = t.strip()

lemma = self.lemmatizer.lemmatize(t) lemmas.append(lemma)

return lemmas

lemmaTokenizer = LemmaTokenizer()

vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, stop_words=sw.words(‘english’)) tfidf_X = vectorizer.fit_transform(corpus)

<h2>2.2         Cluster characterization by means of word clouds and itemset mining</h2>

In many real cases, the <em>real </em>clustering subdivision is not accessible at all<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. Indeed, it is what you want to discover by clustering your data. For this reason, it is commonplace to add a further step to the pipeline and try to characterize the clusters by inspecting their points’ characteristics. This is especially true while working with news, where the description can lead to the identification of a topic shared among all the documents assigned to it (e.g. one of your clusters may contain news related to sports).

In this exercise you will exploit word clouds and frequent itemset algorithms to characterize the clusters obtained in the previous exercise.

<ol>

 <li>Split your initial data into separate chunks accordingly to the cluster labels obtained in the previous exercise. For each of them, generate a Word Cloud image using the wordcloud library. Take a look at the library <a href="https://amueller.github.io/word_cloud/">documentation</a> to learn how to do it. Can you figure out the topic shared among all the news of each cluster?</li>

 <li>(*) Provide a comment for each word cloud and upload the images and the comments. Head to section 3 to know how.</li>

 <li>(*) One further analysis can exploit the frequent itemset algorithms. Choose one algorithm and run it for each cluster of news. Try to identify the most distinctive set of words playing around with different configurations of the chosen algorithm. Based on the results, can you identify any topic in any of your clusters?</li>

</ol>

<h1>3          Submitting you work</h1>

For this laboratory, you should upload two files to two different web sites. The first file contains the clustering results, the second file contains a report on the identified clusters. The following sections provide further details on that.

<strong>Deadline. </strong>You can submit your work until <strong>Monday November 11, 11:59PM</strong>. Later submissions will not be evaluated.

<h2>3.1         Upload clustering results</h2>

You are required to upload a single CSV file. Please respect the following requirements:

<ul>

 <li>use the format UTF-8 (see <a href="https://docs.python.org/3.8/library/functions.html#open">open()</a><a href="https://docs.python.org/3.8/library/functions.html#open">’</a>s encoding parameter)</li>

 <li>the first line must contain a two columns header equal to: Id,Predicted</li>

 <li>each of the <em>N </em>following lines must contain the document ID followed by the cluster ID assigned to it. IDs must be increasing integer numbers starting from 0.</li>

</ul>

The file must be uploaded to our submission platform located at <a href="http://35.158.140.217/">http://35.158.140.217/</a><a href="http://35.158.140.217/">.</a> You can find an <a href="http://dbdmg.polito.it/wordpress/wp-content/uploads/2019/11/lab5_sample_submission.csv">example file</a> on the course website.

Please reference the <a href="http://dbdmg.polito.it/wordpress/wp-content/uploads/2019/11/Data_Science_Lab___Laboratory_submission.pdf">guide</a> from the course website to go through the submission procedure.

<h2>3.2         Upload your report</h2>

You are required to upload a single PDF file. Please respect the following requirements:

<ul>

 <li>state clearly how many clusters you have found;</li>

 <li>for each of them, report the description that you obtained in exercise 2.</li>

</ul>

If you have developed your solution on a Jupyter notebook, you can export it as a PDF and use it for the submission. However the file must be sufficiently commented.

The file must be uploaded to the <a href="https://www.polito.it/intranet/">“Portale della Didattica”</a><a href="https://www.polito.it/intranet/">,</a> under the Homework section of the course. Please use as description: report_lab_5.

<h2>3.3         Evaluation</h2>

Your clustering results will be evaluated via the Rand-index score. You can read more about it on <a href="https://en.wikipedia.org/wiki/Rand_index">Wikipedia</a><a href="https://en.wikipedia.org/wiki/Rand_index">.</a>

Your report will be evaluated via the old, always-working human reading.

<h1>Appendix</h1>

<h2>Notions on linear transformations and dimensionality reduction</h2>

In many real cases, your data comes with a large number of features. However, there is often a good chance that some of them are uninformative or redundant and have the only consequence of making your analysis harder. The simplest example could be features that are linearly dependent with each other (e.g. a feature that describes the same information with different, yet correlated units, like degrees Celsius and Fahrenheit).

One additional detail can be addressed in your preprocessing step, other than the dimensionality reduction. There might be cases where the distribution of your data has hidden underlying dynamics that could be enhanced by choosing different features (i.e. dimensions). Figure 1 shows several points distributed in a Gaussian cluster (see Laboratory 4). Let’s make now an assumption: quantitatively we assess that directions with largest variances in our space contain the dynamics of interest. In Figure 1 the direction with the largest variance is not (1,0) nor (0,1), but the direction along the long axis of the cluster.

As you will soon learn, in the known literature there is a proven method that addresses the aforementioned problems: the <em>Principal Component Analysis </em>(PCA). This technique frames the problem as a linear change of basis. The final basis vector are commonly termed <em>principal components</em>. Let’s qualitatively understand why and how it is made by means of a few algebraic notions.

In PCA we assume that there exist a more meaningful basis to re-express our data. The hope is that this new basis will filter out the noise and reveal hidden dynamics. Following the assumption presented beforehand, the new basis must align the directions with the highest variance. Also, the change of basis follows another strict, yet powerful assumption: it is assumed that the new basis is a linear combination of the original one (studies expanded on this to non linear domains). In Figure 1 PCA would likely identify a new basis in the directions of the two black arrows.

In other words, if we call <strong>X </strong>the original set of data, in PCA we are interested in finding a matrix <strong>P </strong>that stretches and rotates our initial space to obtain a more convenient representation <strong>Y</strong>:

<em>PX </em>= <em>Y                                                                                          </em>(1)

Now that foundations are laid, we know that we are looking for a new basis that highlights the inner dynamics and we assume that the change of basis can be achieved with a simple linear transformation. This linearity assumption let us solve analytically the problem with matrix decomposition techniques. Even if the simpler eigendecomposition can be used, the state-of-the-art solution is obtained through the <em>Singular</em>

<em>Value Decomposition </em>(SVD).

Many, many theoretical and technical details have been left behind in this short summary. If you are willing to learn more, you can find a thorough tutorial about PCA, SVD and their relationship in Shlens 2014.

The use of the PCA algorithm via SVD decomposition in Python is straightforward. The following lines show how you can apply the change of basis transforming your data.

from sklearn.decomposition import TruncatedSVD

<em># X: np.array, shape (1000, 20)</em>

svd = TruncatedSVD(n_components=5, random_state=42)

red_X = svd.fit_transform(X)                                   <em># red_X will be: np.array, shape (1000, 5)</em>

Note that the TruncatedSVD class lets you choose how many top-principal components to retain (they are ranked by explained variance). Doing so, you will be applying the dimensionality reduction at the same time.

<h1>References</h1>

<ul>

 <li>Ken Lang. “Newsweeder: Learning to filter netnews”. In: <em>Proceedings of the Twelfth International Conference on Machine Learning</em>. 1995, pp. 331–339.</li>

 <li>Jonathon Shlens. “A Tutorial on Principal Component Analysis”. In: <em>CoRR </em>abs/1404.1100 (2014).</li>

</ul>

arXiv: <a href="https://arxiv.org/abs/1404.1100">1404.1100</a><a href="https://arxiv.org/abs/1404.1100">.</a> url: <a href="https://arxiv.org/abs/1404.1100">http://arxiv.org/abs/1404.1100</a><a href="https://arxiv.org/abs/1404.1100">.</a>

Figure 1: Points distributed in a Gaussian cluster with mean (1,3) and standard deviation 3.             Source: <a href="https://en.wikipedia.org/wiki/Principal_component_analysis">Wikipedia</a>

<a href="#_ftnref1" name="_ftn1">[1]</a> Or worse, it might not <em>exist </em>at all.