"""make_lda_data. Compile corpus + dictionary from textfile across all submissions.

Usage:
  make_lda_data <textfile> <datafile>
"""

import string, os, codecs, gensim, save

from docopt import docopt
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

stemmer = EnglishStemmer()
STOPWORDS = set((stemmer.stem(w) for w in stopwords.words('english')))


# Stopwrods include words <= 2 characters.
def remove_stopwords(words):
#    return words
    return [ w for w in words if w not in STOPWORDS and len(w) >= 2 ]

def transform_word(word):
    # don't stem
    # return word.lower().rstrip(string.punctuation)
    return stemmer.stem(word.lower().strip(string.punctuation))

def transform_text(text):
    words = text.split()
    return remove_stopwords(map(transform_word, words))

def texts_iter(filename):
    for f in sorted(os.listdir('.')):
        if f[0] != '.' and os.path.isdir(f):
            with codecs.open(f + "/" + filename, "r", "utf-8", "ignore") as a:
                print "Submission by ", f
                raw_text = a.read()
                yield (f, raw_text, transform_text(raw_text))

if __name__ == "__main__":
    args = docopt(__doc__)

    students, raw_texts, texts = zip(*list(texts_iter(args['<textfile>'])))
    dictionary = gensim.corpora.dictionary.Dictionary(texts)
    dictionary.compactify()

    corpus = map(dictionary.doc2bow, texts)

    save.save(args['<datafile>'], students=students, texts=texts,
            raw_texts=raw_texts, corpus=corpus, dictionary=dictionary)

