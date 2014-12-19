topicmatch
==========

Match papers (e.g. for peer review) by subject area, using a topic model.

A straightforward application of LDA and maximum-weight matching to assign a set of papers to each other, e.g. for peer review. I put this together while TAing for Prof. David Blei's graphical models course.

This is not (yet) an automatic pipeline. You need to do some tuning, especially of the topic counts and sparsity parameters LDA when you don't have a lot of data (say, in just one classroom).

Dependencies
==========
 - numpy, scipy, pandas, gensim, docopt, nltk, networkx

Usage
==========
This package is a series of scripts which work on pickle files.

1. You must prepare a directory for each author, and their documents need to be in the same filename. For example
  - mary/
    - abstract.txt
    - paper.txt
  - han/
    - abstract.txt
    - paper.txt
2. Run make_data.py to generate a corpus-dictionary file.
3. Run fit_lda.py to generate a fitted model. See the gensim documentation on what the optional parameters mean.
  - This runs batch LDA to minimize noise. With small data, online LDA has trouble converging.
  - The default parameters were what I found well to work for a dataset of ~40 paragraph-long abstracts. The important issue was have a document-topic sparsity (alpha) greater than the topic-word sparisty (eta) in order to encourage the model to find more discriminative topics, which was an issue for me.
  - Note that the fit_hdp.py code also works, but I haven't yet figured out how to tune it for this application. You are welcome to try.
4. Run match.py to generate the matching.
  - A csv with student names will be produced.
  - In addition, a textual report that lists a "topic distribution concordance" as well as the document texts for each match is printed. The concordance allows you to compare the topic distributions for each pair, and breaks down the contribution to the loss (in the matching problem) of each topic. Then the raw documents are printed to allow you to subjectively evaluate the result.
  - You can train with paper texts but only print out abstracts. As long as your directory structure is as shown in step 1, then you can just make data files for both papers and abstracts, pass the papers file to training, but pass the abstracts file to match. Match only uses the order of the names and the raw text for display, and make_data processes all files in sorted order, so this will work.
More documentation and examples will be forthcoming; I just wanted to put something up ASAP.

Happy topic matching!
