"""fit_lda. Fit LDA given corpus + dictionary.

Usage:
  fit_lda <datafile> <modelfile>
"""
import cPickle, gensim, logging
from docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__)
    d = None
    with open(args['<datafile>'], "r") as f:
        d = cPickle.load(f)

    # Set up logging to see LDA training output.
    logging.basicConfig(level=logging.INFO)
    hdp = gensim.models.hdpmodel.HdpModel(d['corpus'], d['dictionary'], chunksize=10, kappa=1.0, tau=64.0, K=8, T=100, alpha=1, gamma=1, eta=0.01, scale=1.0, var_converge=0.00001, outputdir=None)

    print "\n\n\n\n\n"
    print "TRAINING DONE; FINAL TOPICS:"

    hdp.optimal_ordering()
    hdp.print_topics(-1)

    hdp.save(args['<modelfile>'])

