"""fit_lda. Fit LDA given corpus + dictionary.

Usage:
  fit_lda <datafile> <modelfile> [--num_topics=<K>] [--iterations=<p>] [--passes=<q>] [--alpha=<alpha>] [--eta=<eta>]

Options:
  --num_topics=<K>  Number of topics [default: 15].
  --iterations=<p>  Number of M-step iterations [default: 50].
  --passes=<q>      Number of passes over data [default: 50].
  --alpha=<alpha>   Document-topic sparsity hyperparameter [default: 0.1]
  --eta=<eta>       Topic-word sparsity hyperparameter [default: 0.05]
"""
import gensim, logging, save
from docopt import docopt

forward_opts = { 'num_topics': int,
                 'passes': int,
                 'iterations': int,
                 'alpha': float,
                 'eta': float }

if __name__ == "__main__":
    args = docopt(__doc__)

    # Update the topics and convergence parameters from the command line.
    # If an option, excluding --, matches a forward_opts, keep it.
    lda_kwargs = dict((k[2:], forward_opts[k[2:]](v))
            for (k, v) in args.iteritems()
                if k[2:] in forward_opts and v)

    d = save.load(args['<datafile>'])

    logging.basicConfig(level=logging.INFO)
    # BATCH LDA.
    lda = gensim.models.ldamodel.LdaModel(
            corpus=d['corpus'], id2word=d['dictionary'],
            update_every=0, eval_every=5,
            **lda_kwargs)

    lda.print_topics(-1)

    lda.log_perplexity(d['corpus'])

    lda.save(args['<modelfile>'])

