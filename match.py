"""match. Match papers given a corpus and trained model.

Usage:
  match <datafile> <modelfile> <outfile> <reportfile>

"""
from __future__ import print_function
import gensim, save, csv, codecs
from docopt import docopt

import numpy as np
import networkx as nx
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#from scipy.stats import entropy

def symmetric_smooth_kl(p, q, epsilon=1e-5):
    denom = 1 + len(p)*epsilon
    ps = (p + epsilon) / denom
    qs = (q + epsilon) / denom

    return 0.5 * (np.dot(ps, np.log(ps) - np.log(qs)) +
                  np.dot(qs, np.log(qs) - np.log(ps)))

def compute_topic_concordance(p, q, name_p, name_q, epsilon=1e-5):
    denom = 1 + len(p)*epsilon
    ps = (p + epsilon) / denom
    qs = (q + epsilon) / denom

    loss = 0.5 * (ps * (np.log(ps) - np.log(qs)) + qs * (np.log(qs) - np.log(ps)))
    assert(np.isclose(symmetric_smooth_kl(p, q, epsilon), np.sum(loss)))

    return pd.DataFrame({name_p: ps, name_q: qs, 'loss': loss}).transpose()

def densify(sparse_vec, K):
    dense_vec = np.zeros(K)
    for i, v in sparse_vec:
        dense_vec[i] = v
    return dense_vec

def make_matching_graph(vecs):
    N = len(vecs)
    G = nx.complete_graph(N)
    for i, j in G.edges():
        # Negate the distance, since we are solving maximum-weight problem.
        G[i][j]['weight'] = -symmetric_smooth_kl(vecs[i], vecs[j])

    return G

def compute_weight(G, match):
    return sum(G[i][j]['weight'] for i, j in match.iteritems())

if __name__ == "__main__":
    args = docopt(__doc__)

    d = save.load(args['<datafile>'])
    lda = gensim.models.ldamodel.LdaModel.load(args['<modelfile>'])
    K = lda.num_topics

    # Obtain the posterior topic distribution for each document.
    sparse_vecs   = [ lda[doc] for doc in d['corpus'] ]
    # Convert to dense vectors so that we can compute KL divergences.
    dense_vecs    = [ densify(v, K) for v in sparse_vecs ]

    G = make_matching_graph(dense_vecs)

    # Find the maximum-weight (minimum-distance) matching
    # *among the max-cardinality matchings*. This ensures that as many
    # papers are matched as possible.

    print("About the run matching. This could take a while.")

    match = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    unmatched = set(G.nodes()) - set(match.keys()) - set(match.values())

    print("Matching completed. Max weight = %g and %d unmatched." % (
        compute_weight(G, match), len(unmatched)))

    # TODO: Handle the case with more than one unmatched.
    assert len(unmatched) <= 1

    # Greedily select the max-weight neighbor
    neighbor_weight = lambda v: G[v]['weight']

    for u in unmatched:
        max_weight_neighbor = max(G.neighbors(u),
                                  key=lambda v: G[u][v]['weight'])
        # Add the match
        match[u] = max_weight_neighbor

    # Recover the students from the numerical matchings
    st = d['students']

    # Store matches as list for stable order.henceforth.
    match_list = match.items()
    # Both directions of the matching are listed. We just want to keep one.
    match_list = list(set(tuple(sorted(m)) for m in match_list))
    n_matches  = len(match_list)

    student_matches = [(st[k], st[v]) for k, v in match_list]

    # Write matchings to outfile
    with open(args['<outfile>'], "w") as f:
        writer = csv.writer(f)
        writer.writerows(student_matches)

    # Write a more textual report, for analysis
    with codecs.open(args['<reportfile>'], "w", "utf-8", "ignore") as f:
        for n, (i, j) in enumerate(match_list):
            display_tuple = (n, st[i], st[j])
            loss_ij = -G[i][j]['weight']

            print("\n[Match %d] ====== [   %s <-> %s   ] ======\n" % display_tuple, file=f)

            print("Topic distribution concordance (total loss = %g):" % loss_ij, file=f)
            print(compute_topic_concordance(dense_vecs[i], dense_vecs[j], st[i], st[j]), file=f)

            print(d['raw_texts'][i], file=f)

            print("\n[Match %d] ------ ^^^ %s  |  %s vvv ------\n" % display_tuple, file=f)

            print(d['raw_texts'][j], file=f)

#        print("\n\n", file=f)
#        print("Unmatched: ", file=f)
#        for i, u in enumerate(sorted(unmatched)):
#            print("\n[Unmatched %d] ====== [ %s ] ======\n" % (i, st[u]), file=f)
#            print(d['raw_texts'][u], file=f)
#
