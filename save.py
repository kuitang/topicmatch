# Save some local names as in MATLAB.
import cPickle

def save(filename, **kwargs):
    """save(filename, v1, v2, ...) saves a pickle with v1, v2, into filename."""

    # See http://stackoverflow.com/questions/6618795/get-locals-from-calling-namespace-in-python

    with open(filename, 'w') as f:
        cPickle.dump(kwargs, f, -1)

def load(filename):
    with open(filename, 'r') as f:
        return cPickle.load(f)

