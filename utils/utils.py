import re, os
import hashlib
import numpy as np

# Permanent hash for object o
def permanent_hash(o):
    """
    Implements a hashing that is constant across sessions.
    :param o: the object to be encoded
    :return: a string representation
    """
    return hashlib.md5((str(o)).encode()).hexdigest()

# Load vectors from dict
def load_vectors_as_dict(path):
    vectors = {}
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            # Split on white spaces
            line = line.strip().split(' ')
            if len(line) > 2:
                vectors[line[0]] = np.array([float(l) for l in line[1:]], dtype=np.float32)
            line = f.readline()
    return vectors

# Save oov vectors to path
def save_oov(nlp, path):
    if 'oov' not in nlp.__dir__():
        return
    else:
        np.savetxt(path, nlp.oov)
        return

# Load vectors in a spacy nlp
def load_vectors_in_lang(nlp, vectors_loc):
    wv= load_vectors_as_dict(vectors_loc)
    nlp.wv = wv

    # Check if list of oov vectors exists
    # If so, load, if not, create
    oov_path,ext = os.path.splitext(vectors_loc)
    oov_path = oov_path+'.oov.txt'
    if os.path.exists(oov_path):
        nlp.oov = np.loadtxt(oov_path)
    else:
        fk = list(wv.keys())[0]
        nf = wv[fk].shape[0]
        nlp.oov = np.random.normal(size=(100,nf))

# Get vector representation of word
def get_vector(w, nlp, nf=300):
    if 'wv' in nlp.__dir__():
        word = w.string.strip()
        v = nlp.wv.get(word, None)
        if v is None and 'oov' in nlp.__dir__():
            idx = np.random.randint(0, nlp.oov.shape[0])
            v = nlp.oov[idx]
        elif v is None:
            v = np.random.normal(size=nf)
    else:
        v = w.vector
    return v.astype(np.float32)

# Some cleaning especially with respect to weird punctuation
def clean_text(s):
    s = re.sub("([.,!?()-])", r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s

# Utility function to get a GloVe representation of a text
def get_matrix_rep(text, nlp, pos_to_remove=['PUNCT'], normed=True,
    lemmatize=False):

    text = clean_text(str(text)).lower()
    text = 'NULL '+text

    # Process the document via Spacy's nlp
    doc = nlp(text)

    # Lemmatize if desired
    if lemmatize:
        text = ' '.join([w.lemma_ for w in doc])
        doc = nlp(text)

    # Get processed words removing undesired POS
    words = [w for w in doc if w.pos_ not in pos_to_remove]
    
    # Get all vectors
    vecs = np.array([get_vector(w, nlp) for w in words], dtype=np.float32)
    if len(vecs) == 0:
        vecs = np.zeros((1,300), dtype=np.float32)

    # Normalize vectors if desired
    if normed:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs /= norms
    return vecs

# Pad an array up to maxlen
def pad(X, maxlen):
    """Pads with 0 or truncates a numpy array along axis 0 up to maxlen
    Args:
        X (ndarray): array to be padded or truncated
        maxlen (int): maximum length of the array
    Returns:
        ndarray: padded or truncated array
    """

    nrows = X.shape[0]
    delta = maxlen - nrows
    if delta > 0:
        padding = ((0,delta), (0,0))
        return np.pad(X, pad_width=padding, mode='constant')
    elif delta < 0:
        return X[:maxlen,:]
    else:
        return X