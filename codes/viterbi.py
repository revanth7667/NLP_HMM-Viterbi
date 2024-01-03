"""
Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

NLP, Duke University, Fall-2023
Professor: Patrick Wang
Assignment: 03

Group Members:
1. Revanth Chowdary Ganga (rg361)
2. Divya Sharma (ds655)
"""
from typing import Sequence, Tuple, TypeVar
import numpy as np
import nltk


Q = TypeVar("Q")
V = TypeVar("V")
alpha = 1  # smoothening factor, adjust as required


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[Q], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)


def normalize_smooth(mat):
    """Adds the smoothening Factor to 2D array and Normalizes rowise"""

    arr = mat.copy()  # deep-copy to prevent changing original input

    arr = arr + alpha  # add smoothening factor

    arr = arr / arr.sum(axis=1)[:, None]  # normalize rowise

    return arr


def initial_prob(corp):
    """Calcualte initial probabilites and set order of tags"""
    start_tags = {}

    for _ in corp:
        start_tags[_[0][1]] = start_tags.get(_[0][1], 0) + 1

    tag_order = [_ for _ in start_tags.keys()]  # to ensure or  1der stays consistent
    len_tag = len(tag_order)

    # Save in 1D numpy array =
    ist = np.zeros(len_tag, dtype=float)
    for _ in range(len_tag):
        ist[_] = start_tags[tag_order[_]]

    # smoothen and normalize
    ist = ist + alpha
    ist = ist / sum(ist)

    return (ist, tag_order, len_tag)


def mat_transition(corp, tag_order, len_tag):
    """To Calculate the Transition Matrix"""
    tran_count = {}
    len_flat = len(corp)

    for i in range(len_flat - 1):
        tran_count[(corp[i][1], corp[i + 1][1])] = (
            tran_count.get((corp[i][1], corp[i + 1][1]), 0) + 1
        )

    # initialize 2D numpy array to store the transition probabilities
    tran_prob = np.zeros((len_tag, len_tag), dtype=float)

    # populate values in the numpy array
    for _ in tran_count.keys():
        tran_prob[tag_order.index(_[0]), tag_order.index(_[1])] += tran_count[_]

    # smoothen and normalize the rows to convert to probabilites
    return normalize_smooth(tran_prob)


def mat_emission(corp, tag_order, len_tag):
    """To Calculate the emission matrix"""
    # list to store the unique words for ordering
    uni_vocab = []

    for _ in corp:
        if _[0] not in uni_vocab:
            uni_vocab.append(_[0])

    # add extra unknown variable to uni_vocab to account for unknown/new words
    uni_vocab.append(None)

    # initilzie 2D numpy array to store emission prob
    emi_prob = np.zeros((len_tag, len(uni_vocab)), dtype=float)

    # populate the counts
    for _ in corp:
        emi_prob[tag_order.index(_[1]), uni_vocab.index(_[0])] += 1

    # return smoothened and normalized array
    return (normalize_smooth(emi_prob), uni_vocab)


def main():
    # get the train data and test data
    train_set = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
    test_set = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]

    # Calculate the Arrays------------------------------------------------------
    # calucate initial probabilities
    (pi, tag_order, len_tag) = initial_prob(train_set)

    # flatten out the list for easier computation
    flat_train = [tup for sent in train_set for tup in sent]

    # Calculate the Transition probabilites
    A = mat_transition(flat_train, tag_order, len_tag)
    print(A)

    # calcualte the emission probabilities and get vocab
    (B, vocab) = mat_emission(flat_train, tag_order, len_tag)

    # Use Viterbi and Test------------------------------------------------------

    for x in test_set:
        obs_word = []
        for _ in x:
            try:
                ind = vocab.index(_[0])
            except:
                ind = len(vocab) - 1
            obs_word.append(ind)

        # call the viterbi function
        obs_out, prob = viterbi(obs_word, pi, A, B)
        correct = 0
        tot = len(x)
        for i in range(tot):
            print(
                f"Word: {x[i][0]} | Actual Tag: {x[i][1]} | Model Tag: {tag_order[obs_out[i]]}"
            )

            if x[i][1] == tag_order[obs_out[i]]:
                correct += 1

        print(f"Model probability:{prob}")
        print(f"Total Words: {tot} | Correct Tagging: {correct}")
        print(f"Accuracy: {correct/tot:%}")
        print("------------------------------------")


if __name__ == "__main__":
    main()
