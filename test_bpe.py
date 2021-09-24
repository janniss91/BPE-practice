import pytest
from collections import Counter

from bpe import BytePairEncoder


@pytest.fixture
def sample_bpe():
    sample_corpus = "This is a simple text."
    return BytePairEncoder(sample_corpus, vocab_size=20)


@pytest.fixture
def counts():
    counts = Counter({
        ("i", "s"): 2,
        ("s", "#"): 2,
        ("T", "h"): 1,
        ("h", "i"): 1,
        ("a", "#"): 1,
        ("s", "i"): 1,
        ("i", "m"): 1,
        ("m", "p"): 1,
        ("p", "l"): 1,
        ("l", "e"): 1,
        ("e", "#"): 1,
        ("t", "e"): 1,
        ("e", "x"): 1,
        ("x", "t"): 1,
        ("t", "."): 1,
        (".", "#"): 1,
    })
    return counts


def test_init_vocab(sample_bpe):
    current_vocab = {"t", "s", "h", "T", "e", "i", ".", "a", "p", "x", "l", "m"}
    assert sample_bpe.vocab == current_vocab


def test_split_by_whitespace(sample_bpe):
    target = ["This#", "is#", "a#", "simple#", "text.#"]
    sample_split = sample_bpe._split_by_whitespace()
    assert target == sample_split


def test_get_bigrams(sample_bpe):
    bigrams = [
        [("T", "h"), ("h", "i"), ("i", "s"), ("s", "#")],
        [("i", "s"), ("s", "#")],
        [("a", "#")],
        [("s", "i"), ("i", "m"), ("m", "p"), ("p", "l"), ("l", "e"), ("e", "#")],
        [("t", "e"), ("e", "x"), ("x", "t"), ("t", "."), (".", "#")],
    ]
    assert sample_bpe._get_bigrams() == bigrams


def test_count_bigrams(sample_bpe, counts):
    assert sample_bpe.count_bigrams() == counts


def test_add_most_common(sample_bpe, counts):
    current_vocab = {"t", "s", "h", "T", "e", "i", ".", "a", "p", "x", "l", "m"}
    assert sample_bpe.current_vocab_size == 12
    assert current_vocab == sample_bpe.vocab

    most_common = sample_bpe.add_most_common(counts)
    current_vocab.add("is")

    assert most_common == ("i", "s")
    assert sample_bpe.current_vocab_size == 13
    assert current_vocab == sample_bpe.vocab


def test_merge_most_common_1_iter(sample_bpe, counts):
    most_common = sample_bpe.add_most_common(counts)
    sample_bpe.merge_most_common(most_common)

    sample_bigrams = [
        [("T", "h"), ("h", "is"), ("is", "#")],
        [("is", "#")],
        [("a", "#")],
        [("s", "i"), ("i", "m"), ("m", "p"), ("p", "l"), ("l", "e"), ("e", "#")],
        [("t", "e"), ("e", "x"), ("x", "t"), ("t", "."), (".", "#")],
    ]

    assert sample_bpe.current_bigrams == sample_bigrams


def test_merge_most_common_2_iter(sample_bpe):
    for _ in range(5):
        counts = sample_bpe.count_bigrams()
        most_common = sample_bpe.add_most_common(counts)
        sample_bpe.merge_most_common(most_common)

    result_bigrams = [
        [("s", "i"), ("i", "m"), ("m", "p"), ("p", "l"), ("l", "e"), ("e", "#")],
        [("t", "e"), ("e", "x"), ("x", "t"), ("t", "."), (".", "#")],
    ]

    result_vocab = {"t", "s", "h", "T", "e", "i", ".", "a", "p", "x", "l", "m", "is", "is#", "Th", "This#", "a#"}

    assert sample_bpe.current_bigrams == result_bigrams
    assert sample_bpe.vocab == result_vocab
