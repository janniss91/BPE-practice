from collections import Counter
from typing import Dict


CORPUS = "This is a long text but at the moment it is not as long as it should be. In the future it will become longer and this is good. For now it is ok if it is kept as long as this."

class BytePairEncoder:
    # TODO: Merge operation needs to be refined.
    def __init__(self, corpus: str, vocab_size: int):
        """
        An implementation of the Byte-Pair Encoding.

        The corpus must be inserted as a single string of text.
        """
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.vocab = set("".join(corpus.split()))  # Add all single symbols to the vocabulary.
        self.current_vocab_size = len(self.vocab)
        self.current_split = self._split_by_whitespace()
        self.current_bigrams = self._get_bigrams()

    def _split_by_whitespace(self):
        whitespace_split = self.corpus.split()
        return [token + "#" for token in whitespace_split]

    def _get_bigrams(self):
        return [[(token[idx], token[idx + 1]) for idx in range(len(token)) if idx < len(token) - 1] for token in self.current_split]

    def count_bigrams(self) -> Counter:
        """
        In order to set up initial candidates for the next most common token, 
        first set up all bigrams and count their occurrences.

        :return: The initial Counter object with bigram candidates.
        """
        flat = [bigram for token in self.current_bigrams for bigram in token]
        bigram_counts = Counter(flat)
        return bigram_counts

    def add_most_common(self, bigram_counts: Counter) -> str:
        """
        Add the most common token from the candidate Counter.

        :return: The most common token that was added.
        """
        most_common_token = bigram_counts.most_common(1)[0][0]
        self.vocab.add("".join(most_common_token))
        self.current_vocab_size += 1
        return most_common_token

    def merge_most_common(self, most_common_bigram):
        new_current_bigrams = []
        for token in self.current_bigrams:
            new_token = []
            last_mc = None
            for idx, bigram in enumerate(token):
                # Ignore the tuple after the new token.
                if bigram == most_common_bigram:
                    # Store the occurrence of the bigram to prevent the next token from being used.
                    last_mc = idx
                    new_piece = "".join(bigram)
                    # Account for beginning of string.
                    if idx > 0:
                        previous_piece = token[idx-1][0]
                        new_token.append((previous_piece, new_piece))
                    # Remove the tuple before the new token.
                        new_token.pop(-2)
                    # # Account for end of string.
                    if idx < len(token) - 1:
                        next_piece = token[idx+1][1]
                        new_token.append((new_piece, next_piece))
                # Simply append all other tokens.
                else:
                    if last_mc != idx - 1:
                        new_token.append(bigram)
            
            # Only keep track of the token when it has not been entirely resolved.
            if new_token:
                new_current_bigrams.append(new_token)

        self.current_bigrams = new_current_bigrams

    def encode(self):
        """
        Encode the text in order to obtain a vocabulary.
        """
        while self.current_vocab_size < self.vocab_size:
            counts = self.count_bigrams()
            if len(counts.items()) == 0:
                print("The maximum number of merges for this corpus has been reached.")
                print("Vocabulary Size: {}".format(self.current_vocab_size))
                break
            most_common = self.add_most_common(counts)
            self.merge_most_common(most_common)


if __name__ == "__main__":
    # Example text for initial testing.

    bpe = BytePairEncoder(CORPUS, 70)
    print(bpe.current_vocab_size)
    bpe.encode()

    print(bpe.vocab)
    print(len(bpe.vocab))

    # print(bpe.current_bigrams)
    # print(bpe.vocab)