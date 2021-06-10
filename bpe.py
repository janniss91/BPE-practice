import re
from collections import Counter
from typing import Dict


class BytePairEncoder:
    # TODO: Merge operation needs to be refined.
    def __init__(self, text: str, vocab_size: int):
        """
        An implementation of the Byte-Pair Encoding.
        """
        self.text = text
        self.vocab_size = vocab_size
        self.vocab = set(text)
        self.current_vocab_size = len(self.vocab)
        self.candidates = self._count_bigrams()

    def _count_bigrams(self) -> Dict[str, int]:
        """
        In order to set up initial candidates for the next most common token, 
        first set up all bigrams and count their occurrences.

        :return: The initial Counter object with bigram candidates.
        """
        bigram_counts = Counter(
            [
                text[position] + text[position + 1]
                for position in range(len(self.text) - 1)
                if " " not in (text[position], text[position + 1])
                and position + 1 < len(text)
            ]
        )
        return bigram_counts

    def add_most_common_token(self) -> str:
        """
        Add the most common token from the candidate Counter.

        :return: The most common token that was added.
        """
        most_common_token = self.candidates.most_common(1)[0][0]
        self.vocab.add(most_common_token)
        return most_common_token

    def find_new_candidates(self, stored_token: str):
        """
        Find all possible candidates that result from the recently added token.

        :param stored_token: The most common token that was recently stored.
        """
        matches = re.finditer(stored_token, self.text)

        # Add combinations with the stored token in the beginning.
        new_candidates = [
            text[match.start() - 1:match.end()]
            for match in matches
            if match.start() > 0 and " " not in text[match.start()-1:match.end()]
        ]
        # Add combinations with the stored token in the end.
        new_candidates.extend([
            text[match.start():match.end() + 1]
            for match in matches
            if match.end() < len(text) and " " not in text[match.start():match.end() + 1]
        ])
        
        self.candidates.update(new_candidates)
        del self.candidates[stored_token]

    def encode(self):
        """
        Encode the text in order to obtain a vocabulary.
        """
        while self.current_vocab_size < self.vocab_size:
            stored_token = self.add_most_common_token()
            self.find_new_candidates(stored_token)
            self.current_vocab_size += 1


if __name__ == "__main__":
    # Example text for initial testing.
    text = "This is a long text but at the moment it is not as long as it should be. In the future it will become longer and this is good. For now it is ok if it is kept as long as this."

    bpe = BytePairEncoder(text, 150)
    bpe.encode()

    print(bpe.vocab)
    print(len(bpe.vocab))
