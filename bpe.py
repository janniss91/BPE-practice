import re
from collections import Counter


class BytePairEncoder:
    def __init__(self, text: str, vocab_size: int):
        self.text = text
        self.vocab_size = vocab_size
        self.vocab = set(text)
        self.current_vocab_size = len(self.vocab)
        self.candidates = self._count_bigrams()

    def _count_bigrams(self):
        bigram_counts = Counter(
            [
                text[position] + text[position + 1]
                for position in range(len(self.text) - 1)
                if " " not in (text[position], text[position + 1])
                and position + 1 < len(text)
            ]
        )
        return bigram_counts

    def add_most_common_token(self):
        most_common_token = self.candidates.most_common(1)[0][0]
        self.vocab.add(most_common_token)
        return most_common_token

    def find_new_candidates(self, stored_token):
        matches = re.finditer(stored_token, self.text)
        new_candidates = [
            text[match.start() : match.end() + 1]
            for match in matches
            if match.end() < len(text) and " " not in text[match.start() : match.end() + 1]
        ]
        self.candidates.update(new_candidates)
        del self.candidates[stored_token]

    def encode(self):
        while self.current_vocab_size < self.vocab_size:
            stored_token = self.add_most_common_token()
            self.find_new_candidates(stored_token)
            self.current_vocab_size += 1


if __name__ == "__main__":
    text = "This is a long text but at the moment it is not as long as it should be. In the future it will become longer and this is good. For now it is ok if it is kept as long as this."

    bpe = BytePairEncoder(text, 150)
    bpe.encode()

    print(bpe.vocab)
    print(len(bpe.vocab))
