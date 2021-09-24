# Byte-Pair Encoding

A primitive but functional implementation of the Byte-Pair encoding in Python.
Note that the training algorithm is rather slow, however, it speeds up as pieces are merged.

## Running

To get a Byte-Pair encoding from your text, you must provide a single plain text file.

Example:

    python3 bpe.py file.txt 100

Arguments:
1. Your input file
2. The desired vocabulary size.

## No libraries needed

No libraries must be installed to run the script, the standard Python library provides the necessary packages.