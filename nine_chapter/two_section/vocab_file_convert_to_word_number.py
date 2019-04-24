import codecs
import sys


raw_data = "../../path/to/data/ptb.train.txt"
vocab_file = "ptb.vocab"
output_data = "ptb.train"

with codecs.open(vocab_file, "r", "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]


fin = codecs.open(raw_data, "r", "utf-8")
fout = codecs.open(output_data, 'w', 'utf-8')
for line in fin:
    words = line.strip().split() + ["<eos>"]
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()