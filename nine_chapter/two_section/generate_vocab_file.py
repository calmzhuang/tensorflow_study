import codecs
import collections
from operator import itemgetter


raw_data = '../../path/to/data/ptb.train.txt'
vacab_output = 'ptb.vocab'

counter = collections.Counter()
with codecs.open(raw_data, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(counter.items(),
                            key=itemgetter(1),
                            reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

sorted_words = ["<eos>"] + sorted_words

with codecs.open(vacab_output, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")