import sys
from multiprocessing import Pool
import multiprocessing as multi
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok


def detok(input):
    tokens = input.split()
    detokenizer = Detok()
    text = detokenizer.detokenize(tokens)
    text = re.sub('\s*,\s*', ', ', text)
    text = re.sub('\s*\.\s*', '. ', text)
    text = re.sub('\s*\?\s*', '? ', text)
    text = text.strip()

    return text


def main(args):
    data = [l.strip() for l in open(args[1])]
    p = Pool(multi.cpu_count())
    data = p.map(detok, data)
    p.close()
    fw = open(args[2], 'w')
    for l in data:
        fw.write(l + '\n')

if __name__ == '__main__':
    args = sys.argv
    main(args)
