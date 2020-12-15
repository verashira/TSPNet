import argparse

from bpemb import BPEmb


parser = argparse.ArgumentParser(description="Preprocess sign dataset")
parser.add_argument('--vocab-size', default=25000, help="size of pretrained word embedding vocabulary")
parser.add_argument('--dim', default=300, help="dim of word embedding")
parser.add_argument('--save-vecs', default=None, help="save path for extracted word embedding")
parser.add_argument('input_file', )
parser.add_argument('output_file', )

args = parser.parse_args()


def process(texts, vocab_size=25000, dim=300):
    emb = BPEmb(lang='de', vs=vocab_size, dim=dim)

    texts = [emb.encode(t) for t in texts]

    unique_words = set([w for t in texts for w in t])
    vecs = [wv for  (i, wv) in enumerate(zip(emb.words, emb.vectors)) 
        if i < 3 or wv[0] in unique_words]  # reserve the special tokens

    return texts, vecs


if __name__ == "__main__":
    print (args)
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        texts = [l.strip() for l in f.readlines()]

    texts, vecs = process(texts, args.vocab_size, args.dim)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(' '.join(t) + '\n')

    if args.save_vecs:
        with open(args.save_vecs, 'w', encoding='utf-8') as f:
            f.write('{} {}\n'.format(len(vecs), 300))
            for w,v in vecs:
                # print (w, v)
                f.write('{} {}\n'.format(w, ' '.join([str(n) for n in v])))


