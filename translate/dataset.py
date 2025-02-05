import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
import warnings

train_dataset = list(Multi30k(split="train", language_pair=('de','en')))

# python -m spacy download de_core_news_sm
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0,1,2,3
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'

de_tokens = []
en_tokens = []

for de, en in train_dataset:
    de_tokens.append(de_tokenizer(de))
    en_tokens.append(en_tokenizer(en))


de_vocab = build_vocab_from_iterator(de_tokens, specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM], special_first=True)
de_vocab.set_default_index(UNK_IDX)

en_vocab = build_vocab_from_iterator(en_tokens, specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM], special_first=True)
en_vocab.set_default_index(UNK_IDX)


def de_process(de_sentence):
    tokens= [BOS_SYM] + de_tokenizer(de_sentence) + [EOS_SYM]
    ids = de_vocab(tokens)
    return tokens, ids


def en_process(en_sentence):
    tokens= [BOS_SYM] + en_tokenizer(en_sentence) + [EOS_SYM]
    ids = en_vocab(tokens)
    return tokens, ids


if __name__ == "__main__":
    print(f"len de_vocab={len(de_vocab)}, len en_vocab={len(en_vocab)}")
    de, en = train_dataset[0]
    print(de_process(de))
    print(en_process(en))