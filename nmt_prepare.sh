#!/bin/sh
TEXT=data
DATADIR=data-bin/nmt_en_vi
TRAIN=trainings/nmt_en_vi

# download, unzip, clean and tokenize dataset. 
# python ./preprocess/wmt.py

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
MOSESDECODER=./mosesdecoder
$MOSESDECODER/scripts/training/clean-corpus-n.perl $TEXT/train en vi $TEXT/train.clean 1 80
$MOSESDECODER/scripts/training/clean-corpus-n.perl $TEXT/valid en vi $TEXT/valid.clean 1 80
$MOSESDECODER/scripts/training/clean-corpus-n.perl $TEXT/test en vi $TEXT/test.clean 1 80

# build subword vocab
SUBWORD_NMT=./subword-nmt
NUM_OPS=32000

# learn codes and encode separately
CODES=codes.${NUM_OPS}.bpe
echo "Encoding subword with BPE using ops=${NUM_OPS}"
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.clean.en > $TEXT/${CODES}.en
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.clean.vi > $TEXT/${CODES}.vi

echo "Applying vocab to training"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en < $TEXT/train.clean.en > $TEXT/train.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.vi < $TEXT/train.clean.vi > $TEXT/train.${NUM_OPS}.bpe.vi

VOCAB=vocab.${NUM_OPS}.bpe
echo "Generating vocab: ${VOCAB}.en"
cat $TEXT/train.${NUM_OPS}.bpe.en | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.en

echo "Generating vocab: ${VOCAB}.vi"
cat $TEXT/train.${NUM_OPS}.bpe.vi | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.vi

# encode validation
echo "Applying vocab to valid"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.en < $TEXT/valid.clean.en > $TEXT/valid.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.vi --vocabulary $TEXT/${VOCAB}.vi < $TEXT/valid.clean.vi > $TEXT/valid.${NUM_OPS}.bpe.vi

# encode test
echo "Applying vocab to test"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.en < $TEXT/test.clean.en > $TEXT/test.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.vi < $TEXT/test.clean.vi > $TEXT/test.${NUM_OPS}.bpe.vi

# generate preprocessed data
echo "Preprocessing datasets..."
DATADIR=data-bin/nmt_en_vi
rm -rf $DATADIR
mkdir -p $DATADIR
fairseq-preprocess --source-lang en --target-lang vi \
     --trainpref $TEXT/train.${NUM_OPS}.bpe --validpref $TEXT/valid.${NUM_OPS}.bpe --testpref $TEXT/test.${NUM_OPS}.bpe \
     --thresholdsrc 3 --thresholdtgt 3 --destdir $DATADIR