input=$1
gpu=$2
beam=5
bert_type=bert-base-cased
SUBWORD_NMT=../subword
FAIRSEQ_DIR=../bert-nmt
BPE_MODEL_DIR=../gec-pseudodata/bpe
MODEL_DIR=../bert-fuse
OUTPUT_DIR=$MODEL_DIR/output
PREPROCESS=../bert-fuse

mkdir -p $OUTPUT_DIR

$SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $input > $OUTPUT_DIR/test.bpe.src

python -u detok.py $input $OUTPUT_DIR/test.bert.src
paste -d "\n" $OUTPUT_DIR/test.bpe.src $OUTPUT_DIR/test.bert.src > $OUTPUT_DIR/test.cat.src

echo Generating...
CUDA_VISIBLE_DEVICES=$gpu python -u ${FAIRSEQ_DIR}/interactive.py $PREPROCESS \
    --path ${MODEL_DIR}/checkpoint_best.pt \
    --beam ${beam} \
    --nbest ${beam} \
    --no-progress-bar \
    -s src \
    -t trg \
    --buffer-size 1024 \
    --batch-size 32 \
    --log-format simple \
    --remove-bpe \
    --bert-model-name $bert_type \
    < $OUTPUT_DIR/test.cat.src > $OUTPUT_DIR/test.nbest.tok

cat $OUTPUT_DIR/test.nbest.tok | grep "^H"  | python -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if (i % ${beam} == 0) ]); print(x)" | cut -f3 > $OUTPUT_DIR/test.best.tok
sed -i '$d' $OUTPUT_DIR/test.best.tok
