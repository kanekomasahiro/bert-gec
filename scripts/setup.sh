
mkdir -p ../data

echo Loading bert-nmt
git clone https://github.com/bert-nmt/bert-nmt.git ../bert-nmt

echo Loading subword
git clone https://github.com/rsennrich/subword-nmt.git ../subword

echo Loading gec-pseudodata
git clone https://github.com/butsugiri/gec-pseudodata.git ../gec-pseudodata

echo Loading pre-trained GEC model
mkdir -p ../pseudo_model
wget -P ../pseudo_model https://gec-pseudo-data.s3-ap-northeast-1.amazonaws.com/ldc_giga.spell_error.pretrain.checkpoint_last.pt

echo Loading pre-trained BERT model
mkdir -p ../bert-base-cased
wget "https://drive.google.com/uc?export=download&id=1wwrTLQ2cg8VYDqXezqZCm6ErJcPKppdm" -O ../bert-base-cased/config.json
wget "https://drive.google.com/uc?export=download&id=1D2YcxaSO-NQN8-HaybGg16v3FELNxOI5" -O ../bert-base-cased/vocab.txt

FILE_ID=1WHQhFaknhZIvkLpj5mIxf07HmOCKgXCH
FILE_NAME=../bert-base-cased/pytorch_model.bin
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
