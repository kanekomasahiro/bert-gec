# Can Encoder-Decoder Models Benefit from Pre-trained Language Representation in Grammatical Error Correction?
Code for the paper: "Can Encoder-Decoder Models Benefit from Pre-trained Language Representation in Grammatical Error Correction?" (In ACL 2020).
If you use any part of this work, make sure you include the following citation:
```
@inproceedings{Kaneko:ACL:2020,
    title={Can Encoder-Decoder Models Benefit from Pre-trained Language Representation in Grammatical Error Correction?},
    author={Masahiro Kaneko, Masato Mita, Shun Kiyono, Jun Suzuki and Kentaro Inui},
    booktitle={Proc. of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2020}
}
```
## Requirements
- python >= 3.5
- torch == 1.2.0
- [bert-nmt](https://github.com/bert-nmt/bert-nmt)
- [subword](https://github.com/rsennrich/subword-nmt)
- [gec-pseudodata](https://github.com/butsugiri/gec-pseudodata)

## How to use
- First download the necessary tools using the following command:
```
cd scripts
./setup.sh
```
- It is necessary to create train.src, train.trg, valid.src, valid.trg, test.src and test.trg from the previously acquired [data](https://www.cl.cam.ac.uk/research/nl/bea2019st/) in the `data` directory.
- You can train the BERT-GEC model with the following command:
```
./train.sh
```
- You can also correct your ungrammatical data with the following command:
```
./generate.sh /path/your/data
```

## License
See the LICENSE file
