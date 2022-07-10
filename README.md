# Text Segmentation Long Documents

The repository contains my research of approaches for solving the problem of automatic segmentation of long texts, namely, splitting into paragraphs and sections.

# Running the model
+ To run code in Google Colab, open `segmentation_research.ipynb`, run the first cell to clone the repository and then follow the instructions to get data.
+ As steps, you will need, first train model 
```
python train.py --config --emb_type --binary_mode --sec_model
```
1) *The config file must be passed as command line parameters.*
2) *emb_type*: 'SBERT' or 'LABSE'
3) *binary_mode*: choosing a classification task default=None - 3 classes, binary for paragraph - 'paragraph', binary for sections - 'section'. 
4) *sec_model*: default='BERT', also available 'LSTM'.

+ And of course, evaluation step, where you can see the result of training the model on the selected metrics
```
python test.py --config --emb_type --binary_mode --sec_model
```
___
## Google Colab:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grgera/Text-Segmentation/blob/main/segmentation_research.ipynb)
___
## References
In my implementation there may be constructions from papers:
+ [Sentence-BERT](https://arxiv.org/pdf/1908.10084.pdf)
+ [LaBSE](https://arxiv.org/pdf/2007.01852.pdf)
+ [Text Segmentation](https://aclanthology.org/2020.emnlp-main.380.pdf)

