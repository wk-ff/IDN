# Inverse Discriminative Networks

Non-official implement for IDN, paper:

> P. Wei, H. Li and P. Hu. Inverse Discriminative Networks for Handwritten Signature Verification. [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Inverse_Discriminative_Networks_for_Handwritten_Signature_Verification_CVPR_2019_paper.pdf)



## Dataset

[CEDAR](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar): English signature dataset

[BHSig260](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E): Bengali and Hindi signature dataset

Put dataset in `./dataset` and run `./dataset/preprocess.py` to resize and prepare pairs for training.