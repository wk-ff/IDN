# Inverse Discriminative Networks

Non-official implement for IDN, paper:

> P. Wei, H. Li and P. Hu. Inverse Discriminative Networks for Handwritten Signature Verification. [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Inverse_Discriminative_Networks_for_Handwritten_Signature_Verification_CVPR_2019_paper.pdf)



## Dataset

[CEDAR](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar): English signature dataset

[BHSig260](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E): Bengali and Hindi signature dataset

[ChiSig](https://drive.google.com/file/d/176bG9Hp_uX9bJvIFt437wqAbEqEMqsO7/view?usp=sharing): Chinese signature dataset from paper
> Yan, K., Zhang, Y., Tang, H., Ren, C., Zhang, J., Wang, G., Wang, H. (n.d.). Signature Detection, Restoration, and Verification: A Novel Chinese Document Signature Forgery Detection Benchmark. [CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/SketchDL/papers/Yan_Signature_Detection_Restoration_and_Verification_A_Novel_Chinese_Document_Signature_CVPRW_2022_paper.pdf)

[SigComp2011](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)): ICDAR 2011 Signature Verification Competition dataset

Put dataset in `./dataset` and run `./dataset/preprocess.py` to resize and prepare pairs for training.