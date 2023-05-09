# Active Learning for Computational Seismic Interpretation
This repository contains codes implementing active learning sampling strategies for seismic facies interpretation, as described in our works [1] and [2]. 
In addition to the working codes, we provide
- Machine learning models trained over four different sampling strategies
- Mean Intersection-Over-Union (mIOU) values recorded over five cycles for each of four sampling strategies
- Test and train split results 
- Results from three different seeds/initializations for each query strategy

[1] Mustafa, A. and AlRegib, G., 2021, September. Man-recon: Manifold learning for reconstruction with deep autoencoder for smart seismic interpretation. In 2021 IEEE International Conference on Image Processing (ICIP) (pp. 2953-2957). IEEE.
[2] Mustafa, A. and AlRegib, G., 2023. Active Learning with Deep Autoencoders for Seismic Facies Interpretation. Geophysics, 88(4), pp.1-43. 

## Abstract
Machine learning-assisted seismic interpretation tasks require large quantities of labeled data annotated by expert interpreters, which is a costly and time-consuming process. Where existing works to minimize dependence on labeled data assume the data annotation process to already be completed, active learning---a field of machine learning---works by selecting the most important training samples for the interpreter to annotate in real time simultaneously with the training of the interpretation model itself, resulting in high levels of performance with fewer labeled data samples than otherwise possible. Whereas there exists significant literature on active learning for classification tasks with respect to natural images, there exist very little to no works for dense prediction tasks in geophysics like interpretation. We develop a unique and first-of-a-kind active learning framework for seismic facies interpretation using the manifold learning properties of deep autoencoders. By jointly learning representations for supervised and unsupervised tasks and then ranking unlabeled samples by their nearness to the data manifold, we are able to identify the most relevant training samples to be labeled by the interpreter in each training round. On the popular F3 dataset, we obtain close to 10 percentage point difference in terms of interpretation accuracy between the proposed method and the baseline with only three fully annotated seismic sections. 

