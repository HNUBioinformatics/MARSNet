# MARSNet: Multiple efficient convolutional attention residual shrinkage networks to predict RBPs’ binding sites in RNA sequences

****
**Introduction**

  In this study, a novel multi-efficient convolutional attention residual contraction network model, MARSNet, was constructed, in which the residual contraction network uses soft thresholding to remove noise data in RNA sequences and identify RBPs binding sites with high accuracy. MARSNet combines efficient channel attention (ECA) and convolutional block attention mechanism (CBAM). The combination of efficient channel attention (ECA) and convolutional block attention mechanism (CBAM) can automatically identify key information in RNA sequences.
****

****
**Requirements**
* pytorch 1.8.1
* python  3.8.5
****
**datasets**

Download and unzip training and test data:http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2
****
**trian**
```
python mian.py 
```
**detect motif**
```
python Detect_motif.py 
```
****           
**Notice**

If MARSNet does not converge in your datasets, you can replace Adam with SGD or RMSprop.



