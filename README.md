# MARSNet:基于卷积注意力机制的多重残差收缩网络预测RNA-蛋白质结合位点

****
**环境**
* pytorch 1.8.1
* python  3.8.5
****
**数据**

下载并解压训练和测试数据:http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2
****
**训练方法**
```
python train_sum.py 
```

****           
**注意**

如果MARSNet在你的数据集中无法收敛,可以将Adam替换成SGD或者RMSprop.



