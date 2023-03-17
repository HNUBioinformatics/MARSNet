# MARSNet:基于卷积块注意力模块的多重深度残差收缩网络预测RNA-蛋白质结合位点

****
# 环境

* pytorch 1.8.1
* python  3.8.5
****

# 数据
下载并解压训练和测试数据:http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2
****

# 训练并预测模型
```
python train_sum.py 
```

****           
#注意
如果在你的数据集中模型无法收敛,可以改变优化方法,比如: 将Adam替换成SGD或者RMSprop.



