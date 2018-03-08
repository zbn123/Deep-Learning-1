# Deep-CTR 实践

> 刚开始接触 deep learning 在CTR上的应用。主要从Deep Learning over Multi-field Categorical Data
> – A Case Study on User Response Prediction 和Product-based Neural Networks for User Response
> Prediction 两篇文章入手。

代码参照论文作者的[源码](https://github.com/Atomu2014/product-nets)来写，主要是加了一些便于自己理解的注释。（如若感兴趣，请参考论文作者的源码）

### 文件说明

- utils.py : 数据处理（清洗、格式转换以及一些常用的功能实现）
- deep_ctr.py  : 模型的构建，LR、FM、FNN、IPNN、OPNN
- test_model.py : 用data/ 文件夹下的数据对模型进行训练、测评
- data : 包含模型所需的小型数据集

