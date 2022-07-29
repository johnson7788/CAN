# When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition

This is the official pytorch implementation of [CAN](https://arxiv.org/abs/2207.11463) (ECCV'2022). 

>*Bohan Li, Ye Yuan, Dingkang Liang, Xiao Liu, Zhilong Ji, Jinfeng Bai, Wenyu Liu, Xiang Bai*

## Abstract

<p align="justify">
最近，大多数手写数学表达式识别（HMER）方法采用了编码器-解码器网络，通过注意力机制直接从公式图像中预测标记序列。
然而，这类方法可能无法准确读取结构复杂的公式或生成长的标记序列，因为由于书写方式或空间布局的巨大方差，
注意力的结果往往是不准确的。为了缓解这个问题，我们提出了一个非常规的HMER网络，名为计数感知网络（CAN），
它共同优化了两个任务。HMER和符号计数。具体来说，我们设计了一个弱监督的计数模块，
可以在没有符号级位置标注的情况下预测每个符号类的数量，然后将其插入一个典型的基于注意力的HMER编码器-解码器模型。
在HMER的基准数据集上的实验验证了联合优化和计数结果都有利于纠正编码器-解码器模型的预测误差，而且CAN的性能一直优于最先进的方法。
特别是，与HMER的编码器-解码器模型相比，所提出的计数模块所造成的额外时间成本是微不足道的。
</p>

## Pipeline

<p align="left"><img src="assets/CAN.png" width="585"/></p>

## Counting Module

<p align="left"><img src="assets/MSCM.png" width="580"/></p>

## Datasets

Download the CROHME dataset from [BaiduYun](https://pan.baidu.com/s/1qUVQLZh5aPT6d7-m6il6Rg) (downloading code: 1234) and put it in ```datasets/```.

The HME100K dataset can be download from the official website [HME100K](https://ai.100tal.com/dataset).

## Training
检查配置文件``config.yaml``并使用CROHME数据集进行训练。
```
python train.py --dataset CROHME
```
默认情况下，"batch size"被设置为8，你可能需要使用32GB内存的GPU来训练你的模型。

## Testing

在配置文件``config.yaml``中填写``checkpoint``（预训练的模型路径），用CROHME数据集进行测试。

```
python inference.py --dataset CROHME
```
注意，测试数据集的路径是在``inference.py``中设置的。

## Citation

```
@inproceedings{CAN,
  title={When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition},
  author={Li, Bohan and Yuan, Ye and Liang, Dingkang and Liu, Xiao and Ji, Zhilong and Bai, Jinfeng and Liu, Wenyu and Bai, Xiang},
  booktitle={ECCV},
  year={2022}
}
```
