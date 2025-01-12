# 实现强化学习DPO训练代码

## Quick start
```python
python dpo_train.py
```

## 说明
。

精度设置fp16可能会出现loss 为nan的现象

```dpo_train.py```为训练主路径， 相关loss计算在```loss.py```.

某个pair上win/lose比较低有两种原因，一种是在两个pair得分差不多的情况下，原先回答较好将会抑制在ref模型表现较差的pair上的优化，我们的方法将会导致在表现较好的pair上优化不多，trade off
Pair之间得分差距大，导致概率差很大，得分差距大的概率很低，从而优化受到抑制，得分地的过优化，不利于泛化性
从模型在质量相差差不多的pair和pair质量相差大的两种情况入手
无论是哪种情况都有提升，
在sft loss扩展下已经取得验证
在simpo解决长度问题
长度问题原因 越长得分越好，simpo越长降低reward，降低reward后近似解决不同pair的得分不同问题，
泛化性观点，当数据集表现很好时，win lose都降低有助于泛化性，
