## 顯卡問題

[https://blog.csdn.net/qq_39388410/article/details/108027769](https://blog.csdn.net/qq_39388410/article/details/108027769)

## Pytorch conv padding

Transpose Conv outputsize 計算
```
output = (input - 1) * stride + output_padding - 2*padding + kernel_size
```

## 評估模型的方式

### 準確度 Accuracy

Accuracy = 正確分類數目 / 所有樣本數

最直接的缺點就是當某個分類類別占比太大時，很容易受該類別影響，比如說，如果要做 A B 分類 B 占了 90%，那我的模型只要將所有 data 都分類到 B 上，自然就有 90% 的準確度了

### 精確度與召回率
