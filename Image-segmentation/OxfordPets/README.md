原版本採用 Hypercolumns + CBAM 的技巧來降低 loss

+ v2: 將 CBAM 由 Decoder 的部分往前移置 Encoder 時就執行
+ v3: 在 v2 的基礎上，更動 Decoder 將 Decoder 中 Upsample 的部分改由反卷積實作 (可減少參數量)
+ v4: 在 v1 的基礎上，更動 Decoder 將 Decoder 中 Upsample 的部分改由反卷積實作 (可減少參數量)。與 v3 的差別在於本版本不去更動 CBAM 的位置

## ResNet

|               | val_loss   | # of paramters |
|---------------|------------|----------------|
|ResNet34       | 0.3075     | 35.8M          |
|ResNet34 v2    | 0.3043     | 35.9M          |
|ResNet34 v3    | **0.3013** | 25.1M          |
|ResNet34 v4    | 0.3212     | 25.0M          |
|SEResNeXt50    | 0.3164     | 111.7M         |
|SEResNeXt50 v2 | 0.4114     | 112.6M         |
|SEResNeXt101   | 0.3265     | 133.1M         |
|SEResNeXt101 v2| 0.3039     | 134.0M         |

## SKNet

**kernel=3 + kernel=5**

|               | val_loss   | # of paramters |
|---------------|------------|----------------|
|SKNet50        | 0.3313     | 125.2M         |
|SKNet50 v2     | 0.3856     | 126.1M         |
