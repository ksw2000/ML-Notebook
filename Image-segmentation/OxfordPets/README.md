
## ResNet

|                     | val_loss   | # of paramters |
|---------------------|------------|----------------|
|ResNet50 U-Net       | 0.3075     | 35.8M          |
|ResNet50 U-Net v2    | 0.3043     | 35.9M          |
|ResNet50 U-Net v3    | **0.3013** | 25.1M          |
|SEResNeXt50 U-Net    | 0.3164     | 111.7M         |
|SEResNeXt50 U-Net v2 | 0.4114     | 112.6M         |
|SEResNeXt101 U-Net   | 0.3265     | 133.1M         |
|SEResNeXt101 U-Net v2| 0.3039     | 134.0M         |

## SKNet

**kernel=3 + kernel=5**

|                     | val_loss   | # of paramters |
|---------------------|------------|----------------|
|SKNet50 U-Net        | 0.3313     | 125.2M         |
|SKNet50 U-Net v2     | 0.3856     | 126.1M         |
