# BELT-Fusion 代码提取说明

本文档说明了如何从完整项目中提取 BELT-Fusion 的核心代码并在其他项目中使用。

## 📦 核心组件

BELT-Fusion 的核心代码已经提取到以下模块中：

### 1. Agent-Level Uncertainty (`belt_fusion/models/uncertainty_heads/`)

**文件**: [`probabilistic_head.py`](belt_fusion/models/uncertainty_heads/probabilistic_head.py)

包含三个关键类：

- **`ProbabilisticRegressionHead`**: 回归不确定性量化
  - 预测均值和方差（异方差高斯模型）
  - 损失函数：$L = \frac{1}{2} e^{-s} \|y_{gt} - u\|^2 + \frac{1}{2} s$
  
- **`EvidentialClassificationHead`**: 分类不确定性量化（证据深度学习）
  - 基于 Dirichlet 分布和主观逻辑
  - 输出证据值、Dirichlet 参数和不确定性
  
- **`ProbabilisticDetectionHead`**: 组合的检测头
  - 可同时用于 3D 目标检测
  - 即插即用，可替换现有检测器的标准头

**使用示例**:
```python
from belt_fusion.models import ProbabilisticDetectionHead

# 创建概率检测头
head = ProbabilisticDetectionHead(
    in_channels=256,   # 骨干网络输出通道数
    num_classes=3,     # 类别数
    num_regs=7         # 回归参数数量
)

# 前向传播
outputs = head(features)
# outputs 包含:
#   - reg_mean: 回归预测
#   - reg_log_var: 回归不确定性（对数方差）
#   - evidence: 分类证据
#   - alpha: Dirichlet 参数
#   - cls_uncertainty: 分类不确定性

# 计算损失
losses = head.compute_loss(outputs, reg_target, cls_target, epoch)
```

### 2. Fusion-Level Uncertainty (`belt_fusion/models/fusion_modules/`)

**文件**: [`uncertainty_fusion.py`](belt_fusion/models/fusion_modules/uncertainty_fusion.py)

包含三个关键类：

- **`RegressionUncertaintyQuantifier`**: 回归不确定性量化
  - 使用 Mahalanobis 距离测量两个高斯分布的差异
  
- **`ClassificationUncertaintyQuantifier`**: 分类不确定性融合
  - 基于 Dempster-Shafer 证据理论
  - 支持多智能体质量分配融合
  
- **`UncertaintyAwareAdaptiveFusion`**: 完整的自适应融合模块
  - 对象匹配（匈牙利算法）
  - 基于不确定性的最优配对选择
  - 加权融合边界框
  - **无需重新训练，即插即用**

**使用示例**:
```python
from belt_fusion.models import UncertaintyAwareAdaptiveFusion

# 初始化融合模块
fusion = UncertaintyAwareAdaptiveFusion(
    num_classes=3,
    score_threshold=0.3,
)

# 准备多个智能体的检测结果
agent_detections = [
    {
        'boxes': boxes_1,        # (N, 7) [x, y, z, l, w, h, yaw]
        'scores': scores_1,      # (N, num_classes)
        'covariances': cov_1,    # (N, 7, 7) 协方差矩阵
        'evidence': evidence_1,  # (N, num_classes) 证据值
    },
    # ... 更多智能体
]

# 执行融合
fused_results = fusion(agent_detections)
# fused_results 包含:
#   - fused_boxes: 融合后的边界框
#   - cls_belief: 融合后的分类置信度
#   - cls_uncertainty: 融合后的分类不确定性
#   - reg_uncertainty: 回归不确定性（Mahalanobis 距离）
```

### 3. 数据集工具 (`belt_fusion/datasets/`)

- **`DAIRV2XDataset`**: DAIR-V2X 数据集加载器
- **`OPV2VDataset`**: OPV2V 数据集加载器
- **`build_dataset`**: 数据集构建工厂函数

## 🔧 集成到现有项目

### 方法 1: 作为 Python 包安装

```bash
cd BELT-Fusion
pip install -e .
```

然后在你的项目中使用：
```python
from belt_fusion.models import ProbabilisticDetectionHead, UncertaintyAwareAdaptiveFusion
```

### 方法 2: 复制核心文件

如果不想安装整个包，可以直接复制核心文件到你的项目：

```bash
# 复制不确定性头
cp belt_fusion/models/uncertainty_heads/probabilistic_head.py your_project/

# 复制融合模块
cp belt_fusion/models/fusion_modules/uncertainty_fusion.py your_project/
```

然后直接导入：
```python
from probabilistic_head import ProbabilisticDetectionHead
from uncertainty_fusion import UncertaintyAwareAdaptiveFusion
```

### 方法 3: 与 MMDetection3D 集成

如果你使用 MMDetection3D，可以修改配置文件：

```python
# configs/your_config.py
model = dict(
    bbox_head=dict(
        type='ProbabilisticDetectionHead',
        in_channels=256,
        num_classes=3,
        num_regs=7,
        loss_cfg=dict(
            type='UncertaintyLoss',
            reg_weight=1.0,
            cls_weight=1.0,
        ),
    ),
    fusion_module=dict(
        type='UncertaintyAwareAdaptiveFusion',
        num_classes=3,
        score_threshold=0.3,
    ),
)
```

## 📊 训练流程

### 步骤 1: 数据准备

```bash
# 组织 DAIR-V2X 数据
data/
└── DAIR-V2X/
    ├── v2x-cooperative/
    │   ├── image/
    │   ├── pointcloud/
    │   └── label/
    └── v2x_infos_train.pkl
```

### 步骤 2: 训练模型

```bash
# 单 GPU 训练
python tools/train.py \
    --dataset dair-v2x \
    --data-root data/DAIR-V2X/ \
    --ann-file data/DAIR-V2X/v2x_infos_train.pkl \
    --backbone PointPillars \
    --epochs 50 \
    --work-dir ./work_dirs/belt_fusion

# 多 GPU 训练
./tools/dist_train.sh configs/belt_fusion_pointpillars_dairv2x.py 8
```

### 步骤 3: 评估

```bash
python tools/test.py \
    --config configs/belt_fusion_pointpillars_dairv2x.py \
    --checkpoint work_dirs/belt_fusion/checkpoint_epoch_50.pth \
    --data-root data/DAIR-V2X/ \
    --ann-file data/DAIR-V2X/v2x_infos_val.pkl
```

## 🎯 关键优势

1. **模块化设计**: 每个组件都是独立的，可以轻松集成
2. **即插即用**: 融合模块不需要重新训练
3. **最小开销**: 仅增加约 8ms 推理时间
4. **显著提升**: 在噪声环境下提升 7-12% AP

## 📝 注意事项

1. **依赖项**: 确保安装了 PyTorch 1.9+ 和必要的依赖
2. **GPU 内存**: 概率头会增加少量内存使用（约 10%）
3. **超参数**: 可能需要根据你的数据集调整阈值
   - `score_threshold`: 默认 0.3
   - `nms_iou_threshold`: 默认 0.1
   - `anneal_epoch`: KL 退火周期，默认 10

## 🔍 调试技巧

如果遇到训练不稳定：

1. 检查梯度是否爆炸：添加梯度裁剪
2. 降低学习率或使用 warmup
3. 增加 KL 正则化权重
4. 确保数值稳定性（epsilon 设置）

## 📞 联系

如有问题，请提交 GitHub Issue 或联系作者：
- Email: zhc@tongji.edu.cn
- GitHub: https://github.com/ZhiguoZhao/BELT-Fusion
