# BELT-Fusion 快速开始指南

## 5 分钟快速上手

### 1. 安装（2 分钟）

```bash
# 克隆或进入 BELT-Fusion 目录
cd BELT-Fusion

# 创建 conda 环境
conda create -n belt python=3.8 -y
conda activate belt

# 安装依赖
pip install torch numpy scipy tqdm opencv-python
pip install -e .
```

### 2. 使用预训练模型（1 分钟）

```python
from belt_fusion.models import ProbabilisticDetectionHead, UncertaintyAwareAdaptiveFusion
import torch

# 加载模型
model = ProbabilisticDetectionHead(
    in_channels=256,
    num_classes=3,
    num_regs=7
)

# 加载权重
checkpoint = torch.load('checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()
model.eval()

print("✓ 模型加载完成")
```

### 3. 运行推理（1 分钟）

```python
# 准备输入特征（从你的骨干网络获取）
features = torch.randn(4, 256).cuda()  # (batch_size, channels)

# 前向传播
with torch.no_grad():
    outputs = model(features)

print(f"✓ 预测完成")
print(f"  - 回归均值形状：{outputs['reg_mean'].shape}")
print(f"  - 分类不确定性形状：{outputs['cls_uncertainty'].shape}")
```

### 4. 多智能体融合（1 分钟）

```python
from belt_fusion.models import UncertaintyAwareAdaptiveFusion

# 初始化融合模块
fusion = UncertaintyAwareAdaptiveFusion(num_classes=3)

# 模拟两个智能体的检测
agent1 = {
    'boxes': torch.randn(5, 7),      # 5 个目标
    'scores': torch.rand(5, 3),
    'covariances': torch.eye(7).unsqueeze(0).expand(5, 7, 7),
    'evidence': torch.rand(5, 3) * 10,
}

agent2 = {
    'boxes': torch.randn(5, 7),
    'scores': torch.rand(5, 3),
    'covariances': torch.eye(7).unsqueeze(0).expand(5, 7, 7),
    'evidence': torch.rand(5, 3) * 10,
}

# 执行融合
fused = fusion([agent1, agent2])

print(f"✓ 融合完成，得到 {len(fused)} 个融合目标")
```

## 下一步

- 📖 阅读 [README.md](README.md) 了解完整功能
- 🔧 查看 [EXTRACTION_GUIDE.md](EXTRACTION_GUIDE.md) 学习如何集成到你的项目
- 🏃 运行 `python tools/train.py --help` 查看训练选项
- 📊 使用 `python tools/demo.py` 进行可视化演示

## 常见问题

**Q: 如何将 BELT-Fusion 集成到我的检测器？**  
A: 只需将你的标准检测头替换为 `ProbabilisticDetectionHead`，无需修改骨干网络。

**Q: 融合模块需要重新训练吗？**  
A: 不需要！`UncertaintyAwareAdaptiveFusion` 是即插即用的，可以直接应用于已训练的模型。

**Q: 不确定性量化的计算开销大吗？**  
A: 非常小，仅增加约 8ms 推理时间（~9%），但能带来 7-12% 的精度提升。

**Q: 支持哪些数据集？**  
A: 目前已内置 DAIR-V2X 和 OPV2V 支持，也可轻松适配其他 V2X 数据集。

## 获取帮助

- 📝 查看文档：[GitHub Wiki](https://github.com/ZhiguoZhao/BELT-Fusion/wiki)
- 🐛 报告问题：[GitHub Issues](https://github.com/ZhiguoZhao/BELT-Fusion/issues)
- 📧 联系我们：zhc@tongji.edu.cn

---

祝你使用愉快！🚀
