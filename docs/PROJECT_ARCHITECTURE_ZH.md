# DiffPhysDrone 项目架构解读（含神经网络框架）

本文从“训练入口 → 仿真环境 → 神经网络 → 损失与优化 → CUDA 扩展”五个层次梳理项目结构，便于你快速定位与二次开发。

## 1. 项目总体分层

- **训练编排层（Python）**：`main_cuda.py`
  - 负责参数解析、环境创建、模型创建、训练循环、损失计算、日志和保存。
- **策略网络层（PyTorch）**：`model.py`
  - 输入视觉特征（深度图）+ 状态向量（速度/姿态/目标方向等），输出控制相关向量。
- **仿真环境层（Python + CUDA）**：`env_cuda.py` + `quadsim_cuda`
  - `Env` 在 Python 侧管理场景随机化与状态；核心动力学与渲染通过 CUDA 扩展实现。
- **算子绑定层（C++）**：`src/quadsim.cpp`
  - 用 pybind11 将 `render/find_nearest_pt/update_state_vec/run_forward/run_backward` 暴露给 Python。
- **CUDA 内核层**：`src/quadsim_kernel.cu`、`src/dynamics_kernel.cu`
  - 承担高性能渲染与动力学前后向计算。

## 2. 训练主流程（main_cuda.py）

### 2.1 初始化阶段

- 通过 argparse 定义训练超参数（batch size、迭代数、各 loss 系数、环境随机化开关等）。
- 固定使用 `cuda` 设备。
- 创建环境：`Env(args.batch_size, 64, 48, ...)`。
- 根据 `--no_odom` 决定模型观测维度：
  - 有里程计：`Model(10, 6)`（即 `7+3`）
  - 无里程计：`Model(7, 6)`
- 优化器与学习率：`AdamW + CosineAnnealingLR`。

### 2.2 单次迭代逻辑

每个训练迭代（`for i in range(num_iters)`）主要步骤：

1. `env.reset()` 随机场景与动力学参数；`model.reset()`（目前为空实现）。
2. 循环 `timesteps` 个控制步：
   - `depth, flow = env.render(ctl_dt)` 获取观测（当前默认第二返回为事件或 None）。
   - 构造目标速度、局部坐标系状态向量 `state`。
   - 深度预处理：`x = 3 / depth.clamp_(0.3,24) - 0.6 + noise`，再 `max_pool2d`。
   - `act, values, h = model(x, state, h)` 得到网络输出和 GRU 隐状态。
   - 将网络输出映射回世界系，计算控制量并 `env.run(...)` 推进仿真。
3. 轨迹结束后计算多项损失：速度跟踪、碰撞/避障、控制平滑（acc/jerk/snap）、速度预测等。
4. `loss.backward()` 反向，`optim.step()` + `sched.step()`。
5. TensorBoard 记录统计；周期性保存 checkpoint。

## 3. 神经网络框架（model.py）

`Model` 是一个“**视觉编码器 + 状态融合 + 时序记忆 + 控制头**”的轻量策略网络：

1. **视觉编码器 `stem`**
   - 3 层卷积（`1->32->64->128`）+ LeakyReLU。
   - 展平后接线性层到 192 维。
2. **状态投影 `v_proj`**
   - 将低维状态向量（7 或 10 维）线性投影到 192 维。
3. **融合方式**
   - 直接做 `img_feat + v_proj(v)`，再激活。
4. **时序模块**
   - `GRUCell(192,192)` 维护隐状态 `hx`，让策略具备短时记忆。
5. **动作头**
   - `fc(192 -> dim_action)`，训练脚本中 `dim_action=6`。

`forward` 返回 `(act, None, hx)`，其中第二项当前未使用（训练脚本里记为 `values`）。

## 4. 观测、状态与动作语义（训练脚本视角）

- **视觉输入**：单通道深度图（池化后尺寸更小，便于实时训练）。
- **状态输入 state**：
  - （可选）机体局部速度 `local_v`
  - 目标方向速度（转到局部坐标系）
  - 姿态相关项 `env.R[:, 2]`
  - 安全边界 `env.margin`
- **网络输出拆分**：
  - `a_pred`、`v_pred` 等由 `act.reshape(B,3,-1)` 与旋转矩阵变换得到。
  - 其中 `v_pred` 被用于速度预测损失 `loss_v_pred`。

这本质是“端到端控制 + 辅助自监督（速度预测）”的联合训练。

## 5. 仿真环境架构（env_cuda.py）

### 5.1 Env 负责什么

- 维护场景几何（球、体素、圆柱）、无人机状态、风扰、相机参数。
- 在 `reset()` 中做大量 domain randomization：障碍布局、最大速度、尺寸、姿态偏置等。
- `render()` 调用 `quadsim_cuda.render` 生成深度画布。
- `run()` 调用自定义 autograd 的 `run_forward/run_backward` 完成可微动力学推进。

### 5.2 自定义可微动力学

- `RunFunction(torch.autograd.Function)` 封装了动力学前向与反向：
  - `forward` 调 `quadsim_cuda.run_forward`
  - `backward` 调 `quadsim_cuda.run_backward`
- 这使得控制输出对长期轨迹损失可端到端反传。

### 5.3 事件相机部件（当前仓库新增）

- `DepthEventCamera` 使用连续深度帧构建对数逆深度变化，并按正负阈值累计为两通道事件计数（pos/neg）。
- 在 `Env(..., use_event_camera=True)` 时启用；`render()` 将返回 `(canvas, events)`。
- 当前训练脚本尚未消费 `events`，因此默认训练路径依旧是深度图驱动。

## 6. CUDA 扩展如何接入 Python

- `src/setup.py` 通过 `CUDAExtension` 编译三个文件：
  - `quadsim.cpp`（绑定）
  - `quadsim_kernel.cu`（渲染/几何相关）
  - `dynamics_kernel.cu`（动力学相关）
- 在 `quadsim.cpp` 中通过 `PYBIND11_MODULE` 暴露函数给 Python 模块 `quadsim_cuda`。
- Python 侧 `env_cuda.py` 直接调用这些函数，构成“PyTorch 训练循环 + 自定义 CUDA 算子”闭环。

## 7. 你最关心的“神经网络框架结论”

简要总结：

- 这不是复杂的 Transformer 或大型视觉 backbone，而是**轻量 CNN + GRUCell 的时序控制网络**。
- 训练核心不在“网络很深”，而在“**可微物理仿真 + 大规模随机化 + 多项任务损失**”。
- 网络输出既承担控制，又承担辅助速度估计（`loss_v_pred`），增强可观测性不足场景下的学习稳定性。
- 如果你要扩展事件相机，建议在现有架构中将事件表征与深度表征做双分支融合，再复用现有 GRU 控制头。

## 8. 推荐的扩展切入点

1. **输入层改造**：在 `Model` 新增事件分支 CNN，再与当前 `stem` 融合。
2. **训练脚本改造**：在 `main_cuda.py` 使用 `events`，构建事件窗口/体素并送入模型。
3. **损失设计**：为事件分支增加重建或对比学习辅助损失，减小仅凭控制损失带来的稀疏梯度。
4. **算子层升级**：若需更物理真实事件，直接在 CUDA 渲染核中输出亮度或光流，再做事件触发。
