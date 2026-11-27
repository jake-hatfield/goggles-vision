<div align="center">
  <img src="./assets/logo.png" width="120" alt="OpenHoopVision 标志" />
</div>
<h3 align="center"><b>OpenHoopVision</b></h3>
<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/README-English-blue?style=flat-square" alt="English README" /></a>
</p>

**TL;DR：** OpenHoopVision 为业余和青少年篮球提供易用的 AI 工具与比赛数据，服务于篮球从业者、教练及其家人。这个可扩展的框架支持接入更多事件、数据与应用。导出的跟踪数据可进一步用于 AI 分析，帮助球队理解比赛并提升表现。

## 最新更新

**2026-11-02**
- **[AI Highlights](https://github.com/feifeifei85/OpenHoopVision-AI-Highlights)：** OpenHoopVision 的 iOS 配套应用，可在设备本地分析视频、识别可能的进球瞬间，并支持回顾与导出集锦。
- **[OpenHoopVision Shooting](https://github.com/feifeifei85/OpenHoopVision-shooting)：** 基于核心视觉流水线的投篮分析工具，可估计投篮动作、输出回顾视频，并生成时序与姿态指标。

**2026-10-26**
- **升级球员姿态推理：** 用项目内的纯 PyTorch RTMPose-S 替换整帧 YOLOv8n-Pose，复用现有人体检测框批量估计 ROI 姿态，避免重复进行整帧人物检测。
- **提升场地映射可靠性：** 使用项目训练的 MobileNetV4-Conv-Small 场地关键点模型替换 YOLO26s，并增加可见性分数（896×512；PCK@10 为 88.17%，PCK@20 为 94.26%）。短期关键点历史和相机运动补偿有助于在检测稀疏时维持小地图连续性。
- **降低球衣号码误识别：** 微调后的 SVTRv2-B 使用 CTC 空白输出和置信度过滤，拒绝无法可靠识别的图片。

**2026-10-20**
- **更新篮球检测模型：** RF-DETR 权重新增 `ball-in-basket` 类别，为投篮结果判断提供视觉线索。
- **优化球场映射：** 通过单应矩阵校验、失效重定位和相机运动补偿提升映射稳定性。
- **结构化投篮事件导出：** 版本化 JSON 记录时间、结果、球场位置、投篮属性及可用的球员/球队信息。

## 演示

<p align="center">
  <img src="./assets/demo_17.gif" alt="观看演示 17" width="410" />
  <img src="./assets/demo_26.gif" alt="观看演示 26" width="410" />
</p>
<p align="center">
  <img src="./assets/demo_29.gif" alt="观看演示 29" width="410" />
  <img src="./assets/demo_30.gif" alt="观看演示 30" width="410" />
</p>

## OpenHoopVision 是什么？

OpenHoopVision 逐帧分析篮球比赛视频，检测球员、篮球和场地特征，持续跟踪球员，为球员分配球队标签和全局 ID，并推断篮球动作与控球事件。处理过程会输出带标注的视频和逐帧事件记录。

## 功能

- 检测球员、篮球、篮筐、投篮球员、持球人、裁判及球在蓝网中。
- 将球员检测结果与姿态关键点和局部轨迹关联起来。
- 估计球队颜色，并通过体育 ReID 和球衣号码 OCR 跨帧匹配球员身份。
- 将检测到的场地关键点映射到球场布局，并绘制迷你地图。
- 在视频证据充分时推断投篮、运球、传球、篮板、失误和抢断。
- 将球员标签、事件状态、记分牌和球场叠加信息绘制到 H.264 视频中。

## 工作原理

视频处理流水线逐帧分析画面，将检测和姿态估计结果与局部跟踪结合，再更新球员身份和事件状态。渲染器和 JSONL 写入器使用同一帧上下文，确保视觉输出和记录数据保持一致。

| 阶段 | 职责 |
| --- | --- |
| 检测与姿态估计 | 检测篮球相关目标，并估计球员和场地关键点。 |
| 局部跟踪 | 在连续视频帧中稳定跟踪球员和篮球。 |
| 身份与球队 | 跨帧匹配球员图像、识别球队，并收集球衣号码读数。 |
| 事件推断 | 跟踪篮球动作和控球变化，处理事件重叠，判断投篮归属并更新记分牌。 |
| 输出 | 绘制分析结果，并生成带标注的视频和逐帧 JSONL 记录。 |

### 投篮事件（Shot Event）

OpenHoopVision 使用专门的检测器识别篮球动作和控球变化，再在统一时间线上处理重叠事件。已确认的投篮事件优先于与其重叠的传球。传球需要有可见的球传递证据和稳定的接球控制；如果发送者球队已知，则必须与接球者球队匹配。

投篮检测器结合篮球运动轨迹和篮筐信息，将投篮与运球区分开来，并将结果标记为 `made`、`miss` 或 `unknown`。当场地位置可靠时，系统根据位置计算两分或三分值；否则会给出带标签的两分回退结果。

## 性能与模型指标

| 指标 | 结果 |
| --- | --- |
| RF-DETR 检测器（`Basketball-Detection-v6`） | mAP@0.5：84.9%；mAP@0.5:0.95：62.3% |
| MobileNetV4-Conv-Small 场地关键点模型（`court-detection-v2`，896×512） | PCK@10：88.17%；PCK@20：94.26% |
| 端到端投篮流水线（`basketball-event-recognition`） | 精确率：85.51%；召回率：83.10%；F1：84.29%；球队归属准确率：84.75%（50/59） |
| Shot made（融合 `ball-in-basket`） | 精确率：89.13%；召回率：82.00%；F1：85.42% |

所有数据集均由项目团队整理和标注，模型指标使用验证集计算。当前端到端回归指标覆盖 7 段视频中的 71 次投篮，其中 50 次为 made，全部按时间 IoU 0.5 精确匹配，并设置 `ball-in-basket` 阈值 0.55。球队归属准确率使用 59 次匹配投篮作为分母。

## 运行 OpenHoopVision

### 环境要求

- Python 3.10 或更高版本。
- H.264 视频输出需要 FFmpeg。请从 [FFmpeg 官方下载页面](https://ffmpeg.org/download.html) 安装适用于当前平台的版本，并确保 `ffmpeg` 可从 `PATH` 中访问。
- 需要下方列出的模型权重文件。程序会读取对应路径中的文件，不会自动下载。
- 建议使用 NVIDIA GPU 以获得实用的处理速度。CUDA 不可用时，程序会使用 CPU 推理。

在仓库根目录安装项目及其依赖：

```bash
python -m pip install -e .
```

`requirements.lock` 记录了已验证的 CUDA 13.0 环境所使用的直接依赖版本。如需安装这些版本：

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

### 模型权重

默认权重路径相对于项目目录：

| 组件 | 默认路径 |
| --- | --- |
| 篮球检测器 | `weights/rf-detr-basketball.pth` |
| 球员姿态模型 | `weights/rtmpose-s_simcc-aic-coco.pth` |
| 场地姿态模型 | `weights/court_mobilenetv4_896x512.pt` |
| 球员 ReID | [`weights/osnet_x1_0_msmt17.pt`](https://github.com/KaiyangZhou/deep-person-reid) — Torchreid MSMT17 OSNet x1.0 |
| 球衣号码 OCR | 微调后的 SVTRv2-B，权重位于 `weights/basketball_jersey_number.ckpt` |

### 分析视频

在项目目录中运行，并通过 `--video` 指定输入视频：

```bash
python main.py --video {your_video}.mp4
```

运行 `python main.py --help` 可查看所有检测阈值、跟踪设置、模型路径和输出选项。

程序会在画面背景匹配 `configs/courts/` 中的已保存机位时自动应用对应边界。未匹配的机位继续使用球场模型和原有回退逻辑。固定机位也可通过 `--court-boundary path/to/boundary.json` 手动指定场内区域。JSON 文件包含按边界顺序排列的至少三个 `[x, y]` 顶点，组成凸多边形。坐标以画面左上角为原点，按画面宽度和高度归一化到 0 至 1。球员脚部接地点位于区域内或边界上时，可进入身份和控球跟踪。复用边界时请保持相同取景。球场坐标与迷你地图仍使用球场关键点模型。

推理结束后，程序会使用重复的球衣号码读数、当前球衣颜色和局部轨迹连续性自动修正有充分证据的身份区间，并填补短暂的 ID 缺口。同一条局部轨迹切换到队友时，短裤和球鞋外观的持续变化有助于重新关联到此前通过号码确认的球员。外观相近的同队球员仍可能难以区分。使用 `--identity-review path/to/review.json` 可覆盖自动修正，应用已人工核对的球员身份区间。请使用与核对时相同的视频和帧编号。每项修正对一个或多个 `[local_track_id, first_frame, last_frame]` 区间（含首尾帧）设置全局 ID、球队和球衣号码。系统仅修正已有的场内球员记录，并检查同一帧内的全局 ID 是否重复。参见[示例](configs/identity-reviews/baseline-sample.json)。

### 输出文件

- `--output` 用于指定带标注的 H.264 MP4 视频。默认路径为从项目目录运行时的 `../output.mp4`。使用 `--output none` 可跳过所有视频渲染和写入，保留分析和 JSONL 输出。
- `--tracks` 用于指定 JSONL 文件，每���写入一个 JSON 对象。默认路径为 `../tracks.jsonl`。
- `--events` 将已完成的投篮、运球、传球、篮板和失误按时间顺序写入 `v2` JSON 事件列表。默认路径为 `../events.json`。抢断记录为失误事件，其 `attrs.cause` 为 `steal`。

每条 JSONL 帧记录都使用 `basketball_frame_v2` 结构，包含帧编号、时间戳、球员、持球人、篮球和篮筐特征、原始检测结果及事件。第一条记录还包含本次运行配置。事件记录使用帧区间，并包含投篮结果与得分、传球接收人或篮板类型等字段。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `main.py` | 读取命令行参数并启动视频分析。 |
| `pipeline/` | 协调逐帧处理，并在各阶段间传递结果。 |
| `models/` | 加载视觉模型，并将模型输出转换为流水线可用的数据。 |
| `tracking/` | 跨帧跟踪球员和篮球，并估计球队和场地位置。 |
| `reid/` | 匹配不同轨迹中的球员身份，并将球衣号码读数作为辅助依据。 |
| `events/` | 检测篮球动作和控球变化，判断事件归属并计算得分。 |
| `render/` | 在画面上绘制分析结果并输出带标注的视频。 |
| `assets/` | 存放项目标志和本 README 展示的演示动图。 |

## 当前范围

目前系统可以推断投篮、运球、传球、篮板、失误和抢断；暂不支持助攻、盖帽和犯规。如果场地位置不可靠，系统会使用并标记为两分的默认得分。

已知限制：遮挡、镜头切换和低分辨率画面可能导致篮球跟踪、球员身份识别和事件结果不可靠。低机位若无法看清场地边线，迷你地图可能无法正常工作。该流水线离线处理录制视频、在低性能 GPU 下可能较慢，并要求单独安装模型权重文件。

下一步，我正在探索基于本项目开发实用工具，例如自动生成比赛集锦，敬请期待！

## 参考项目

[RF-DETR](https://github.com/roboflow/rf-detr) — 用于加载篮球检测器权重的目标检测框架。

[timm](https://github.com/huggingface/pytorch-image-models) — [`models/court.py`](models/court.py) 中本地实现的 MobileNetV4-Conv-Small 配置参考来源。

[McByte++](https://github.com/tstanczyk95/McBytePlusPlus) — 本地跟踪实现的参考来源。

[ByteTrack](https://github.com/FoundationVision/ByteTrack) — 本地跟踪器所采用的高低置信度关联方法和卡尔曼滤波基础。

[OSNet / Torchreid](https://github.com/KaiyangZhou/deep-person-reid) — 在 [`models/appearance.py`](models/appearance.py) 中改编的 OSNet x1.0 架构。

[OpenOCR](https://github.com/Topdu/OpenOCR) — [`models/jersey_number.py`](models/jersey_number.py) 中 SVTRv2-B OCR 架构的改编来源，用于识别球衣号码。

## 许可证

OpenHoopVision 的原创代码和项目训练的模型文件依据 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 发布。第三方模型、代码和依赖仍遵循各自的许可证约定。演示 GIF 是从网络视频中提取的预览素材，原始视频仍受其版权和使用条款约束。
