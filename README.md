# ground-target

这是 Jetson + ArduPilot 地面目标定位项目。当前完成的模块会把目标检测输出的
bbox 与飞控在图像曝光时刻的位置、姿态进行融合，把图像像素投影成地面 GPS
坐标。

实时入口是根目录的 `stream.py`：OpenCV读取CSI相机，YOLO + ByteTrack生成bbox，
然后向定位控制器发送逐帧JSON。

当前版本只做目标定位和日志记录，不会解锁飞控、切换飞行模式或控制舵机。

## 建议阅读顺序

1. [定位模块中文教程](target_geolocation/README.md)：先理解整体数据流和公式。
2. [投影核心](target_geolocation/core.py)：学习像素、相机、机体、NED 和 GPS 的转换。
3. [核心测试](target_geolocation/test_core.py)：用简单例子理解方向是否正确。
4. [控制器](target_geolocation/controller.py)：学习 MAVLink 缓存、时间对齐、UDP JSON 和结果输出。
5. [控制器测试](target_geolocation/test_controller.py)：理解一帧 bbox 如何与飞控状态融合。
6. [JSON测试发送器](target_geolocation/send_test_detection.py)：不运行YOLO时也能测试数据链路。
7. [离线重放](target_geolocation/replay.py)：落地后用新参数重算同一次飞行。
8. [误差分析](target_geolocation/analyze_results.py)：使用已知目标真值判断偏差来源。

## 当前目录

```text
stream.py                         # CSI取帧、YOLO跟踪、发送bbox JSON
target_geolocation/
├── core.py                   # 坐标变换和地面求交
├── controller.py             # MAVLink + bbox JSON 融合控制器
├── replay.py                 # 重放飞行遥测与bbox
├── analyze_results.py        # 统计精度并给出偏差线索
├── config.example.json       # 相机和高度配置模板
├── send_test_detection.py    # 发送模拟bbox
├── test_core.py              # 投影方向测试
├── test_controller.py        # JSON与遥测融合测试
└── README.md                 # 中文教程
```

## 快速测试

```bash
cd ~/ai
.venv/bin/python -m unittest discover \
  -s target_geolocation -p 'test_*.py' -v
```

运行YOLO bbox发送器：

```bash
cd ~/ai
.venv/bin/python stream.py --bbox-host 127.0.0.1 --bbox-port 15100
```

Jetson应优先使用JetPack/Ubuntu自带的OpenCV，因为它包含CSI相机所需的
GStreamer支持。不要在这个虚拟环境中安装要求NumPy 2的`opencv-python 5`，
否则会与Jetson的Torch和Matplotlib二进制包冲突。

真实运行前必须完成相机标定，并根据
[定位模块中文教程](target_geolocation/README.md) 填写配置。

## 推荐的飞行调试方式

飞行时让YOLO记录全部bbox，并以最高2 FPS保存含目标的原图：

```bash
.venv/bin/python stream.py --no-display \
  --record-dir runs/test01/camera \
  --record-images detections \
  --record-image-fps 2
```

定位控制器同时记录可重放的MAVLink和bbox事件：

```bash
.venv/bin/python -m target_geolocation.controller \
  --config target_geolocation/config.json \
  --mavlink udpin:0.0.0.0:14550 \
  --listen 127.0.0.1:15100 \
  --output runs/test01/results.ndjson \
  --events runs/test01/events.ndjson
```

落地后可以修改内参、安装角或高度配置，再重放而不需要重新飞：

```bash
.venv/bin/python -m target_geolocation.replay \
  --events runs/test01/events.ndjson \
  --config target_geolocation/config.json \
  --output runs/test01/replayed.ndjson
```

如果地面标志的GPS真值已知：

```bash
.venv/bin/python -m target_geolocation.analyze_results \
  runs/test01/results.ndjson --truth 1.2345678 103.1234567
```

## systemd开机自启

完整服务文件、环境参数和安装方法见
[deploy/systemd/README.md](deploy/systemd/README.md)。服务包含就绪通知、20秒
看门狗、失败重启、启动限流、独立飞行记录目录和低磁盘保护。

## 录制训练视频

YOLO运行时可以同时保存未画框的原始训练视频：

```bash
.venv/bin/python stream.py --no-display \
  --training-video-dir runs/training01/video \
  --video-fps 10 \
  --video-segment-seconds 60 \
  --video-max-total-gb 20
```

默认使用Jetson的`nvjpegenc`写入分段MJPEG AVI；视频之外还会生成
`video_frames.ndjson`，用于对应源帧编号和采集时间。systemd环境默认开启该功能，
保存到`runs/latest/training_video`。
