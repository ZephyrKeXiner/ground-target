# ground-target systemd部署

## 最小化IMX219开机录像服务

仅采集训练视频时，使用独立服务，不要求MAVLink、YOLO或相机标定：

```bash
cd ~/ai
sudo ./deploy/systemd/install_video.sh
```

默认参数：

```text
传感器3280×2464，逆时针旋转90°后保存为2464×3280
21 FPS，H.264 Matroska，码率50 Mbps
自动曝光13～2000 µs（最长约1/500秒）
输出目录：/home/argus/ai/training_videos
磁盘保护：剩余空间低于5 GiB自动结束当前录像
卡死保护：文件30秒不增长则失败，由systemd在5秒后重启
```

最高模式实测约占用356 MB/分钟（约21 GB/小时）。这台Jetson没有H.264硬件
编码器，因此使用`x264enc ultrafast`软件编码。服务会保留最后5 GiB空间，避免
录像写满系统盘。

调整体积和画质时修改`/etc/ground-target/video.env`中的码率后重启服务：

```text
GROUND_TARGET_VIDEO_BITRATE_KBPS=50000
GROUND_TARGET_VIDEO_ROTATION=counterclockwise90
GROUND_TARGET_VIDEO_EXPOSURE_MIN_US=13
GROUND_TARGET_VIDEO_EXPOSURE_MAX_US=2000
```

训练素材建议保持50000；低于40000时，8MP运动画面中的小目标纹理可能明显
损失。50 Mbps仍比原MJPEG实测体积小约6倍。

每次服务启动创建新文件，不会覆盖旧视频：

```text
imx219_20260720T120000Z_3280x2464_21fps.mkv
imx219_20260720T120000Z_3280x2464_21fps.json
```

管理命令：

```bash
systemctl status ground-target-video.service
journalctl -fu ground-target-video.service
sudo systemctl restart ground-target-video.service
sudo systemctl stop ground-target-video.service
```

录像服务独占IMX219，并与`ground-target-yolo.service`声明为冲突单元。需要恢复
YOLO时，先停止并禁用独立录像：

```bash
sudo systemctl disable --now ground-target-video.service
```

服务拆成三个单元：

- `ground-target-prepare.service`：验证配置并为本次启动创建独立记录目录；
- `ground-target-controller.service`：接收MAVLink并计算目标GPS；
- `ground-target-yolo.service`：读取IMX219、运行YOLO并发送bbox；
- `ground-target.target`：统一启动、停止和设置开机自启。

## 安装

安装需要root权限，因为systemd单元位于`/etc/systemd/system`：

```bash
cd ~/ai
sudo ./deploy/systemd/install.sh
```

安装器会保留已经存在的`/etc/ground-target/ground-target.env`，不会覆盖现场参数。
如果`target_geolocation/config.json`缺失、`calibrated=false`、分辨率不匹配或模型
不存在，单元仍会注册和启用，但不会启动算法。

完成标定配置后：

```bash
sudo systemctl restart ground-target.target
```

## 常用命令

```bash
# 整体状态
systemctl status ground-target.target \
  ground-target-controller.service \
  ground-target-yolo.service

# 实时日志
journalctl -fu ground-target-controller.service
journalctl -fu ground-target-yolo.service

# 本次启动记录
readlink -f ~/ai/runs/latest
ls -lh ~/ai/runs/latest

# 整体停止/启动/重启
sudo systemctl stop ground-target.target
sudo systemctl start ground-target.target
sudo systemctl restart ground-target.target

# 取消/恢复开机自启
sudo systemctl disable ground-target.target
sudo systemctl enable ground-target.target
```

修改`/etc/ground-target/ground-target.env`后必须重启target：

```bash
sudoedit /etc/ground-target/ground-target.env
sudo systemctl restart ground-target.target
```

## 稳定性设计

- `Type=notify`：只有MAVLink/UDP或相机/YOLO真正就绪后，systemd才认为启动成功；
- `WatchdogSec=20s`：主循环卡住时由systemd终止并重启；
- `Restart=on-failure`：相机断开、网络异常或进程崩溃后3秒重启；
- 启动限流：一分钟最多连续失败10次，防止故障时无限快速重启；
- 每次整体启动使用独立UTC目录，`runs/latest`指向最近一次；
- 图像记录限频，剩余空间低于5 GiB时停止保存图片，但bbox JSON继续；
- 默认以最高10 FPS录制未画框的MJPEG AVI训练视频，每段60秒；
- 单次服务启动的视频达到20 GiB或磁盘只剩5 GiB时停止录像，算法继续运行；
- SIGINT和15秒停止超时让日志、相机和socket有机会正常关闭；
- 配置错误使用退出码78，禁止无意义的自动重启循环；
- 服务以`argus`而不是root运行，并启用基础systemd沙箱保护。

systemd无法修复物理断线、错误标定或飞控没有发送MAVLink。服务显示失败时优先看：

```bash
journalctl -u ground-target-prepare.service -b --no-pager
journalctl -u ground-target-controller.service -b --no-pager -n 100
journalctl -u ground-target-yolo.service -b --no-pager -n 100
```

记录数据不会被自动删除。长期连续运行时应定期归档`~/ai/runs`，或把
`GROUND_TARGET_RECORD_IMAGES`改成`none`。

## 训练视频

默认环境文件启用了：

```text
GROUND_TARGET_RECORD_VIDEO=1
GROUND_TARGET_VIDEO_FPS=10
GROUND_TARGET_VIDEO_SEGMENT_SECONDS=60
GROUND_TARGET_VIDEO_QUALITY=85
GROUND_TARGET_VIDEO_MAX_TOTAL_GB=20
```

文件位于：

```text
~/ai/runs/latest/training_video/
├── video_session.json
├── video_frames.ndjson
├── segment_00000.avi
├── segment_00001.avi
└── ...
```

AVI内容是未画bbox的原始图像，适合后续抽帧标注。每段独立关闭，异常或断电时
之前的分段仍然可用。`video_frames.ndjson`记录视频帧与相机源帧、单调时钟的
对应关系。

关闭训练录像：

```bash
sudoedit /etc/ground-target/ground-target.env
# 设置 GROUND_TARGET_RECORD_VIDEO=0
sudo systemctl restart ground-target.target
```

使用ffmpeg抽取每秒2张图片：

```bash
mkdir -p ~/ai/dataset/images
ffmpeg -i ~/ai/runs/latest/training_video/segment_00000.avi \
  -vf fps=2 ~/ai/dataset/images/frame_%06d.jpg
```
