# VFCExtractor-CLI
> :bulb:  视频人脸批量提取 + 相似度聚类，CLI 操作<br>
> :bulb:  Batch extract faces from videos, cluster them by similarity, via CLI

## 一. 项目介绍 / Project Introduction
### 1. 核心功能 / Core Functions
- 批量从本地视频文件中检测、提取人脸图像，并基于相似度自动聚类
Batch detect and extract facial images from local video files in bulk, and automatically cluster based on similarity 
- 命令行（或IDE）使用，支持自定义参数 
Via CLI or IDE, supporting custom parameters

### 2. 应用场景 / Application Scenarios
- 个人/小团队视频处理自动化、轻量化人脸分组需求
Video processing automation and lightweight face grouping needs for individuals/small teams
- AI 视觉任务上下游数据预处理（如人脸数据集构建、标注辅助）
Data preprocessing for AI related tasks (e.g., face dataset construction, annotation assistance)

### 3. 效果演示 / Effect Demo
以《一生所爱》mv作为演示视频（MP4格式，分辨率1104x622，时长4分52秒），输出结果如下：
- 以视频名为目录存放导出的帧图片及face_set的聚类文件夹：
![demo_outputs](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo_outputs.PNG)
- 聚类文件夹1：紫霞（朱茵），18张
![demo_cluster1](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo_cluster1.PNG)
- 聚类文件夹2：至尊宝（猴哥），8张
![demo_cluster2](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo_cluster2.PNG)
- 聚类文件夹3：至尊宝（星爷），5张
![demo_cluster3](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo_cluster3.PNG)
- 文件夹4（Others）： 群演与配角，53张
![demo_cluster_others](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo_cluster_others.PNG)

> :bulb: 说明<br>
> - 通过调整不同的自定义参数，可达成不同聚类效果，其中Others为不满足最小聚类数量的统一汇总<br>
> - 本演示效果使用的关键参数为：人脸检测阈值0.8，相似度阈值0.393，最小聚类数4


## 二. 如何使用 / How to Use
### 1. 前提条件 / Prerequisites
- 需要安装python执行环境，并安装[requirements]中的依赖库
- 本项目未直接上传ffmpeg相关程序，需要使用者自行准备ffmpeg.exe文件（必须）和ffprobe.exe文件（如果视频导出参数不用A，则可以不需要），并放置在libs目录
- 本项目暂未直接上传AI模型的onnx文件，需要使用者自行准备det_10g.onnx和w600k_r50.onnx文件，并放置在models目录
- 本项目目前仅在自用的RTX30系列显卡下充分执行，未测试及验证在其他系列显卡或纯CPU下的执行兼容性（理论上OK）
### 2. 执行示例 / Execution Examples
1) 首先将要处理的本地视频文件，放入项目的workspace目录下（默认工作目录，可在配置文件中修改）
> :warning: 注意：当前仅支持后缀为mp4/avi/mov/mkv格式的文件，且当前不会遍历工作目录下的子目录中的任何文件
2) 接下来选择如下两种方式中任意一种执行：
- IDE执行，参考：
![IDE_execute](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/IDE_execute.png)
- 命令行执行(powershell or cmd)，参考：
![CLI_execute](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/CLI_execute.png)

3) 查看输出结果，如聚类效果不满意，则修改配置文件中的参数后重复执行
> :bulb: 输出结果说明：在工作目录下，按照视频名称生成对应文件夹目录，内含视频帧切片及face_set文件夹，face_set下按人脸聚类分不同目录归档了人脸图片，聚类文件夹名称中的括号内的数字为其中的人脸图片数量，方便快速检索

### 3. 参数说明 / Parameter Description
 本项目目录下的VFC_Configuration.json为参数配置文件，可通过修改其中的参数来调整最终的聚类及图片效果：
```
{
  "work_path": "./workspace",
  "exp_video_mode": "A",
  "exp_video_frame_limit": 30,
  "face_confidence_thresh": 0.8,
  "img_max_face": 10,
  "exp_face_scale_factor": 0.75,
  "exp_face_z_fix": -20,
  "exp_face_size": 512,
  "export_cluster_min_len": 20,
  "similarity_threshold": 0.5
}
```
- work_path: 工作路径，程序会自动扫描并处理该路径下的所有视频文件，默认./workspace
> 支持mp4/avi/mov/mkv格式<br>
> 当前不会遍历工作目录下的子目录中的任何文件
- exp_video_mode: 视频帧导出模式，仅支持以下有效值（非有效值会触发程序异常，暂未做异常保护）  
 > "I": 仅导出视频的I帧（关键帧，画面完整度/清晰度最高）<br>
 > 数字: 按帧率导出，如2=每秒导出2帧，支持分数，如1/2=每2秒导出1帧<br>
 > "A": 自动模式（默认），结合exp_video_frame_limit参数使用，逻辑如下<br>
 > 1) 先提取视频前1分钟的所有I帧<br>
 > 2) 若I帧数量 ≤ exp_video_frame_limit → 按"I"模式导出全部I帧<br>
 > 3) 若I帧数量 > exp_video_frame_limit → 按"I帧数量/2"导出（减半采样，避免I帧过多<br>
 > 注：自动模式是项目实践经验值，仅做1次减半处理，若仍需调整可手动切换为FPS数字模式  
- exp_video_frame_limit: 自动模式下I帧数量阈值，用于判断是否需要对I帧减半采样（默认30）
> 取值建议20~50
- face_confidence_thresh: AI人脸检测置信度阈值（默认0.8），范围 0\~1
> 过低（如<0.7）：易检出模糊/非人脸区域（无效人脸），增加后续计算量<br>
> 过高（如>0.9）：易漏检侧脸/遮挡人脸，降低检测召回率<br>
> 建议取值：0.75\~0.85（平衡准确率和召回率）
- img_max_face: 单张图片最大检出人脸数（默认10），超出该数量的人脸会被忽略
> 取值建议： 5\~20（过少易漏检，过多增加单帧处理耗时）<br>
> 业务场景：适合单人/小群体视频场景，若为多人场景可调整至 20\~30
- exp_face_scale_factor: 导出人脸区域的缩放系数（默认0.75）
> 非必要不推荐修改
- exp_face_z_fix: 导出人脸中心点的Z轴（垂直方向）偏移像素值（默认-20）
> 非必要不推荐修改  
- exp_face_size: 导出人脸图片的分辨率（默认512），格式为N（代表N×N像素）  
> 取值建议：128/256/512/1024  
- export_cluster_min_len: 人脸聚类最小数量阈值（默认5）  
> 作用：聚类后，数量小于该值的人脸簇会被归类到"Others"类别，不单独导出<br>
> 取值建议：3\~10（过滤少量零散的误检/低频次人脸，减少无效导出文件）  
- similarity_threshold: 人脸相似度阈值（默认0.4），范围0~1  
> 作用：判断两张人脸是否为同一人，用于聚类分组<br>
> 取值说明：<br>
> 过低: 不同人脸易被归为同一类，聚类准确率低<br>
> 过高: 同一人不同角度的人脸易被归为不同类，聚类召回率低<br>
> 建议取值: 无，根据视频中图像质量差别较大，聚类不满意请反复调整该值重试
 
### 4. 进阶选项 / Advanced Options
 适用于有经验的程序员或开发者，可参考实现介绍部分信息，自行修改代码实现灵活调用
 > :bulb:  提示<br>
 > - VFCWorks类中封装了工作流中的所有核心节点，Main中仅起到对VFCWorks中的功能做编排（顺序）调用<br>
 > - 进阶用户可修改并调用Main_debug，自行编排，或更灵活的调用VFCWorks中的核心功能节点<br>
 > - Main_debug中已经默认封装了首次执行与重复执行的两种策略，可通过注释不同代码行切换<br>
 > - Main_debug中的重复执行策略，核心是使用了VFCWorks中的导出/导入中间数据，来避免重复执行工作流的视频导出和模型执行相关步骤，直接做相似度聚类的步骤，方便针对单个视频，反复调测最佳的相似度阈值<br>
 > - Main程序的logging.level为WARNING，Main_debug程序执行时为INFO，可按需设置为DEBUG查看更多信息
 

## 三. 实现介绍 / Implementation Introduction
### 1. 原理流程
**核心流程**：将视频导出帧图片->逐张图片识别人脸->人脸embedding->相似度聚类->导出
![workflow](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/workflow.PNG)
### 2. 架构/代码结构/功能说明 / Architecture/Code Structure/Function Description
![architecture](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/architecture.PNG)
- VFCWorks类封装提供所有需执行的元子功能，管理数据结构（含任务列表、人脸坐标、embedding等）
- Main/Main_Debug按工作流组装的方式，调用VFCWorks完成自动任务
- VFCWorks依赖调用2大关键类：FFmpegExport、AIRuntime
- FFmpegExport（未封装类）：提供通过FFmpeg执行的相关功能，包括导出帧图片，计算帧等
- AIRuntime（封装Models类）：提供需AI模型执行的功能：人脸检测、人脸embedding
- 所有类均依赖Utils/ConfigLoader提供基本公共能力，包括文件/目录处理、全局变量、日志等

## 四. 调参效果与性能数据参考 / Parameter Tuning Effect & Performance Data Reference
为具体呈现不同参数下的效果及性能数据，以方便参考使用
准备了覆盖不同类型和属性的4个视频作为Demo视频，如下：
|Demo|视频格式|视频时长|视频分辨率|视频帧率|视频内容简述|
|-|-|-|-|-|-|
|Demo1|mp4|4m52s|低分辨率(1104x622)|25|一生所爱MV（大话西游）|
|Demo2|mp4|16m25s|1080P(1920x1080)|30|周星驰电影串剪|
|Demo3|mkv|2h49m3s|720P(1280x720)|12|电影-星际穿越|
|Demo4|mp4|2h3m6s|4K(3840x1608)|24|电影-战狼2|



###  各Demo达成基本聚类效果的关键参数及聚类效果信息：
|Demo|关键聚类参数|导出视频帧数量|识别人脸总数|最大可聚类数|最终聚类数|最终聚类简介|
|-|-|-|-|-|-|-|
|Demo1|相似度阈值：0.4; 最小聚类数：4|96|84|40|4|女主-17/猴哥-8/男主-5/Others-54|
|Demo2|相似度阈值：0.3; 最小聚类数：5|197|116|53|5|星爷-26/女配1-8/女配2-7/女配3-5/Others-70|
|Demo3|相似度阈值：0.4; 最小聚类数：20|3697|917|86|9|男主-319/女主-123/成年女儿-70/小孩女儿-61/NPC1-48/NPC2-31/NPC3-32/反派-49/Others-184|
|Demo4|相似度阈值：0.5; 最小聚类数：20|3336|2015|643|12|男主-460/女主-162/男配1-77/男配2-50/小孩1-58/小孩2-36/NPC1-41/NPC2-33/NPC3-29/反派1-52/反派2-33/Others-984|
---
- **Demo1** <br>
聚类结果： <br>
![demo1_clusters](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo1_clusters.png) <br>
聚类效果： <br>
![demo1_effect](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo1_effect.png) <br>
---
- **Demo2** <br>
聚类结果： <br>
![demo2_clusters](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo2_clusters.png) <br>
聚类效果： <br>
![demo2_effect](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo2_effect.png) <br>
---
- **Demo3** <br>
聚类结果： <br>
![demo3_clusters](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo3_clusters.png) <br>
聚类效果： <br>
![demo4_effect](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo3_effect.png) <br>
---
- **Demo4** <br>
聚类结果： <br>
![demo4_clusters](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo4_clusters.png) <br>
聚类效果： <br>
![demo4_effect](https://github.com/iciferdai/VFCExtractor-CLI/blob/main/images/demo4_effect.png) <br>
---
> :bulb: **效果调试参考：**<br>
> - 可以看到一般影视作品，主角提取数量远大于其他，所以设置合理的最小聚类数，可有效突出主体，当然通过聚类文件夹命名提示的人脸数量，可以快速找到<br>
> - 以Demo1举例，最佳的相似度参数是效果演示章节使用的0.393，与此处测试使用的0.4，实测差异为其中1张主角人脸，因色彩及角度原因，相似度判定精度低，实测采用0.3一直到0.39时，虽可以将这张图片正确聚类，但会导致同时将不少不同NPC判定为相似并聚类，而采用0.4时，则本张图片未被聚类，被丢到Others，于是通过再增加小数点后一位精度，反复尝试，找到最佳阈值0.393，但实际使用时，不追求完全召回，实际使用0.4，舍弃这张特殊人脸，也完全是可以的<br>
> - 以Demo4举例，视频分辨率/清晰度高（但非所有帧的人脸均清晰），相比其他视频，相似度阈值也是设置到了0.5，当设置0.4时，有大量NPC被错误聚类到主角目录，在上调至0.5后，观察到聚类的人物目录下已无错误聚类，而且Others内没有明显的主角人物，于是便可不再提高相似度阈值

###  各Demo执行的分阶段性能与整体耗时统计信息：
|Demo|FFmpeg|KPS|EMBED|SIMILAR|EXPORT|总耗时(SUM)|
|-|-|-|-|-|-|-|
|Demo1|6.197|0.933|0.672|0.018|1.219|~9s|
|Demo2|24.039|1.847|2.634|0.040|3.404|~32s|
|Demo3|112.429|22.722|6.294|2.006|14.848|~158s|
|Demo4|485.528|23.551|12.433|9.118|173.692|~704s|

> :bulb: 说明<br>
> - 单位均为秒，总耗时为各阶段统计耗时的累加，实际程序中各任务及功能调度间也会占用一些耗时，所以实际项目执行的真实耗时略大于上述统计的总耗时<br>
> - 多次重复执行同一任务，其耗时虽不完全相同，但波动范围较小，属于正常误差范围，上述统计仅采用其中一次耗时作为代表<br>
> - 运行上述Demo使用的硬件主要信息为：CPU：i9-9900K; GPU：RTX3090; RAM：16GB; 磁盘：SSD

# 后继 / Future Plans
如有空闲时间计划优化或升级的点：
- [x]  支持按指定图片中的人物导出视频中的人脸
- [x]  优化AIRuntime，支持更多模型，模型可配置
- [x]  提取更多参数到配置文件配置
- [ ] 前端是不可能做的，需要可交互界面的有很多正经选择

#  结束 / End
>:loudspeaker: Notice：<br>
> 本项目为个人学习与实验性项目<br>
> This is personal learning and experimental project
