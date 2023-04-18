# 基于对抗强化学习的FlappyBird游戏AI
## 项目介绍
本项目为HIT大一年度项目内容，
基于DQN算法设计基础的FlappyBird游戏AI，
并使用对抗训练算法进行改进。
## 安装
### 本体
``` git clone https://github.com/Dimweaker/FlappyDRL.git .```
### 依赖
```pip install -r requirements.txt```
## 使用说明
### 训练
```python train.py [-h] [--iter ITER] [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE] [--optimizer OPTIMIZER] [--lr LR] [--gamma GAMMA] [--epsilon EPSILON] [--initial_epsilon INITIAL_EPSILON] [--final_epsilon FINAL_EPSILON] [--num_iters NUM_ITERS] [--memory_size MEMORY_SIZE] [--log_path LOG_PATH] [--saved_path SAVED_PATH]```
### 测试
```python test.py [-h] [--model_name MODEL_NAME] [--image_size IMAGE_SIZE] [--log_path LOG_PATH]```
## 效果演示
[![Watch the video](demo.png)](https://www.bilibili.com/video/BV1Vv4y1V7j8)
## 项目报告
本项目是大一年度项目，报告内容暂不公开，如有需要请联系作者。
## 参考
- [FlappyBird](https://github.com/sourabhv/FlapPyBird) Flappy Bird的游戏本体文件
## 作者
#### [@Dimweaker](https://github.com/Dimweaker)
- [email:huiyyh@qq.com](<huiyyh@qq.com>)
