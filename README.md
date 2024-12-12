# [PYTORCH] Super Mario Bros游戏

使用Pytorch实现DRL训练智能体通关Mario游戏。

## 环境

使用的[ gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) 来模拟Mario游戏的环境。


## v1

该版本是简易实现版本，使用的 `stable_baselines3` 库来完成，里面封装好了相关的算法。

训练时调用算法填入对应参数即可。

```python
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log, **train_params)
```

测试时调用 `predict()`方法获取动作，然后调用 `step()`方法获取采取动作后的状态。然后调用 `env.render()`即可在UI中展示效果。


## v2

这个版本是看了一些教程之后，自己编写网络实现的。

使用的算法为 `Dueling_DQN` 

代码结构部分的话参照了 [Super-mario-bros-PPO-pytorch](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch) 。以及 [动手学强化学习](https://hrl.boyuai.com/chapter/1/%E5%88%9D%E6%8E%A2%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/) 这个DRL的教程的 `Dueling_DQN`部分。

> 基本就是结合二者魔改了一番

自己写完差不多跑了几千个epoch，然后拿第一关测试了一下，通关是没什么问题。但是时间有限，没进一步的完善一些细节。后续闲下来了会回来完善的(应该？)
