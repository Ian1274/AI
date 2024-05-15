# 导入相关的库
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# 创建环境
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# 导入动作集：COMPLEX_MOVEMENT，这个动作集是最复杂的，包含了所有的动作
# 将动作集应用到环境中
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# env = JoypadSpace(env, [['right'], ['right', 'A']]) # 只使用右移和跳跃两个动作

# 重置环境
state = env.reset()
for step in range(5000):
    # 渲染游戏画面
    env.render()
    # 获取游戏的状态，奖励，是否结束，额外信息
    state, reward, done, info = env.step(env.action_space.sample())
    # 如果游戏结束，那么重置游戏
    if done:
        state = env.reset()
# 关闭环境
env.close()
