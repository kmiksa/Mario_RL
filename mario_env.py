# Part taken from adborghi fantastic implementation
# https://github.com/aborghi/retro_contest_agent/blob/master/fastlearner/ppo2ttifrutti_sonic_env.py
import numpy as np
import gym
import gym_super_mario_bros
from baselines.common.atari_wrappers import FrameStack


from baselines.common.distributions import make_pdtype


# import gym_remote.client as grc


# This will be useful for stacking frames
# from baselines.common.atari_wrappers import FrameStack

# Library used to modify frames (former times we used matplotlib)
import cv2

# setUseOpenCL = False means that we will not use GPU (disable OpenCL acceleration)
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plot


class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
        - Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        # Set frame to gray
        #print(type(frame))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 96x96x1
        frame = frame[35: , :,None]

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]
        # if flag.DEBUG:
        #     cv2.imshow("frame",frame)
        #     cv2.waitKey(0)
        return frame

class ActionsDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = ["right", "A", "B", "NOOP",'left','down','up']
        actions = [['NOOP'],['right'],['right', 'A'],['right', 'B'],['right', 'A', 'B'],['A'],['left'],
                ['left', 'A'],['left', 'B'],['left', 'A', 'B'],['down'],['up']
            ]
        #right B=run faster
        #right A B=jump faster
        self.actions = []
        """
        What we do in this loop:
        For each action in actions
            - Create an array of 12 False (12 = nb of buttons)
            For each button in action: (for instance ['LEFT']) we need to make that left button index = True
                - Then the button index = LEFT = True
            In fact at the end we will have an array where each array is an action and each elements True of this array
            are the buttons clicked.
        """
        for action in actions:
            arr = np.array([False] * 7)
            for button in action:
                arr[buttons.index(button)] = True
            self.actions.append(arr)

        print("action is ", self.actions)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def step(self, a): # pylint: disable=W0221
        print('action:', self.actions[a])
        return self.actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):


        return reward


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


def make_env(env_idx):
    """
    Create an environment with some standard wrappers.
    """


    # Make the environment


    levelList = ['SuperMarioBros-1-1-v0','SuperMarioBros-1-2-v0','SuperMarioBros-1-3-v0','SuperMarioBros-1-4-v0','SuperMarioBros-2-1-v0','SuperMarioBros-2-2-v0','SuperMarioBros-2-3-v0','SuperMarioBros-2-4-v0']


    # record_path = "./records/" + dicts[env_idx]['state']
    env = gym_super_mario_bros.make(levelList[env_idx])
    

    env = ActionsDiscretizer(env)

    env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)


    # Stack 4 frames
    env = FrameStack(env,6)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)

    return env


def make_train_0():
    return make_env(0)

def make_train_1():
    return make_env(1)

def make_train_2():
    return make_env(2)

def make_train_3():
    return make_env(3)

def make_train_4():
    return make_env(4)

def make_train_5():
    return make_env(5)

def make_train_6():
    return make_env(6)

def make_train_7():
    return make_env(7)

def make_test_level_Green():
    return make_test()


def make_test():
    """
    Create an environment with some standard wrappers.
    """

    # Make the environment
    env = gym_super_mario_bros.make('SuperMarioBros-3-1-v0')
    
    env = ActionsDiscretizer(env)

    # Scale the rewards
    env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)

    # Stack 4 frames
    env = FrameStack(env, 4) # This can be changed. 

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)

    return env




