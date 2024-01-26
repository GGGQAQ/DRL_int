import argparse
import highway_env
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--env','-e',type=str, default = "highway-fast-modify-v0",help="env:highway-fast-modify-v0/highway-fast-v0")
parser.add_argument('--alg','-a',type=str, default="PPO",help='algorithm:DQN/PPO')
parser.add_argument('--obs','-obs_space',type=str, default='GrayscaleObservation', help='obs_space:GrayscaleObservation/Kinematics')
parser.add_argument('--act','-act',type=str, default='ContinuousAction', help="ContinuousAction/DiscreteMetaAction")
parser.add_argument('--net','-net_arch', type=str, required=True, help='netarch')
parser.add_argument('--policy_freq', '-pf', type=int, default=2, help="policy_freq")
parser.add_argument('--simulate-freq', '-sf', type=int, default=5, help="simulate-freq")
parser.add_argument('--reward_speed_range', '-rsr', type=tuple, default=[20,30], help="reward_speed_range")
parser.add_argument('--high_speed_reward', '-hsr', type=float, default=0.8, help="high_speed_reward")
parser.add_argument('--number_of_expirements', '-noe', type=int, required=True, help='number_of_expirements')
parser.add_argument('--total_timesteps', '-tt', type=int, required=True, help='total timesteps')
args = parser.parse_args()
Root_dir = 'D:/0Projects\DRL_based_AV_interpretability'
model_name = f"{args.env}_{args.alg}_{args.obs}_{args.act}_{args.number_of_expirements}"
model_path = f'{Root_dir}/model/AV_model/{model_name}'
model_file = f'{model_path}/model'

net_arch = net_arch_tuple = tuple(map(int, args.net.split(',')))
print(type(net_arch))
print(net_arch)

if(args.obs == 'GrayscaleObservation'):
    observation = {
        'type': "GrayscaleObservation",
        'observation_shape': (128, 64),
        'stack_size':4,
        'weights': [0.2989, 0.5870, 0.1140],
        'scaling': 1.75,
    }
    policy = 'CnnPolicy'
elif(args.obs == 'Kinematics'):
    observation = {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],
    }
    policy = 'MlpPolicy'
else:
    print('obs false!!')
    exit()

env = gym.make(args.env)
env.configure({
    "collision_reward": -2,
    "high_speed_reward": args.high_speed_reward, # 0.8
    "reward_speed_range": args.reward_speed_range, # [20,30]
    'policy_frequency': args.policy_freq, # 2
    "simulation_frequency": args.simulate_freq, # 5
    "observation": observation,
    'action': {
            'type': args.act
    },
    'offroad_terminal': True, # 车辆出了车道则终止episode，之前ppo训练不好就是因为这里
    'negative_speed_terminal': True,  # 车辆速度为负则终止episode
    'lane_change_reward': -0.2,
})
env.reset()

eval_env = gym.make(args.env)
eval_env.configure({
    "collision_reward": -2,
    "high_speed_reward": args.high_speed_reward, # 0.8
    "reward_speed_range": args.reward_speed_range, # [20,30]
    'policy_frequency': args.policy_freq, # 2
    "simulation_frequency": args.simulate_freq, # 5
    "observation": observation,
    'action': {
            'type': args.act
    },
    'offroad_terminal': True, # 车辆出了车道则终止episode，之前ppo训练不好就是因为这里
    'negative_speed_terminal': True,  # 车辆速度为负则终止episode
    'lane_change_reward': -0.2,
})
eval_env.reset()

if(args.alg == 'PPO'):
    policy_kwargs = dict(net_arch=[dict(pi=net_arch, vf=net_arch)])
    model = PPO(policy, env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=model_path)

elif(args.alg == 'DQN'):
    policy_kwargs = dict(net_arch=net_arch)
    model = DQN(policy, env,
                policy_kwargs=policy_kwargs,
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log=model_path)
else:
    print('alg false!')
    exit()

checkpoint_callback = CheckpointCallback(save_freq=args.total_timesteps/10, save_path=f'{model_path}/')
eval_callback = EvalCallback(eval_env, best_model_save_path=f'{model_path}',
                             log_path=f'{model_path}', eval_freq=args.total_timesteps/10,
                             deterministic=True, render=False)
callback = CallbackList([checkpoint_callback, eval_callback])

model.learn(total_timesteps=args.total_timesteps, callback=callback)
model.save(model_file)

