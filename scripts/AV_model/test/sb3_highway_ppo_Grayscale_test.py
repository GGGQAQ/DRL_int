
from configurator.configuration import *

# 测试轮次
Round = 5

# Create environment
env = gym.make(env_name, render_mode=render_mode)
env.configure(env_config)
env.config['duration'] = 80
env.reset()

pprint.pprint(env.config)
# load model
model = PPO.load(model_file, env=env)



for _ in range(Round):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
