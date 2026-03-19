from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import torch
import os, sys

from myQnet import QNetwork



main_dir = os.getcwd()
video_dir = os.path.join(main_dir, "videos/lunar_lander")
model_dir = os.path.join(main_dir, "models/lunar_lander") 
  
device = torch.device("cpu")

wind = bool(sys.argv[1]) 
hidden_dim = int(sys.argv[2])
num_episodes = int(sys.argv[3])

render_env = gym.make(
    "LunarLander-v3",
    continuous=False,
    gravity=-10,
    enable_wind=wind,
    wind_power=15,
    turbulence_power=1.5,
    render_mode="rgb_array"
)

video_env = RecordVideo(
    render_env, 
    video_dir, 
    episode_trigger=lambda ep: True,
    name_prefix='lander')

obs_space = video_env.observation_space
action_space = video_env.action_space

state_dim = obs_space.shape[0]
action_dim = action_space.n
 
qnet = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)

qnet.load_state_dict(torch.load(os.path.join(model_dir, f"best_lunar_lander_{hidden_dim}.pth")))
qnet.eval()

for ep in range(num_episodes):
    state, _ = video_env.reset()

    done = False

    total_reward = 0

    while not done:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = qnet(state).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward 

    print(f"Episode {ep}, reward: {total_reward}")

video_env.close()