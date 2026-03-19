import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import deque
import numpy as np
from torch import optim, nn
import torch 
import matplotlib.pyplot as plt
import os, sys, random, time
from myQnet import QNetwork

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")
print(f"device: {device}")

wind = bool(sys.argv[1])
print(f"wind: {wind}")
hidden_dim = int(sys.argv[2])
print(f"hidden_dim: {hidden_dim}")
num_episodes = int(sys.argv[3])
print(f"num_episodes: {num_episodes}")

NUM_EPISODES = num_episodes
BATCH_SIZE = 64
GAMMA = 0.98
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99

TARGET_UPDATE_FREQ = 50

BEST_EVAL = -float('inf')
EVAL_FREQ = 50
EVAL_EPISODES = 10
PATIENCE = 20
IMPROVEMENT_COUNT = 0
MIN_IMPROVEMENT = 5.0

main_dir = os.getcwd()
video_dir = os.path.join(main_dir, "videos/lunar_lander")
model_dir = os.path.join(main_dir, "models/lunar_lander") 
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

env = gym.make(
    "LunarLander-v3", 
    continuous=False, 
    gravity= -10, 
    enable_wind=wind, 
    wind_power=15, 
    turbulence_power=1.5)
state_dim = env.observation_space.shape[0]; print(f"state_dim: {state_dim}")
action_dim = env.action_space.n; print(f"action_dim: {action_dim}") 
qnet = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
target_net = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
target_net.load_state_dict(qnet.state_dict())

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self,batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones =  zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    def __len__(self):
        return len(self.buffer)

def select_action(state, qnet, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = qnet(state)
        return q_values.argmax(dim=1).item()

def train_step(qnet, target_net, buffer, optimizer, batch_size, beta, device):
    if len(buffer) < batch_size:
        return None
    state, action, reward, next_state, done = buffer.sample(batch_size)
    states = torch.tensor(state, dtype=torch.float32, device=device)
    actions = torch.tensor(action, dtype=torch.int32, device=device).unsqueeze(1)
    rewards = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(next_state, dtype=torch.float32, device=device)
    dones = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)
    q_values = qnet(states).gather(1, actions)
    with torch.no_grad(): # double q
        next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
        target = rewards + beta * ( 1 - dones ) * next_q
    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def record(qnet,wind,video_dir,device, eps=3, name_prefix=f"lunar_lander_"):
    env = gym.make(
        "LunarLander-v3", 
        continuous=False, 
        gravity= -10, 
        enable_wind=wind, 
        wind_power=15, 
        turbulence_power=1.5,
        render_mode="rgb_array"
    )
    env = RecordVideo(
        env,
        video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=name_prefix
    )
    qnet.eval()
    for ep in range(eps):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = qnet(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
        env.close()
        qnet.train()

def Evaluate(env, qnet,num_episodes, device):
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = qnet(state).argmax(dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

buffer = ReplayBuffer(capacity=100000)
state, _ = env.reset()
for iter in range(1000):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    buffer.push(state, action, reward, next_state, done)
    state = next_state
    if done: 
        state, _ = env.reset()


optimizer = optim.Adam(qnet.parameters(), lr=5e-4)
episode_rewards = []
episode_losses = []
episode_epsilons = []
for epi in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0
    losses = []

    while not done:
        action = select_action(state, qnet, EPSILON, action_dim=action_dim, device=device) # get action from qnet
        next_state, reward, terminated, truncated, info = env.step(action) # take action in env
        done = terminated or truncated # check if done
        buffer.push(state, action, reward, next_state, done) # store transition in buffer

        loss = train_step(qnet, target_net, buffer, optimizer, batch_size=BATCH_SIZE, beta=GAMMA, device=device) # train qnet

        if loss is not None:
            losses.append(loss)
            
        state = next_state
        total_reward += reward

    # if epi % TARGET_UPDATE_FREQ == 0:
    #     target_net.load_state_dict(qnet.state_dict())
    #     print(f"Saved model at episode {epi}")

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    avg_loss = np.mean(losses) if losses else 0
    
    episode_rewards.append(total_reward)
    episode_losses.append(avg_loss)
    episode_epsilons.append(EPSILON)
 
    if epi % EVAL_FREQ == 0:
        mean_eval, std_eval = Evaluate(env, qnet, num_episodes=EVAL_EPISODES, device=device)
        if mean_eval > BEST_EVAL:
            target_net.load_state_dict(qnet.state_dict())

            BEST_EVAL = mean_eval
            IMPROVEMENT_COUNT = 0
            torch.save(qnet.state_dict(), os.path.join(model_dir, f"best_lunar_lander_{hidden_dim}.pth"))
            print(f"Episode {epi:3d}: Mean reward {mean_eval:8.2f}, Std reward {std_eval:8.2f}, avg loss {avg_loss:8.4f}, epsilon {EPSILON:.3}")
            record(qnet,wind,video_dir,device, eps=3, name_prefix=f"lunar_lander_{hidden_dim}_at_{epi}")
        else: 
            print(f"No improvement on episode {epi}")
        if IMPROVEMENT_COUNT >= PATIENCE:
            print(f"Early stopping on episode {epi}, no improvement in {PATIENCE} episodes")
            break

from matplotlib import pyplot as plt

def ma(data, window=10, mode='valid'):
    return np.convolve(np.array(data).flatten(), np.ones(window) / window, mode=mode)

plt.plot(ma(episode_rewards), c='black', lw=1, label= f'MA')
plt.plot(episode_rewards, c='gray', alpha=0.5, lw=0.5, label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig(os.path.join(model_dir, f'reward_{hidden_dim}.png'))
np.save(os.path.join(model_dir, f'reward_{hidden_dim}.npy'), episode_rewards)

plt.plot(ma(episode_losses), c='black', lw=1, label='MA')
plt.plot(episode_losses, c='gray', alpha=0.5, lw=0.5, label='Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(model_dir, f'loss_{hidden_dim}.png'))
np.save(os.path.join(model_dir, f'loss_{hidden_dim}.npy'), episode_losses)

plt.plot(episode_epsilons, c='black', lw=1, label='Epsilon') 
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.legend()
plt.savefig(os.path.join(model_dir, f'epsilon_{hidden_dim}.png'))
np.save(os.path.join(model_dir, f'epsilon_{hidden_dim}.npy'), episode_epsilons)