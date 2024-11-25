import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from simpleEnv import BatteryEfficientNavEnv
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),  # 3개의 행동에 대한 로짓값 출력
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        # Actor: get action probabilities
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        # Critic: get state value
        value = self.critic(state)
        
        return dist, value

class PPOTrainer:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n  # Discrete action space
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
    
    def collect_trajectories(self, num_steps):
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        state = self.env.reset()
        episode_reward = 0
        
        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                dist, value = self.model(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_state, reward, done, _ = self.env.step(action.item())  # Discrete action
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob)
            dones.append(done)
            
            episode_reward += reward
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
        
        with torch.no_grad():
            _, final_value = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            
        return states, actions, rewards, values, log_probs, dones, final_value.item(), episode_reward
    
    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        
        for _ in range(10):
            dist, values = self.model(states)
            new_log_probs = dist.log_prob(actions)
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((values - returns) ** 2).mean()
            entropy_loss = -dist.entropy().mean()
            
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

    def train(self, num_episodes=1000):
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Collect trajectories
            states, actions, rewards, values, log_probs, dones, final_value, episode_reward = \
                self.collect_trajectories(num_steps=200)
            
            # Calculate advantages and returns
            advantages, returns = self.compute_gae(rewards, values, final_value, dones)
            
            # Update policy
            self.update_policy(states, actions, log_probs, advantages, returns)
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
        
        return episode_rewards

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

def plot_training_results(rewards, base_path):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig(os.path.join(base_path, 'training_results.png'))
    plt.show()

if __name__ == "__main__":
    # Create environment
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--coef_battery_consumption', '-b', type=float, default=0.050)
    args = parser.parse_args()
    import os 
    base_path = os.path.join("outputs", f"battery_consumption_{args.coef_battery_consumption:.4f}")
    os.makedirs(base_path, exist_ok=True)
    
    env = BatteryEfficientNavEnv()
    env.coef_battery_consumption = args.coef_battery_consumption
    env.coef_velocity_efficiency = 0.11732748693215561
    
    # Create and train PPO agent
    trainer = PPOTrainer(env)
    rewards = trainer.train(num_episodes=100000)
    
    # Plot results
    plot_training_results(rewards, base_path)
    
    # Save model
    torch.save(trainer.model.state_dict(), os.path.join(base_path, 'ppo_model.pth')) 