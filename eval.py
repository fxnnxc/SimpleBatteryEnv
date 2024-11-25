import torch
import numpy as np
import matplotlib.pyplot as plt
from simpleEnv import BatteryEfficientNavEnv
from train_ppo import ActorCritic

def evaluate_episode(env, model, device):
    state = env.reset()
    done = False
    
    # 기록할 데이터 초기화
    positions = []
    velocities = []
    motor_forces = []
    rewards = []
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            dist, _ = model(state_tensor)
            action = dist.sample()
        
        next_state, reward, done, info = env.step(action.item())
        
        # 데이터 저장
        positions.append(env.position)
        velocities.append(env.velocity)
        motor_forces.append(env.current_motor_force)
        rewards.append(reward)
        
        state = next_state
    
    return {
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'motor_forces': np.array(motor_forces),
        'rewards': np.array(rewards),
        'cumulative_reward': sum(rewards)
    }

def plot_episode_data(ax, data, episode_num):
    steps = np.arange(len(data['positions']))
    
    # 첫 번째 y축 (위치와 속도)
    line1 = ax.plot(steps, data['positions'], 'b-', label='Position')
    line2 = ax.plot(steps, data['velocities'], 'g-', label='Velocity')
    ax.set_ylabel('Position / Velocity')
    ax.grid(True)
    
    # 두 번째 y축 (모터력)
    ax2 = ax.twinx()
    line3 = ax2.plot(steps, data['motor_forces'], 'r-', label='Motor Force')
    ax2.set_ylabel('Motor Force')
    
    # 범례 통합
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    # 제목 (총 보상 포함)
    ax.set_title(f'Episode {episode_num+1}\nTotal Reward: {data["cumulative_reward"]:.2f}')
    ax.set_xlabel('Time Steps')

def evaluate_and_visualize(model_path, save_path, num_episodes=9):
    # 환경과 모델 설정
    env = BatteryEfficientNavEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 서브플롯 설정
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    # 여러 에피소드 평가 및 시각화
    episode_data = []
    for i in range(num_episodes):
        data = evaluate_episode(env, model, device)
        plot_episode_data(axes[i], data, i)
        episode_data.append(data)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'evaluation_results.png'))
    plt.show()
    
    # 평균 성능 출력
    avg_reward = np.mean([data['cumulative_reward'] for data in episode_data])
    print(f"\nAverage total reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return episode_data

if __name__ == "__main__":
    import os 
    save_path = 'outputs/battery_consumption_0.0500'
    model_path = os.path.join(save_path, 'ppo_model.pth')  # 학습된 모델 경로
    episode_data = evaluate_and_visualize(model_path, save_path)
