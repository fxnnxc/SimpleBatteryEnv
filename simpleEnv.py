import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

class BatteryEfficientNavEnv(gym.Env):
    def __init__(self):
        super(BatteryEfficientNavEnv, self).__init__()
        
        # Define constants
        self.max_position = 100.0  # Goal position
        self.max_velocity = 10.0
        self.drag_coefficient = 0.05
        self.dt = 0.1  # Time step
        self.max_steps = 500
        
        # Define spaces
        self.max_motor_force = 10.0
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, -self.max_velocity]),
            high=np.array([self.max_position, self.max_velocity]),
            dtype=np.float32
        )
        
        self.coef_velocity_efficiency = 1.0
        self.coef_battery_consumption = 0.01
        
        # 모터 제어를 위한 추가 변수
        self.current_motor_force = 0.0
        self.motor_force_step = 2.0  # 한 번에 변경할 모터값
        
        self.reset()
        
    
    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        self.steps = 0
        return np.array([self.position, self.velocity])
    
    def step(self, action):
        self.steps += 1
        
        # Discrete action을 모터값 변경으로 변환
        if action == 0:    # 감소
            self.current_motor_force = max(-self.max_motor_force, 
                                         self.current_motor_force - self.motor_force_step)
        elif action == 2:  # 증가
            self.current_motor_force = min(self.max_motor_force, 
                                         self.current_motor_force + self.motor_force_step)
        # action == 1은 유지이므로 아무것도 하지 않음
        
        # 이전 속도 저장 (가속도 계산용)
        previous_velocity = self.velocity
        
        # 모터 힘 적용
        motor_force = self.current_motor_force
        
        # 항력 계산
        drag_force = -self.drag_coefficient * self.velocity * abs(self.velocity)
        
        # 속도와 위치 업데이트
        total_force = motor_force + drag_force
        self.velocity += total_force * self.dt
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        self.position += self.velocity * self.dt
        
        # 가속도 계산
        acceleration = (self.velocity - previous_velocity) / self.dt
        
        # 목표까지 거리
        distance_to_goal = abs(self.max_position - self.position)
        
        # 연비 효율성 계산
        optimal_velocity = self.max_velocity * 0.6  # 최대 속도의 60%가 가장 효율적
        velocity_efficiency = 1.0 - abs(abs(self.velocity) - optimal_velocity) / self.max_velocity
        
        # 급가속 페널티
        acceleration_penalty = abs(acceleration) ** 2 * 0.1
        
        # 배터리 소비 (기본 소비 + 가속도에 따른 추가 소비)
        base_consumption = abs(motor_force) * self.dt
        acceleration_consumption = abs(acceleration) ** 2 * self.dt
        battery_consumption = base_consumption + acceleration_consumption
        
        # 종합 효율성 계산
        efficiency_factor = (velocity_efficiency * self.coef_velocity_efficiency  # 최적 속도 유지 보상
                           - acceleration_penalty      # 급가속 패널티
                           - battery_consumption * self.coef_battery_consumption)  # 배터리 소비 패널티
        
        # 최종 보상
        reward = (-0.01 * distance_to_goal    # 거리 패널티
                 + efficiency_factor * 10.0)   # 효율성 보상
        
        self.current_battery_consumption = battery_consumption
        
        # 종료 조건 체크
        done = False
        if self.position >= self.max_position or self.steps >= self.max_steps:
            done = True
            if self.position >= self.max_position:
                reward += 100  # 목표 도달 보너스
        
        info = {
            'distance_to_goal': distance_to_goal,
            'velocity_efficiency': velocity_efficiency,
            'acceleration_penalty': acceleration_penalty,
            'battery_consumption': battery_consumption,
            'current_velocity': self.velocity,
            'optimal_velocity': optimal_velocity
        }
        
        return np.array([self.position, self.velocity]), reward, done, info

def run_episode(env, render=True):
    obs = env.reset()
    done = False
    
    # Lists to store episode data for plotting
    positions = []
    velocities = []
    rewards = []
    battery_consumptions = []
    
    while not done:
        # Simple policy: higher speed when far from goal, lower speed when closer
        distance_to_goal = env.max_position - obs[0]
        action = np.array([min(5.0, max(2.0, distance_to_goal / 10))])
        
        obs, reward, done, info = env.step(action)
        
        # Store data for plotting
        positions.append(obs[0])
        velocities.append(obs[1])
        rewards.append(reward)
        battery_consumptions.append(info['battery_consumption'])
    
    return positions, velocities, rewards, battery_consumptions

def plot_episode(positions, velocities, rewards, battery_consumptions):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle('Episode Statistics')
    
    # Plot position over time
    axs[0].plot(positions, label='Position')
    axs[0].set_ylabel('Position')
    axs[0].set_xlabel('Time Step')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot velocity over time
    axs[1].plot(velocities, label='Velocity', color='orange')
    axs[1].set_ylabel('Velocity')
    axs[1].set_xlabel('Time Step')
    axs[1].grid(True)
    axs[1].legend()
    
    # Plot instantaneous rewards
    axs[2].plot(rewards, label='Reward', color='green')
    axs[2].set_ylabel('Reward')
    axs[2].set_xlabel('Time Step')
    axs[2].grid(True)
    axs[2].legend()
    
    # Plot cumulative battery consumption
    cumulative_battery = np.cumsum(battery_consumptions)
    axs[3].plot(cumulative_battery, label='Cumulative Battery', color='red')
    axs[3].set_ylabel('Battery Consumption')
    axs[3].set_xlabel('Time Step')
    axs[3].grid(True)
    axs[3].legend()
    
    plt.tight_layout()
    plt.savefig('simpleEnv.png')
    plt.show()
