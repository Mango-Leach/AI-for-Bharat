"""
Reinforcement Learning for Optimal Policy Discovery
Uses RL agents to find best interventions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Tuple, Any


class TownEnvironment(gym.Env):
    """Custom RL environment for town management"""
    
    def __init__(self):
        super(TownEnvironment, self).__init__()
        
        # State space: [traffic, pollution, water_level, market_price, ...]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([1000, 500, 100, 1000, 100, 100, 100, 100]),
            dtype=np.float32
        )
        
        # Action space: [traffic_control, pollution_limit, water_allocation, ...]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.state = None
        self.time_step = 0
        self.max_steps = 168  # One week in hours
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            500,  # traffic_volume
            100,  # pollution_level
            80,   # water_level
            500,  # market_price
            25,   # temperature
            60,   # humidity
            50,   # crop_health
            70    # citizen_satisfaction
        ], dtype=np.float32)
        self.time_step = 0
        return self.state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Extract actions
        traffic_control = action[0]
        pollution_limit = action[1]
        water_allocation = action[2]
        market_intervention = action[3]
        
        # Update state based on actions and dynamics
        self._update_traffic(traffic_control)
        self._update_pollution(pollution_limit)
        self._update_water(water_allocation)
        self._update_market(market_intervention)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        self.time_step += 1
        terminated = self.time_step >= self.max_steps
        truncated = False
        
        return self.state, reward, terminated, truncated, {}
    
    def _update_traffic(self, control: float):
        """Update traffic based on control action"""
        # Reduce traffic with control measures
        self.state[0] *= (1 - 0.3 * control)
        # Natural variation
        self.state[0] += np.random.normal(0, 20)
        self.state[0] = np.clip(self.state[0], 0, 1000)
    
    def _update_pollution(self, limit: float):
        """Update pollution based on limits"""
        # Pollution influenced by traffic and limits
        traffic_pollution = self.state[0] * 0.1
        self.state[1] = traffic_pollution * (1 - 0.5 * limit)
        self.state[1] = np.clip(self.state[1], 0, 500)
    
    def _update_water(self, allocation: float):
        """Update water levels based on allocation"""
        demand = 10 + self.state[4] * 0.5  # Temperature affects demand
        supply = allocation * 15
        self.state[2] += supply - demand
        self.state[2] = np.clip(self.state[2], 0, 100)
    
    def _update_market(self, intervention: float):
        """Update market prices"""
        self.state[3] *= (1 + np.random.normal(0, 0.05))
        self.state[3] *= (1 - 0.1 * intervention)  # Intervention stabilizes
        self.state[3] = np.clip(self.state[3], 100, 1000)
    
    def _calculate_reward(self) -> float:
        """Multi-objective reward function"""
        # Minimize pollution (weight: 0.3)
        pollution_penalty = -0.3 * (self.state[1] / 500)
        
        # Minimize traffic congestion (weight: 0.2)
        traffic_penalty = -0.2 * (self.state[0] / 1000)
        
        # Maintain water supply (weight: 0.2)
        water_reward = 0.2 * (self.state[2] / 100)
        
        # Stabilize market (weight: 0.15)
        market_stability = 0.15 * (1 - abs(self.state[3] - 500) / 500)
        
        # Citizen satisfaction (weight: 0.15)
        satisfaction = 0.15 * (self.state[7] / 100)
        
        return pollution_penalty + traffic_penalty + water_reward + market_stability + satisfaction


class RLPolicyOptimizer:
    """Train RL agents for optimal policy discovery"""
    
    def __init__(self, algorithm: str = "PPO"):
        self.env = DummyVecEnv([lambda: TownEnvironment()])
        self.algorithm = algorithm
        self.model = None
    
    def train(self, total_timesteps: int = 100000):
        """Train RL agent"""
        if self.algorithm == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=1)
        elif self.algorithm == "SAC":
            self.model = SAC("MlpPolicy", self.env, verbose=1)
        elif self.algorithm == "DQN":
            # DQN requires discrete actions, convert if needed
            self.model = PPO("MlpPolicy", self.env, verbose=1)
        
        self.model.learn(total_timesteps=total_timesteps)
    
    def get_optimal_action(self, state: np.ndarray) -> np.ndarray:
        """Get optimal action for given state"""
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained policy"""
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward[0]
            
            episode_rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards)
        }


class MultiAgentCoordination:
    """Coordinate multiple RL agents for different subsystems"""
    
    def __init__(self):
        self.agents = {}
    
    def add_agent(self, name: str, agent: RLPolicyOptimizer):
        self.agents[name] = agent
    
    def coordinate_actions(self, global_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get coordinated actions from all agents"""
        actions = {}
        for name, agent in self.agents.items():
            state = global_state.get(name)
            if state is not None:
                actions[name] = agent.get_optimal_action(state)
        return actions
