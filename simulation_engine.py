"""
Digital Twin Simulation Engine
Agent-based modeling and scenario simulation
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from enum import Enum
import random


class AgentType(Enum):
    CITIZEN = "citizen"
    VEHICLE = "vehicle"
    BUSINESS = "business"
    FARM = "farm"


@dataclass
class Agent:
    id: str
    agent_type: AgentType
    location: tuple
    state: Dict[str, Any]
    
    def update(self, environment: 'Environment'):
        """Update agent state based on environment"""
        pass


class Citizen(Agent):
    def __init__(self, id: str, location: tuple):
        super().__init__(
            id=id,
            agent_type=AgentType.CITIZEN,
            location=location,
            state={
                "water_usage": 150,  # liters/day
                "satisfaction": 70,
                "health": 80,
                "income": 30000
            }
        )
    
    def update(self, environment: 'Environment'):
        # Water usage affected by temperature
        temp = environment.get_temperature(self.location)
        self.state["water_usage"] = 150 * (1 + (temp - 25) * 0.02)
        
        # Satisfaction affected by pollution
        pollution = environment.get_pollution(self.location)
        self.state["satisfaction"] = max(0, 100 - pollution * 0.5)
        
        # Health affected by pollution
        self.state["health"] = max(0, 100 - pollution * 0.3)


class Vehicle(Agent):
    def __init__(self, id: str, location: tuple, vehicle_type: str):
        super().__init__(
            id=id,
            agent_type=AgentType.VEHICLE,
            location=location,
            state={
                "type": vehicle_type,
                "speed": 0,
                "emissions": 0,
                "destination": None
            }
        )
    
    def update(self, environment: 'Environment'):
        # Move towards destination
        if self.state["destination"]:
            traffic = environment.get_traffic(self.location)
            self.state["speed"] = max(10, 60 - traffic * 0.5)
            
            # Calculate emissions
            if self.state["type"] == "petrol":
                self.state["emissions"] = 120  # g CO2/km
            elif self.state["type"] == "diesel":
                self.state["emissions"] = 100
            elif self.state["type"] == "electric":
                self.state["emissions"] = 0


class Environment:
    """Simulated environment state"""
    
    def __init__(self, grid_size: tuple = (100, 100)):
        self.grid_size = grid_size
        self.temperature_map = np.random.uniform(20, 35, grid_size)
        self.pollution_map = np.random.uniform(0, 200, grid_size)
        self.traffic_map = np.random.uniform(0, 100, grid_size)
        self.water_availability = np.random.uniform(50, 100, grid_size)
    
    def get_temperature(self, location: tuple) -> float:
        x, y = int(location[0]), int(location[1])
        return self.temperature_map[x % self.grid_size[0], y % self.grid_size[1]]
    
    def get_pollution(self, location: tuple) -> float:
        x, y = int(location[0]), int(location[1])
        return self.pollution_map[x % self.grid_size[0], y % self.grid_size[1]]
    
    def get_traffic(self, location: tuple) -> float:
        x, y = int(location[0]), int(location[1])
        return self.traffic_map[x % self.grid_size[0], y % self.grid_size[1]]
    
    def update(self, agents: List[Agent]):
        """Update environment based on agent actions"""
        # Reset maps
        self.pollution_map *= 0.9  # Natural decay
        self.traffic_map *= 0.8
        
        # Aggregate agent impacts
        for agent in agents:
            x, y = int(agent.location[0]), int(agent.location[1])
            x, y = x % self.grid_size[0], y % self.grid_size[1]
            
            if agent.agent_type == AgentType.VEHICLE:
                self.traffic_map[x, y] += 1
                self.pollution_map[x, y] += agent.state["emissions"] * 0.01


class DigitalTwinSimulation:
    """Main simulation orchestrator"""
    
    def __init__(self, num_citizens: int = 10000, num_vehicles: int = 5000):
        self.environment = Environment()
        self.agents: List[Agent] = []
        self.time_step = 0
        self.metrics_history = []
        
        # Initialize agents
        self._initialize_agents(num_citizens, num_vehicles)
    
    def _initialize_agents(self, num_citizens: int, num_vehicles: int):
        """Create initial agent population"""
        # Create citizens
        for i in range(num_citizens):
            location = (
                random.uniform(0, self.environment.grid_size[0]),
                random.uniform(0, self.environment.grid_size[1])
            )
            self.agents.append(Citizen(f"C{i}", location))
        
        # Create vehicles
        for i in range(num_vehicles):
            location = (
                random.uniform(0, self.environment.grid_size[0]),
                random.uniform(0, self.environment.grid_size[1])
            )
            vehicle_type = random.choice(["petrol", "diesel", "electric"])
            self.agents.append(Vehicle(f"V{i}", location, vehicle_type))
    
    def step(self):
        """Execute one simulation step"""
        # Update all agents
        for agent in self.agents:
            agent.update(self.environment)
        
        # Update environment
        self.environment.update(self.agents)
        
        # Collect metrics
        self._collect_metrics()
        
        self.time_step += 1
    
    def _collect_metrics(self):
        """Collect system-wide metrics"""
        citizens = [a for a in self.agents if a.agent_type == AgentType.CITIZEN]
        vehicles = [a for a in self.agents if a.agent_type == AgentType.VEHICLE]
        
        metrics = {
            "time_step": self.time_step,
            "avg_satisfaction": np.mean([c.state["satisfaction"] for c in citizens]),
            "avg_health": np.mean([c.state["health"] for c in citizens]),
            "total_water_usage": sum([c.state["water_usage"] for c in citizens]),
            "avg_pollution": np.mean(self.environment.pollution_map),
            "avg_traffic": np.mean(self.environment.traffic_map),
            "total_emissions": sum([v.state["emissions"] for v in vehicles])
        }
        
        self.metrics_history.append(metrics)
    
    def run(self, num_steps: int):
        """Run simulation for specified steps"""
        for _ in range(num_steps):
            self.step()
    
    def get_metrics(self) -> List[Dict]:
        """Return collected metrics"""
        return self.metrics_history


class ScenarioGenerator:
    """Generate and compare different scenarios"""
    
    def __init__(self):
        self.scenarios = {}
    
    def add_scenario(self, name: str, config: Dict[str, Any]):
        """Add scenario configuration"""
        self.scenarios[name] = config
    
    def run_scenario(self, name: str, num_steps: int = 168) -> Dict:
        """Run specific scenario"""
        config = self.scenarios[name]
        
        sim = DigitalTwinSimulation(
            num_citizens=config.get("num_citizens", 10000),
            num_vehicles=config.get("num_vehicles", 5000)
        )
        
        # Apply scenario-specific modifications
        if "pollution_policy" in config:
            # Reduce vehicle emissions
            for agent in sim.agents:
                if agent.agent_type == AgentType.VEHICLE:
                    agent.state["emissions"] *= (1 - config["pollution_policy"])
        
        sim.run(num_steps)
        
        return {
            "scenario": name,
            "metrics": sim.get_metrics(),
            "final_state": sim.metrics_history[-1] if sim.metrics_history else {}
        }
    
    def compare_scenarios(self, scenario_names: List[str]) -> Dict:
        """Compare multiple scenarios"""
        results = {}
        for name in scenario_names:
            results[name] = self.run_scenario(name)
        return results
