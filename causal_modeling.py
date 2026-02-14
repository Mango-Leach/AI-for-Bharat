"""
Causal Modeling Module
Uses DoWhy and causal inference to understand relationships
"""

from dowhy import CausalModel
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx


class CausalGraph:
    """Build and analyze causal relationships"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.model = None
    
    def add_relationship(self, cause: str, effect: str, strength: float = 1.0):
        """Add causal edge to graph"""
        self.graph.add_edge(cause, effect, weight=strength)
    
    def build_town_causal_model(self):
        """Define causal relationships for town systems"""
        
        # Traffic causes pollution
        self.add_relationship("traffic_volume", "air_pollution", 0.7)
        
        # Weather affects multiple systems
        self.add_relationship("temperature", "water_demand", 0.6)
        self.add_relationship("rainfall", "water_supply", 0.8)
        self.add_relationship("humidity", "air_pollution", -0.3)
        
        # Economic relationships
        self.add_relationship("market_demand", "crop_prices", 0.9)
        self.add_relationship("crop_yield", "market_supply", 0.85)
        
        # Infrastructure impacts
        self.add_relationship("road_capacity", "traffic_congestion", -0.7)
        self.add_relationship("water_infrastructure", "water_supply", 0.75)
        
        # Policy interventions
        self.add_relationship("pollution_policy", "air_pollution", -0.5)
        self.add_relationship("traffic_policy", "traffic_volume", -0.4)
    
    def estimate_causal_effect(
        self, 
        data: pd.DataFrame, 
        treatment: str, 
        outcome: str,
        confounders: List[str]
    ) -> Dict:
        """Estimate causal effect using DoWhy"""
        
        # Create causal model
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
            graph=self._to_gml()
        )
        
        # Identify causal effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # Estimate effect
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching"
        )
        
        # Refute estimate
        refutation = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause"
        )
        
        return {
            "effect": estimate.value,
            "confidence_interval": estimate.get_confidence_intervals(),
            "refutation": refutation
        }
    
    def _to_gml(self) -> str:
        """Convert graph to GML format for DoWhy"""
        return "\n".join([
            "graph [",
            "  directed 1",
            *[f'  edge [source "{u}" target "{v}"]' for u, v in self.graph.edges()],
            "]"
        ])
    
    def find_intervention_points(self, target_outcome: str) -> List[str]:
        """Identify variables that can influence target outcome"""
        ancestors = nx.ancestors(self.graph, target_outcome)
        
        # Rank by path strength
        intervention_scores = {}
        for node in ancestors:
            paths = list(nx.all_simple_paths(self.graph, node, target_outcome))
            score = sum(
                np.prod([self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])])
                for path in paths
            )
            intervention_scores[node] = score
        
        return sorted(intervention_scores.items(), key=lambda x: abs(x[1]), reverse=True)


class StructuralCausalModel:
    """Structural equations for town systems"""
    
    def __init__(self):
        self.equations = {}
    
    def define_pollution_model(self):
        """Pollution = f(traffic, weather, industrial_activity)"""
        def pollution_eq(traffic, temp, humidity, industrial):
            return (
                0.5 * traffic +
                0.3 * temp +
                -0.2 * humidity +
                0.4 * industrial +
                np.random.normal(0, 5)
            )
        self.equations['pollution'] = pollution_eq
    
    def define_traffic_model(self):
        """Traffic = f(time, weather, events)"""
        def traffic_eq(hour, is_peak, weather_severity, event_factor):
            base = 100
            peak_multiplier = 2.0 if is_peak else 1.0
            weather_impact = 1.0 + (weather_severity * 0.3)
            return base * peak_multiplier * weather_impact * event_factor
        self.equations['traffic'] = traffic_eq
    
    def define_water_model(self):
        """Water demand = f(temperature, population, time)"""
        def water_eq(temp, population, is_summer, industrial_demand):
            base_per_capita = 150  # liters
            temp_factor = 1.0 + ((temp - 25) * 0.02)
            seasonal_factor = 1.3 if is_summer else 1.0
            return (
                population * base_per_capita * temp_factor * seasonal_factor +
                industrial_demand
            )
        self.equations['water'] = water_eq
    
    def simulate(self, variables: Dict[str, float]) -> Dict[str, float]:
        """Run structural equations with given variables"""
        results = {}
        for name, equation in self.equations.items():
            # Extract required parameters
            results[name] = equation(**variables)
        return results


def analyze_intervention_impact(
    causal_graph: CausalGraph,
    data: pd.DataFrame,
    intervention: str,
    intervention_value: float,
    target_outcomes: List[str]
) -> Dict[str, float]:
    """Simulate impact of intervention on multiple outcomes"""
    
    impacts = {}
    for outcome in target_outcomes:
        # Find confounders
        confounders = list(
            set(nx.ancestors(causal_graph.graph, intervention)) &
            set(nx.ancestors(causal_graph.graph, outcome))
        )
        
        # Estimate effect
        result = causal_graph.estimate_causal_effect(
            data, intervention, outcome, confounders
        )
        impacts[outcome] = result['effect'] * intervention_value
    
    return impacts
