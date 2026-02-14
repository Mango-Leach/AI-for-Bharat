"""
Decision Support System
Recommendation engine for optimal interventions
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from ortools.linear_solver import pywraplp
from dataclasses import dataclass


@dataclass
class Intervention:
    name: str
    category: str
    cost: float
    impact: Dict[str, float]
    constraints: Dict[str, Any]
    priority: int


class RecommendationEngine:
    """Generate optimal intervention recommendations"""
    
    def __init__(self):
        self.interventions: List[Intervention] = []
        self.constraints = {}
    
    def add_intervention(self, intervention: Intervention):
        """Add possible intervention"""
        self.interventions.append(intervention)
    
    def set_budget_constraint(self, budget: float):
        """Set total budget constraint"""
        self.constraints["budget"] = budget
    
    def optimize(self, objectives: Dict[str, float]) -> List[Intervention]:
        """Find optimal set of interventions using linear programming"""
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return []
        
        # Decision variables: whether to implement each intervention
        x = {}
        for i, intervention in enumerate(self.interventions):
            x[i] = solver.BoolVar(f'x_{i}')
        
        # Objective: maximize weighted impact
        objective = solver.Objective()
        for i, intervention in enumerate(self.interventions):
            total_impact = sum(
                intervention.impact.get(obj, 0) * weight
                for obj, weight in objectives.items()
            )
            objective.SetCoefficient(x[i], total_impact)
        objective.SetMaximization()
        
        # Budget constraint
        budget_constraint = solver.Constraint(0, self.constraints.get("budget", float('inf')))
        for i, intervention in enumerate(self.interventions):
            budget_constraint.SetCoefficient(x[i], intervention.cost)
        
        # Solve
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            selected = [
                self.interventions[i]
                for i in range(len(self.interventions))
                if x[i].solution_value() > 0.5
            ]
            return selected
        
        return []
    
    def rank_interventions(self, current_state: Dict[str, float]) -> List[Tuple[Intervention, float]]:
        """Rank interventions by expected impact"""
        scores = []
        
        for intervention in self.interventions:
            # Calculate urgency score
            urgency = 0
            for metric, value in current_state.items():
                if metric in intervention.impact:
                    # Higher urgency if metric is critical
                    if value > 80:  # Critical threshold
                        urgency += intervention.impact[metric] * 2
                    else:
                        urgency += intervention.impact[metric]
            
            # Adjust for cost-effectiveness
            cost_effectiveness = urgency / max(intervention.cost, 1)
            
            scores.append((intervention, cost_effectiveness))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)


class RiskAssessment:
    """Assess risks and vulnerabilities"""
    
    def __init__(self):
        self.risk_factors = {}
    
    def assess_pollution_risk(self, pollution_level: float, trend: float) -> Dict[str, Any]:
        """Assess pollution-related risks"""
        risk_level = "low"
        if pollution_level > 150:
            risk_level = "high"
        elif pollution_level > 100:
            risk_level = "medium"
        
        # Consider trend
        if trend > 0.1:
            risk_level = "critical" if risk_level == "high" else "high"
        
        return {
            "risk_level": risk_level,
            "current_value": pollution_level,
            "trend": trend,
            "health_impact": pollution_level * 0.5,
            "economic_impact": pollution_level * 100,  # INR per day
            "recommendations": self._get_pollution_recommendations(risk_level)
        }
    
    def assess_water_risk(self, water_level: float, demand: float) -> Dict[str, Any]:
        """Assess water shortage risks"""
        supply_ratio = water_level / max(demand, 1)
        
        risk_level = "low"
        if supply_ratio < 0.5:
            risk_level = "critical"
        elif supply_ratio < 0.7:
            risk_level = "high"
        elif supply_ratio < 0.9:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "supply_ratio": supply_ratio,
            "days_remaining": water_level / max(demand, 1),
            "recommendations": self._get_water_recommendations(risk_level)
        }
    
    def assess_traffic_risk(self, congestion_level: float, peak_hours: int) -> Dict[str, Any]:
        """Assess traffic congestion risks"""
        risk_level = "low"
        if congestion_level > 80:
            risk_level = "high"
        elif congestion_level > 60:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "congestion_level": congestion_level,
            "peak_hours": peak_hours,
            "economic_loss": congestion_level * 50000,  # INR per day
            "recommendations": self._get_traffic_recommendations(risk_level)
        }
    
    def _get_pollution_recommendations(self, risk_level: str) -> List[str]:
        if risk_level == "critical":
            return [
                "Implement odd-even vehicle scheme",
                "Shut down high-emission industries temporarily",
                "Issue health advisory for vulnerable groups",
                "Deploy water sprinklers on major roads"
            ]
        elif risk_level == "high":
            return [
                "Increase public transport frequency",
                "Restrict heavy vehicle entry",
                "Monitor industrial emissions closely"
            ]
        return ["Continue regular monitoring"]
    
    def _get_water_recommendations(self, risk_level: str) -> List[str]:
        if risk_level == "critical":
            return [
                "Implement water rationing immediately",
                "Ban non-essential water use",
                "Deploy water tankers to critical areas",
                "Expedite alternative source development"
            ]
        elif risk_level == "high":
            return [
                "Reduce supply hours",
                "Launch water conservation campaign",
                "Fix identified leaks urgently"
            ]
        return ["Monitor consumption patterns"]
    
    def _get_traffic_recommendations(self, risk_level: str) -> List[str]:
        if risk_level == "high":
            return [
                "Optimize traffic signal timing",
                "Deploy traffic police at key junctions",
                "Encourage work-from-home",
                "Increase public transport capacity"
            ]
        return ["Continue regular traffic management"]


def create_intervention_library() -> List[Intervention]:
    """Create library of possible interventions"""
    return [
        Intervention(
            name="Traffic Signal Optimization",
            category="traffic",
            cost=500000,
            impact={"traffic": -15, "pollution": -5},
            constraints={},
            priority=1
        ),
        Intervention(
            name="Odd-Even Vehicle Scheme",
            category="traffic",
            cost=2000000,
            impact={"traffic": -30, "pollution": -20},
            constraints={"duration": "temporary"},
            priority=2
        ),
        Intervention(
            name="Water Leak Detection & Repair",
            category="water",
            cost=1000000,
            impact={"water_availability": 10},
            constraints={},
            priority=1
        ),
        Intervention(
            name="Drip Irrigation Subsidy",
            category="agriculture",
            cost=5000000,
            impact={"water_usage": -20, "crop_yield": 15},
            constraints={},
            priority=2
        ),
        Intervention(
            name="Industrial Emission Controls",
            category="pollution",
            cost=10000000,
            impact={"pollution": -40},
            constraints={"compliance_time": 90},
            priority=1
        ),
        Intervention(
            name="Public Transport Expansion",
            category="traffic",
            cost=50000000,
            impact={"traffic": -25, "pollution": -15, "citizen_satisfaction": 10},
            constraints={"implementation_time": 180},
            priority=3
        )
    ]
