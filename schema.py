from pydantic import BaseModel

class Observation(BaseModel):
    money: int = 500
    categories_ratios: list[float] = [0.2, 0.1, 0.4, 0.2, 0.1]
    user_satisfactions: list[float] = [0.5, 0.6, 0.4, 0.4, 0.7]
    goal_achievement: float = 0.4
    investment_profit: int = 80
    
class Prediction(BaseModel):
    categories_ratios: list[float]
    goal_ratio : float 
    investment_ratio : float