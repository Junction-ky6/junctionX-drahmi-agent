from pydantic import BaseModel

class Observation(BaseModel):
    money: int = 500
    categories_ratios: list = [0.2, 0.25, 0.2, 0.2, 0.15]
    user_satisfactions: list = [0.5, 0.6, 0.4, 0.4, 0.5]
    goal_achievement: float = 0.4
    investment_profit: int = 80
    
class Prediction(BaseModel):
    categories_ratios: list
    goal_ratio : float 
    investment_ratio : float