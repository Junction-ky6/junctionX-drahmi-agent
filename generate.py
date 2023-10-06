import json
import random

def uniform(n, epsilon=0.01):
    random_numbers = [random.random() for _ in range(n)]
    total = sum(random_numbers)
    probabilities = [x / total for x in random_numbers]
    uniform_prob = 1.0 / n
    for i in range(n):
        probabilities[i] = probabilities[i] + (uniform_prob - probabilities[i]) * epsilon

    return probabilities

def satisfy(n):
    p = []
    for _ in range(n):
        p.append(random.uniform(0.2, 0.9))
    return p

generations = 10000
gens = []
for _ in range(generations):
    money = random.randint(200, 1000)
    actual_categories_ratios = uniform(5)
    user_satisfactions = satisfy(5)
    investment_icome = random.randint(0, 250)
    
    gens.append({
        'money': money,
        'actual_categories_ratios': actual_categories_ratios,
        'user_satisfactions': user_satisfactions,
        'investment_icome': investment_icome
    })
    
with open("data/record.json", "w") as f:
    json.dump(gens, f, indent=4)