import numpy as np

class NSGA2Fitness(float):

    def __new__(cls, value, *args, **kwargs):
        return super(NSGA2Fitness, cls).__new__(cls, value)
    
    def __init__(self, value, values) -> None:
        super().__init__()
        self.rank = 0
        self.crowding_dist = 0
        self.values = values
    
    def dominates(self, other) -> bool:
        dominates = False
        for a,b in zip(self.values, other.values):
            if a < b:
                return False
            elif a > b:
                dominates = True
        return dominates
    
    def __gt__(self, value) -> bool:
        # Use crowded comparison operator
        if isinstance(value, NSGA2Fitness):
            if self.rank > value.rank:
                return True
            elif self.rank == value.rank:
                return self.crowding_dist > value.crowding_dist
            return False
        return np.mean(self.values) > value
    
    def __lt__(self, value) -> bool:
        if isinstance(value, NSGA2Fitness):
            if self.rank < value.rank:
                return True
            elif self.rank == value.rank:
                return self.crowding_dist < value.crowding_dist
            return False
        return np.mean(self.values) < value
        
    def __str__(self) -> str:
        return f"Rank: {self.rank}, Crowding Distance: {self.crowding_dist}, Values: {self.values}"
    
    def __repr__(self) -> str:
        return self.__str__()