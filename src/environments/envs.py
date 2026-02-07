"""Combinatorial optimization environments."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Union


class KnapsackEnv(gym.Env):
    """Knapsack Problem environment.
    
    The agent must select items to maximize value while staying within capacity constraints.
    """
    
    def __init__(
        self,
        num_items: int = 10,
        capacity: Optional[int] = None,
        value_range: Tuple[int, int] = (1, 100),
        weight_range: Tuple[int, int] = (1, 20),
        seed: Optional[int] = None,
    ):
        """Initialize Knapsack environment.
        
        Args:
            num_items: Number of items to choose from.
            capacity: Knapsack capacity. If None, set to 50% of total possible weight.
            value_range: Range for item values.
            weight_range: Range for item weights.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        
        self.num_items = num_items
        self.value_range = value_range
        self.weight_range = weight_range
        
        # Generate random problem instance
        self.rng = np.random.RandomState(seed)
        self.values = self.rng.randint(value_range[0], value_range[1] + 1, num_items)
        self.weights = self.rng.randint(weight_range[0], weight_range[1] + 1, num_items)
        
        if capacity is None:
            self.capacity = int(0.5 * np.sum(self.weights))
        else:
            self.capacity = capacity
        
        # Action space: which item to select (0 to num_items-1)
        self.action_space = spaces.Discrete(num_items)
        
        # Observation space: current selection state + item info
        # State includes: selected items (binary) + item values + item weights + remaining capacity
        obs_size = num_items + num_items + num_items + 1  # selection + values + weights + capacity
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Initial observation and info dict.
        """
        if seed is not None:
            self.rng.seed(seed)
        
        self.selected_items = np.zeros(self.num_items, dtype=np.float32)
        self.current_weight = 0.0
        self.current_value = 0.0
        
        obs = self._get_observation()
        info = {
            "current_value": self.current_value,
            "current_weight": self.current_weight,
            "capacity": self.capacity,
            "optimal_value": self._get_optimal_value(),
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Item index to select.
            
        Returns:
            Next observation, reward, terminated, truncated, info.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Check if item is already selected
        if self.selected_items[action] == 1:
            reward = -0.1  # Small penalty for invalid action
            terminated = False
            truncated = False
        else:
            # Try to select the item
            new_weight = self.current_weight + self.weights[action]
            
            if new_weight <= self.capacity:
                # Item fits, select it
                self.selected_items[action] = 1
                self.current_weight = new_weight
                self.current_value += self.values[action]
                reward = self.values[action]  # Reward equals item value
            else:
                # Item doesn't fit
                reward = -1.0  # Penalty for exceeding capacity
                terminated = False
                truncated = False
        
        # Check termination conditions
        if np.all(self.selected_items == 1):
            terminated = True
            truncated = False
        elif self.current_weight >= self.capacity:
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = False
        
        obs = self._get_observation()
        info = {
            "current_value": self.current_value,
            "current_weight": self.current_weight,
            "capacity": self.capacity,
            "selected_items": self.selected_items.copy(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Current observation vector.
        """
        # Normalize values and weights to [0, 1]
        normalized_values = self.values / np.max(self.values)
        normalized_weights = self.weights / np.max(self.weights)
        normalized_capacity = self.capacity / np.max(self.weights)
        
        obs = np.concatenate([
            self.selected_items,
            normalized_values,
            normalized_weights,
            [normalized_capacity]
        ]).astype(np.float32)
        
        return obs
    
    def _get_optimal_value(self) -> float:
        """Calculate optimal value using dynamic programming.
        
        Returns:
            Optimal knapsack value.
        """
        # Simple greedy approximation (not optimal but good baseline)
        items = list(zip(self.values, self.weights))
        items.sort(key=lambda x: x[0] / x[1], reverse=True)  # Sort by value/weight ratio
        
        total_value = 0
        total_weight = 0
        
        for value, weight in items:
            if total_weight + weight <= self.capacity:
                total_value += value
                total_weight += weight
        
        return total_value
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment.
        
        Args:
            mode: Rendering mode.
            
        Returns:
            Rendered output if mode is "rgb_array".
        """
        if mode == "human":
            print(f"Knapsack State:")
            print(f"  Selected items: {self.selected_items}")
            print(f"  Current value: {self.current_value}")
            print(f"  Current weight: {self.current_weight}/{self.capacity}")
            print(f"  Items: values={self.values}, weights={self.weights}")
        elif mode == "rgb_array":
            # Could implement visual rendering here
            return None
        else:
            raise ValueError(f"Unknown render mode: {mode}")


class TSPEnv(gym.Env):
    """Traveling Salesman Problem environment.
    
    The agent must visit all cities exactly once and return to the starting city.
    """
    
    def __init__(
        self,
        num_cities: int = 10,
        city_range: Tuple[int, int] = (0, 100),
        seed: Optional[int] = None,
    ):
        """Initialize TSP environment.
        
        Args:
            num_cities: Number of cities to visit.
            city_range: Range for city coordinates.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        
        self.num_cities = num_cities
        self.city_range = city_range
        
        # Generate random city coordinates
        self.rng = np.random.RandomState(seed)
        self.cities = self.rng.uniform(
            city_range[0], city_range[1], (num_cities, 2)
        ).astype(np.float32)
        
        # Calculate distance matrix
        self.distances = self._calculate_distances()
        
        # Action space: which city to visit next
        self.action_space = spaces.Discrete(num_cities)
        
        # Observation space: current tour + city coordinates + distances
        obs_size = num_cities + num_cities * 2 + num_cities * num_cities
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Initial observation and info dict.
        """
        if seed is not None:
            self.rng.seed(seed)
        
        self.tour = []
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.current_city = 0
        self.total_distance = 0.0
        
        obs = self._get_observation()
        info = {
            "tour": self.tour.copy(),
            "total_distance": self.total_distance,
            "optimal_distance": self._get_optimal_distance(),
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: City index to visit next.
            
        Returns:
            Next observation, reward, terminated, truncated, info.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Check if city is already visited
        if self.visited[action]:
            reward = -10.0  # Penalty for revisiting
            terminated = False
            truncated = False
        else:
            # Visit the city
            distance = self.distances[self.current_city, action]
            self.total_distance += distance
            
            self.tour.append(action)
            self.visited[action] = True
            self.current_city = action
            
            # Reward is negative distance (we want to minimize total distance)
            reward = -distance
            
            # Check if all cities visited
            if len(self.tour) == self.num_cities:
                # Return to starting city
                return_distance = self.distances[self.current_city, 0]
                self.total_distance += return_distance
                reward -= return_distance
                terminated = True
                truncated = False
            else:
                terminated = False
                truncated = False
        
        obs = self._get_observation()
        info = {
            "tour": self.tour.copy(),
            "total_distance": self.total_distance,
            "current_city": self.current_city,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Current observation vector.
        """
        # Normalize coordinates
        normalized_cities = self.cities / np.max(self.city_range)
        
        # Normalize distances
        normalized_distances = self.distances / np.max(self.distances)
        
        obs = np.concatenate([
            self.visited.astype(np.float32),
            normalized_cities.flatten(),
            normalized_distances.flatten()
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate Euclidean distances between all cities.
        
        Returns:
            Distance matrix.
        """
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    distances[i, j] = np.sqrt(
                        np.sum((self.cities[i] - self.cities[j]) ** 2)
                    )
        return distances
    
    def _get_optimal_distance(self) -> float:
        """Calculate optimal TSP distance using nearest neighbor heuristic.
        
        Returns:
            Approximate optimal distance.
        """
        # Simple nearest neighbor heuristic
        visited = np.zeros(self.num_cities, dtype=bool)
        current = 0
        total_distance = 0.0
        
        for _ in range(self.num_cities - 1):
            visited[current] = True
            distances = self.distances[current]
            distances[visited] = np.inf  # Don't revisit
            
            next_city = np.argmin(distances)
            total_distance += distances[next_city]
            current = next_city
        
        # Return to start
        total_distance += self.distances[current, 0]
        
        return total_distance
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment.
        
        Args:
            mode: Rendering mode.
            
        Returns:
            Rendered output if mode is "rgb_array".
        """
        if mode == "human":
            print(f"TSP State:")
            print(f"  Tour: {self.tour}")
            print(f"  Total distance: {self.total_distance:.2f}")
            print(f"  Current city: {self.current_city}")
        elif mode == "rgb_array":
            # Could implement visual rendering here
            return None
        else:
            raise ValueError(f"Unknown render mode: {mode}")


def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create environment by name.
    
    Args:
        env_name: Environment name.
        **kwargs: Environment-specific arguments.
        
    Returns:
        Environment instance.
    """
    if env_name == "knapsack":
        return KnapsackEnv(**kwargs)
    elif env_name == "tsp":
        return TSPEnv(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
