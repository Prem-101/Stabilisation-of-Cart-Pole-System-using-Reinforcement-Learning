# CartPole Stabilisation using Deep Reinforcement Learning

This project implements a Deep Q-Learning (DQN) approach to stabiliwe the CartPole system, comparing standard and physics-based reward functions.
The research work was done as a part of Summer Training under the guidance of Dr. Subrat Kumar Swain, Assisstant Professor, Department of Electrical and Electronics Engineering, Birla Institute of Technology, Mesra, Ranchi, Jharkhand.
The code has been inspired by the PyTorch tutorial on Reinforcement Learning (DQN) authored by Sir Adam  Paszke.

## Overview

The project focuses on two approaches:
1. Standard CartPole with default reward (+1 per timestep)
2. Physics-based reward considering angle, position for the cart and pole, and control effort

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/cartpole-stabilisation.git
cd cartpole-stabilisation

# Install dependencies
pip install -r requirements.txt

# Run experiments
jupyter notebook rewexp.ipynb
```

## Implementation Details

### Physics-Based Reward Function

```python
reward = -(α·θ² + β·θ̇² + γ·x² + κ·ẋ² + λ·F²)
```

where:
- θ: Pole angle from vertical
- θ̇: Angular velocity
- x: Cart position
- ẋ: Cart velocity
- F: Applied force

Parameters:
- α = 2.0 (angle penalty)
- β = 0.3 (angular velocity penalty)
- γ = 0.25 (position penalty)
- κ = 0.01 (velocity penalty)
- λ = 0.001 (control effort penalty)

Additionally, a bonus reward of +5 is given when |θ| < 5°.
The logic involves that if the pole angle is within the threshold value, the reward is the `angle_bonus`, otherwise the reward is the negative of the quadratic cart position error.

### DQN Architecture
- Input layer: 4 neurons (state variables)
- Hidden layers: 2x128 neurons with ReLU
- Output layer: 2 neurons (actions)
- Optimizer: AdamW with learning rate 3e-4
- Experience replay buffer: 10000 transitions
- Batch size: 128
- Discount factor (γ): 0.99
- Target network update rate (τ): 0.005

## Results

The physics-based reward approach demonstrates:
1. Faster convergence to stable behavior
2. Better angle stabilization (±5° vs ±10°)
3. More consistent performance
4. Lower control effort

## Project Structure
```
.
├── rewexp.ipynb          # Main experimental notebook
├── README.md            # Project documentation
├── requirements.txt     # Dependencies
├── LICENSE             # MIT License
└── .gitignore         # Git ignore rules
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```math
\ddot{\theta} = \frac{g \sin(\theta) - \cos(\theta) \ddot{x} }{\ell}
```

## Implementation Details

### DQN Architecture
```python
DQN(
    input_size=4,          # State space dimension
    hidden_layers=[128, 128], # Two hidden layers
    output_size=2          # Action space dimension
)
```

### Training Parameters
```yaml
batch_size: 128
gamma: 0.99
learning_rate: 3e-4
memory_size: 10000
target_update: 10
episodes: 600
```

### Reward Functions

1. **Standard Reward**
```python
r = 1.0  # for each timestep
```

2. **Physics-Based Reward**
```python
r = -(α·θ² + β·θ̇² + γ·x² + κ·ẋ² + λ·F²)
where:
α = 2.0  # angle penalty
β = 0.3  # angular velocity penalty
γ = 0.25 # position penalty
κ = 0.01 # velocity penalty
λ = 0.001 # control effort penalty
```

## Results Analysis

### Performance Metrics

1. **Stability**
   - Average episode length: 500+ steps
   - Angle deviation: < ±5°
   - Position bounds: < ±2.4m

2. **Learning Efficiency**
   - Convergence: ~300 episodes
   - Success rate: >95%

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request
