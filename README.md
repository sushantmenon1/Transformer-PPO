# Transformer + PPO Trading Model

## Overview

This repository contains the implementation of a custom trading model that combines Transformer-based neural networks with Proximal Policy Optimization (PPO), a reinforcement learning algorithm. The objective is to generate profitable trade recommendations by leveraging the strengths of both Transformers and PPO to effectively navigate and predict market trends.

## Project Structure

- **`Custom Network Architecture`**: The core of this model is a custom Transformer-based network that operates within the PPO framework. This model is designed to extract and learn complex patterns from sequential financial data, allowing it to make informed trade decisions.
  
- **`Fine-Tuning`**: Critical hyperparameters such as the learning rate, batch size, clip range, and gamma were fine-tuned to achieve optimal model performance.

- **`Evaluation`**: The model's performance was evaluated against a simple trading blotter, showing significant improvement in profitability.

## Features

### 1. Combining Transformer with PPO for Enhanced Decision-Making
The trading model is designed by combining the strengths of Transformers and PPO:
- **Transformer**: Excels in capturing temporal dependencies and complex patterns in time-series data, crucial for financial markets.
- **PPO**: An RL algorithm that learns optimal trading policies by maximizing cumulative rewards over time.

### 2. Custom Transformer-Based Network Architecture
The custom network architecture integrates a Transformer encoder for both policy and value networks:
- **Transformer Encoder**: Configured with 6 layers and 4 attention heads, it effectively captures temporal features from the input market data.
- **Policy and Value Networks**: The extracted features are processed through fully connected layers to generate trade decisions.

### 3. Fine-Tuning for Optimal Performance
While experimenting with various hyperparameters, including learning rate, batch size, clip range, and gamma, it was found that the default PPO settings provided the best performance, offering a balance between computational efficiency and learning effectiveness.

### 4. Evaluation: Performance Comparison - Transformer+PPO vs. Simple Trading Blotter
The model's performance was rigorously evaluated in comparison to a simple trading blotter:
- **Simple Trading Blotter**: This strategy resulted in a loss of approximately $3,400, as it lacked sophisticated pattern recognition and decision-making capabilities.
- **Transformer+PPO Model**: Demonstrated a significant improvement, generating a profit of around $45,000, turning around the losses experienced with the blotter.

### 5. Justification of the Approach
- **Relevance to Financial Data**: The Transformer is particularly suited for time-series analysis, making it highly relevant for financial market data.
- **Robustness and Flexibility**: The PPO ensures effective handling of the exploration-exploitation trade-off, crucial for adapting to changing market conditions.
- **Ease of Integration**: The custom Transformer-based network was seamlessly integrated into the PPO framework, making it easy to adopt within existing systems.
- **Scalability**: The approach is scalable and can be adapted to different market conditions, assets, or even other types of sequential data.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch or TensorFlow
- Required libraries: `gym`, `stable-baselines3`, `pandas`

### Usage
1. Set up your environment:
   ```python
   env = TradingEnvironment(your_ticker_data, daily_trading_limit)
   ```

2. Train the model:
   ```python
   model.learn(total_timesteps=10000)
   ```

3. Evaluate the model:
   ```python
   obs = env.reset()
   for _ in range(len(ticker_data)):
       action, _states = model.predict(obs)
       obs, rewards, done, info = env.step(action)
       if done:
           break
   env.render()
   ```

### Results
- The Transformer+PPO model generated a profit of around **$45,000**, compared to a **$3,400** loss using the simple trading blotter.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
