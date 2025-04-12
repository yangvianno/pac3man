# 🧠 Pac3man: Intelligent Multi-Agent Pac-Man AI

Welcome to **Pac3man**, an AI-powered extension of the classic Pac-Man game built on the UC Berkeley CS188 framework. This project introduces multiple intelligent agents using **heuristics**, **adversarial search**, and **reinforcement learning**, showcasing how AI can plan, adapt, and compete in dynamic environments.

---

## 📌 Features

- 🎯 **Heuristic Evaluation**  
  Score functions that account for food proximity, ghost distance, and scared states.

- 🧭 **Adversarial Agents**  
  Implemented **Minimax** and **Expectimax** agents that reason over future states in a multi-agent environment.

- 🤖 **Q-Learning Agent**  
  Reinforcement learning agent that updates Q-values over time based on environmental feedback.

- ⚡ **Optimized Decision Trees**  
  Reduced search complexity via effective **state pruning** and **Manhattan distance heuristics**.

- 📊 **Performance Tracking**  
  Win rates and behavioral evaluation on various layout scenarios.

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.6+
- No external libraries required (uses standard Python + custom framework)

### 🛠️ Installation

```bash
git clone https://github.com/yangvianno/pac3man.git
cd pac3man
```

### ▶️ Run an Agent

```bash
python3 multiagent/run.py -p MinimaxAgent -l mediumClassic
```

For Q-learning:

```bash
python3 reinforcement/run.py -a QLearningAgent -l smallGrid
```

To see more options:

```bash
python3 run.py -h
```

---

## 🧠 AI Agents Implemented

| Agent             | Type                        | Description                                                  |
|------------------|-----------------------------|--------------------------------------------------------------|
| `ReflexAgent`     | Heuristic                   | Greedy agent using evaluation function                       |
| `MinimaxAgent`    | Adversarial Search          | Chooses actions minimizing worst-case ghost outcomes         |
| `ExpectimaxAgent` | Probabilistic Adversarial   | Models ghosts as stochastic agents (chance nodes)            |
| `QLearningAgent`  | Reinforcement Learning      | Learns optimal policy using Q-value updates                  |

---

## 📂 Project Structure

```
pac3man/
├── multiagent/           # Minimax, Expectimax, Reflex agents
├── reinforcement/        # Q-learning, value iteration agents
├── search/               # DFS, BFS, UCS, A* search agents
├── util/                 # Helper functions
├── README.md             # Project documentation
```

---

## 📷 Sample Visualizations

Coming soon: GIFs and charts of agent behavior across classic layouts.

---

## ✍️ Author

**Alex Vo**  
📧 [vodanghongphat@gmail.com](mailto:vodanghongphat@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/vodanghongphat)  
🐙 [GitHub](https://github.com/yangvianno)

---

## 📜 Acknowledgements

This project is based on the open-source UC Berkeley [CS188 Pac-Man AI Projects](http://ai.berkeley.edu/project_overview.html).  
All enhancements and AI agent implementations were developed for academic exploration and AI engineering practice.

---
