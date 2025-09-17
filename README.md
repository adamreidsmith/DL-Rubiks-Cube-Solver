# Solving Rubik’s Cubes with Deep Learning + Beam Search

---

## 📖 Overview
This project combines **deep learning** and **graph search** to solve the Rubik’s Cube. It extends and improves the methods developed in [this research](https://arxiv.org/pdf/2502.13266).
A neural network is trained to **predict the distance from any scrambled cube state to the solved state**, and this prediction is used as a **heuristic function** in a **beam search algorithm**.

Unlike traditional solvers with handcrafted heuristics, this approach learns the heuristic directly from data.

---

## ✨ Features
- ✅ Neural network heuristic trained from scrambled cube states
- ✅ Multiple deep learning architectures tested
- ✅ Beam search guided by the learned heuristic  
- ✅ Reproducible training + evaluation pipeline  
- ✅ GPU-accelerated training

---

## ⚙️ Technical Details

### 📊 Training Data
- Random walks performed outward from the solved state
- Mean distance-to-solved state recorded for each scrambled state reached
- 2 Billion training samples generated

### 📈 Models
- **MLP with Residual Connections**
    - Input: One-hot encoded cube state
    - Output: Predicted distance-to-solved
    - Loss: Smooth L1 Loss
    - Optimizer: AdamW
    - LR Schedule: Cosine Annealing
- **Embedding MLP**
    - Embeds each cube facet with a learnable embedding
    - Input: Categorized cube facets
    - Output: Predicted distance-to-solved
    - Loss: Smooth L1 Loss
    - Optimizer: AdamW
    - LR Schedule: Cosine Annealing
- **Transformer**
    - Transformer model for tabular data
    - Inputs embedded into tokens and passed through transformer blocks
    - Output: Predicted distance-to-solved
    - Loss: Smooth L1 Loss
    - Optimizer: AdamW
    - LR Schedule: Cosine Annealing
- **Rank Model**
    - Embeds cubes in a dense embedding space using residual SwiGLU blocks and an MLP Encoder
    - Margin Rank Loss relates distances in training data to cosine similarity in embedding space
    - Output: Distance-to-solved in embedding  space
    - Optimizer: AdamW
    - LR Schedule: Cosine Annealing

### 🔎 Search
- Algorithm: **Beam Search**  
- Beam width: 2^11 to 2^15
- Termination: Solved state reached *OR* 50 iterations

---

## 🏆 Results

### Dataset: 400 random scrambled states

| Model         | Beam Width | Avg. Solution Length | Success Rate |
| :------------ | :--------: | :------------------: | :----------: |
| Residual MLP  | $2^{14}$   | 19.49                | 100%         |
| Embedding MLP | $2^{14}$   | 19.30                | 100%         |
| Transformer   | $2^{14}$   | 20.02                | 99.5%        |
| Rank Model    | $2^{14}$   | 21.33                | 98.75%       |

### Dataset: [Kaggle Santa 2023 Challenge](https://www.kaggle.com/competitions/santa-2023)

| Method                                            | Beam Width | # Agents | Dataset Size | Avg. Solution Length | Success Rate |
| :------------------------------------------------ | :--------: | :------: | :----------: | :------------------: | :----------: |
| [Previous Best](https://arxiv.org/pdf/2502.13266) | $2^{24}$   | 1        | 8 Billion    | 19.51                | 100%         |
| **This Project**                                  | $2^{16}$   | 1        | 2 Billion    | **18.72**            | 100%         |
| **This Project**                                  | $2^{15}$   | 7        | 2 Billion    | **18.57**            | 100%         |

---

## 📜 License

[MIT License](LICENSE)
