<h1 align="center">
  <i><font color="#4A7EBB">Fewshot Learning in Image Classification</font></i>
</h1>

<p align="">
  <b>Sepehr Ghamari<sup>*</sup>, Arman Bakhtiari<sup>*</sup></b><br>
  <b><sup>*</sup>Electrical and Computer Engineering Department, Concordia University</b>
</p>

This repository contains the implementation of various **Few-Shot Learning (FSL)** techniques on the CIFAR-10 dataset under two challenges:
- 🔒 **Challenge 1**: No use of pre-trained models
- ✅ **Challenge 2**: Use of pre-trained models allowed

The project evaluates classical and state-of-the-art methods including:
- MAML
- ProtoNet
- Cross-Attention Network (CAN)
- AdaBoost + ResNet
- Self-Optimal-Transport (SOT) Feature Transform


## ⚙️ Methods Implemented

### 🧬 MAML
- Meta-learns an initialization that can adapt to new tasks quickly
- Backbone: Simple CNN, ResNet18 (non-pretrained)
- Optimizer: SGD | LR = 0.001

### 🧬 ProtoNet
- Computes class prototypes using embedding vectors
- Backbone: Simple CNN, ResNet18 (non-pretrained)
- Optimizer: Adam | LR = 0.001

### 🧬 Cross-Attention Network (CAN)
- Adds cross-attention to emphasize object regions
- Backbone: CNN + Attention + Classifier
- Optimizer: Adam | LR = 0.001

### 🧬 AdaBoost + ResNet
- AdaBoost classifier built on top of extracted ResNet features
- Weak learner: Decision Tree
- Backbones: ResNet18, ResNet50
- Hyperparams: `n_estimators=50`, `learning_rate=1.0`

### 🧬 Self-Optimal-Transport (SOT)
- Implements an optimal transport-based feature transform to refine embeddings
- Methods: ProtoSOT and PTMapSOT
- Backbones: ResNet12 and WRN-28-10
- Sinkhorn Iterations: 10
- Entropy Regularization (λ): 0.1
- Optimizer: Adam | Best LR = 0.0002


## 🔬 Implementation Details

- **Datasets**: CIFAR-10, converted into 2-way 5-shot format
    - 60,000 32×32 color images across 10 classes
    - **Training**: 2 classes, 25 samples/class
    - **Testing**: 200 samples/class
    - **Few-shot setup**: 2-way 5-shot, 20-query
    - **SOT Integration**: The transport matrix is computed via Sinkhorn iterations using cosine similarity between features
- **Feature Process**:
  - Feature vectors → Cosine Similarity Matrix → Cost Matrix → Sinkhorn → Output Feature Set
- **Backbones**: Models tested with ResNet12 and WRN-28-10, both initialized from scratch and with DropBlock regularization for generalization
- **Training & Testing**: 5 different seeds, average accuracy and std. dev. reported


## 🔄 Methodology

The study follows a two-pronged evaluation strategy:

### Challenge 1: Training from scratch (no pretraining)
- All models trained on 50 images (2 classes × 25 samples)
- Evaluated on 400 unseen images (2 classes × 200)
- Emphasis on performance vs. GPU training time

### Challenge 2: Using pretrained models
- ResNet models pretrained on ImageNet used as feature extractors
- Classifiers trained on top: linear classifier and AdaBoost
- Measured test accuracy and efficiency

## 📊 Results

### 📍 Challenge 1 – *No Pre-training*

<div align="center">

| 🧪 Method      | 🏗️ Backbone   | 🎯 Accuracy (%)     | ⚡ GPU Time (min) |
|---------------|---------------|---------------------|-------------------|
| Simple CNN    | CNN           | 65.81 ± 5.87        | 1.98 ± 0.45       |
| MAML          | CNN           | 49 ± 0.0            | 14.15 ± 0.78      |
| MAML          | ResNet18      | 56 ± 1.0            | 18.76 ± 0.78      |
| ProtoNet      | CNN           | 66.25 ± 7.40        | 25.10 ± 0.78      |
| ProtoNet      | ResNet18      | 67.5 ± 7.5          | 401.42 ± 8.25     |
| CAN           | CNN           | 66 ± 12             | 1.82 ± 0.42       |
| **ProtoSOT**  | ResNet12      | **76 ± 0.3**        | 7.37 ± 0.31       |
| ProtoSOT      | WRN-28-10     | 68 ± 0.4            | 28.11 ± 0.11      |
| PTMapSOT      | WRN-28-10     | 70 ± 0.4            | 39.22 ± 0.62      |

</div>

---

### 📍 Challenge 2 – *With Pre-training*

<div align="center">

| 🧪 Method       | 🏗️ Backbone   | 🎯 Accuracy (%)     | ⚡ GPU Time (min) |
|----------------|---------------|---------------------|-------------------|
| Linear Class.  | ResNet18      | 92.5 ± 5.26         | 0.13 ± 0.01       |
| **Linear Class.** | **ResNet50** | **95.75 ± 3.58**    | 0.22 ± 0.01       |
| AdaBoost       | ResNet18      | 86.75 ± 6.06        | 0.18 ± 0.01       |
| AdaBoost       | ResNet50      | 87.44 ± 4.25        | 0.40 ± 0.22       |
| ProtoSOT       | ResNet12      | 81 ± 0.5            | 0.12 ± 0.3        |

</div>


The results clearly demonstrate that **Self-Optimal-Transport (SOT)** based methods outperform traditional meta-learning techniques in low-data regimes.

- In **Challenge 1** (*no pre-training*), `ProtoSOT + ResNet12` achieved the highest accuracy (**76%**) while maintaining reasonable GPU time.
- In **Challenge 2** (*with pre-training*), a simple linear classifier on top of a **pre-trained ResNet50** achieved the best overall performance (**95.75%**) with minimal training time.

These findings highlight the effectiveness of advanced feature transformation and the utility of pretrained backbones for efficient and accurate few-shot learning.


## 🚀 Getting Started

To explore the project:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/sepehrgh98/CIFAR10-FewShot-Classification.git

2. Run the Jupyter notebooks inside each project folder to reproduce results and visualize experiments.

## 📚 References

- Finn, C., Abbeel, P., & Levine, S. (2017).  
  [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

- Snell, J., Swersky, K., & Zemel, R. S. (2017).  
  [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)

- Hou, R., Chang, H., Ma, B., Shan, S., & Chen, X. (2019).  
  [Cross Attention Network for Few-shot Classification](https://arxiv.org/abs/1904.09482)

- Shalam, D., & Korman, S. (2022).  
  [The Self-Optimal-Transport Feature Transform](https://arxiv.org/abs/2206.04161)

- Wang, W., Zhang, L., Zhang, M., & Wang, Z. (2020).  
  [Few-shot Learning for Multi-class Classification Based on Nested Ensemble DSVM](https://doi.org/10.1016/j.adhoc.2019.102055)

- Parnami, A., & Lee, M. (2022).  
  *Learning from Few Examples: A Summary of Approaches to Few-Shot Learning*  
  [Google Scholar Search](https://www.google.com/search?q=Learning+from+Few+Examples:+A+Summary+of+Approaches+to+Few-Shot+Learning+Parnami+Lee+2022)


## 📬 Contact

If you have any questions or feedback, feel free to contact us at:

📧 sepehrghamri@gmail.com
📧 se_gham@encs.concordia.ca



