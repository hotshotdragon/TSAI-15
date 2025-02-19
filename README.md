# Smollm2-DeepSeek Architecture

[![PyPI version](https://badge.fury.io/py/smollm2-deepseek.svg)](https://badge.fury.io/py/smollm2-deepseek)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/smollm2-deepseek/badge/?version=latest)](https://smollm2-deepseek.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/username/smollm2-deepseek/workflows/CI/badge.svg)](https://github.com/username/smollm2-deepseek/actions)
[![Coverage Status](https://codecov.io/gh/username/smollm2-deepseek/branch/main/graph/badge.svg)](https://codecov.io/gh/username/smollm2-deepseek)
[![Downloads](https://static.pepy.tech/personalized-badge/smollm2-deepseek?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/smollm2-deepseek)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the implementation of Smollm2 architecture converted into DeepSeek architecture, incorporating Multi-head Linear Attention (MLHA) and Mixture of Experts (MoE).

## Architecture Overview

The model combines the efficient design principles of Smollm2 with DeepSeek's architectural innovations, particularly focusing on:

- **Multi-head Linear Attention (MLHA)**
  - Reduced computational complexity from O(n²) to O(n)
  - Improved memory efficiency while maintaining attention capabilities
  - Linear scaling with sequence length

- **Mixture of Experts (MoE)**
  - Dynamic routing of inputs to specialized expert networks
  - Increased model capacity without proportional computation costs
  - Enhanced representation learning through specialized expert pathways

- **Loss-less Load Balancing**
  - Optimal distribution of computational load across experts
  - Minimized expert capacity loss through advanced routing strategies
  - Improved training stability and resource utilization

## Key Features

1. **Efficient Attention Mechanism**
   - Linear attention implementation
   - Reduced memory footprint
   - Scalable to longer sequences

2. **Expert System**
   - Dynamic expert routing
   - Balanced expert utilization
   - Adaptive computation paths

## Model Architecture Details

```
Input
  │
  ├─ Embedding Layer
  │
  ├─ Transformer Blocks
  │   ├─ MLHA Layer
  │   │   ├─ Linear Projections
  │   │   └─ Efficient Attention Computation
  │   │
  │   ├─ MoE Layer
  │   │   ├─ Router Network
  │   │   ├─ Expert Networks
  │   │   
  │   │
  │   └─ Feed Forward Network
  │
  └─ Output Layer
```
## Model Output
<div align="center">
  <img src="Images/1.jpg" alt="Model Output 1" width="400"/>
</div>
<div align="center">
  <img src="Images/2.jpg" alt="Model Output 2" width="400"/>
</div>
<div align="center">
  <img src="Images/3.jpg" alt="Model Output 3" width="400"/>
</div>
<div align="center">
  <img src="Images/4.jpg" alt="Model Output 4" width="400"/>
</div>
<div align="center">
  <img src="Images/5.jpg" alt="Model Output 5" width="400"/>
</div>
<div align="center">
  <img src="Images/6.jpg" alt="Model Output 6" width="400"/>
</div>


## Implementation Notes

- The architecture maintains compatibility with standard transformer training approaches
- Implements custom attention patterns for improved efficiency
- Features specialized routing algorithms for MoE implementation
- Includes monitoring tools for expert utilization and load distribution


## References

1. Llama2 Architecture
2. DeepSeek V3 Technical Documentation
3. "Linear Attention Mechanisms: Theory and Implementation"
4. "Mixture of Experts with Load Balancing"
