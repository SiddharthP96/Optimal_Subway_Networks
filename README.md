
---

# Multiplex Network Optimization for Urban Transportation

## Overview

This repository delves into the optimization of urban transportation networks, focusing on the interplay between road (local) and subway (global) networks. We use simulated annealing and multi-greedy optimization to study optimal subway structures. On analyzing real-world population distributions, distinct patterns emerge: optimal subway networks with one or two central branches. Similar findings were observed for Atlanta and Boston. Furthermore, we probed real subway efficiencies in these cities, comparing them to multi-greedy solutions. Notably, as local speed (a proxy for congestion) increased, real subways deviated more from the optimal. 



### Model Description

The multiplex network, $\( \mathcal{G} \)$, comprises two distinct layers:

1. **Local Layer**: Represents the road network and is modeled as a triangular lattice. Only nodes at a graph distance less than a specified radius $\( R \)$ are included.
2. **Global Layer**: Symbolizes the subway network. It contains the same nodes as the local layer but only a subset of its edges.

For any pair of replica nodes $\( n \)$ and $\( m \)$ across these layers, we have the map $\( F(n) = m \)$. The weight of the edges in the local layer is consistently one, while the global layer edges have a weight \( \eta \leq 1 \). Replica nodes are interconnected via switching edges with weights $\( c \)$.

Our primary goal is to discover the optimal structure for the subway network (i.e., the global layer). The optimal setup minimizes $\( \tau \)$, the average shortest path length of the local layer nodes to the center. Depending on system parameters, the shortest path to the center may either use or bypass the global layer.

Moreover, we refine this model by incorporating the heterogeneity in population distributions. This addition provides a more realistic representation of actual cities. The optimization objective now becomes a weighted average of the local nodes' shortest path lengths, with weights determined by node population, $\( \mathcal{P} \)$.

To adapt this model for real cities, we superimpose the city landscape onto the local layer. The center node, $\( o \)$, is mapped to the city's heart, typically where the subway system's intersection lies. This area often corresponds to the downtown region. A city radius $\( R_c \)$ encircles the city center, encompassing all areas serviced by real subway lines. Relying on census tract-level population density data, we determine the population distribution $\( \mathcal{P} \)$ for the transformed lattice.

## Repository Files

- **init.py**: Initialization file for the project.
- **requirements.txt**: Contains all the Python dependencies required for running the scripts and notebooks.
- **Tutorial.ipynb**: A Jupyter notebook guide demonstrating how to utilize the provided codebase.
- **functions.py**: Contains utility functions used across the project.
- **network.py**: Core implementation of the multiplex network model.
- **simulated_annealing.py**: Houses the simulated annealing algorithm used for optimization.

## Installation

To set up and run the project:

1. Clone the repository.
2. Navigate to the repository's root directory.
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the `Tutorial.ipynb` notebook for a step-by-step guide.

## Contributors

- Siddharth Patwardhan
- Sirag Erkol
- Filippo Radicchi
- Santo Fortunato
- Marc Barthelemy

For further queries or contributions feel free to contact Siddharth Patwardhan at sidpatwa@iu.edu or siddharthpatwardhan1@gmail.com.
