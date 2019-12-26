# OVERVIEW
This python tool allows you to explore the most energy-efficient neural netwrok (NN) topology for approximate computing, and supports user specified constraints. We apply MLP as the neural network and tranform the topology as a hyperparameter. This allows to apply [SMAC](https://www.cs.ubc.ca/labs/beta/Projects/SMAC/v2.10.03/quickstart.html) to solve the hyperparameter optimization problem. We use [nn_dataflow](https://github.com/stanford-mast/nn_dataflow) to evaluate the neural network runtime and energy consumption on an Eyeriss-style NN accelerator.  


This project is written in python2.7 and not yet Python 3 compatible.


# Installation

This tool tool needs SMAC to perform the search procedure and needs nn_dataflow for evaluation. 
The SMAC installation refers to [SMAC](https://www.cs.ubc.ca/labs/beta/Projects/SMAC/v2.10.03/quickstart.html). After installation, please add the path to your system environment.
The nn_dataflow installation refers to [nn_dataflow](https://github.com/stanford-mast/nn_dataflow). It's worth noting we use nn-dataflow (v1.5).

# USAGE
There are some parameters that need to be specified by the user. 
1) ** benchmark ** the benchmark name you want to test 
2) ** error_bound ** error bound for a specifical benchmark

Example for the 'fiexed topology selection' method:
    
    python fixed_topology_selection.py --benchmark fft --error_bound 0.001

For the SMAC method, you can set the Alpha Beta Gamma parameters in the `train-scenario.txt` to guide the SMAC search results towards the accuracy or energy-efficient. you also need to specify the benchmark name and error bound in the `train-scenario.txt`.Run the SMAC method :

       bash run_SMAC.sh



# License

This program is free software: you can redistribute it and/or modify
it under the terms of the 3-clause BSD license (please see the LICENSE file).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the 3-clause BSD license 
along with this program (see LICENSE file). 
If not, see <https://opensource.org/licenses/BSD-3-Clause>.

