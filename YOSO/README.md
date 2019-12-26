# OVERVIEW
DNN/Accelerator co-design has shown great potential in improving QoR and performance. Typical approaches separate the design flow into two-stage: (1) designing an application- specific DNN model with the highest accuracy; (2) building an accelerator considering the DNN specific characteristics. Though significant efforts have been dedicated to the improvement of DNN accuracy; it may fail in promising the highest composite score which combines the goals of accuracy and other hardware-related constraints (e.g., latency, energy efficiency) when building a specific neural network-based system. In this work, we present a singlestage automated framework, YOSO, aiming to generate the optimal solution of software-and-hardware that flexibly balances between the goal of accuracy, power, and QoS. YOSO jointly searches in the combined DNN and accelerator design spaces, which achieves a better composite score when facing a multi-objective design goal. As the search space is vast and it is costly to directly evaluate the accuracy and performance of the DNN and hardware architecture in design space search, we propose a cost-effective method to measure the accuracy and performance of solutions under consideration quickly. Compared with the two-stage method on the baseline systolic array accelerator and state-of-the-art dataset, we achieve 1.42x~2.29x energy reduction or 1.79x~3.07x latency reduction at the same level of precision, for different user-specified energy and latency optimization constraints, respectively, and the whole search procedure can be finished within 12 hours on single-card GPU. 

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

