
### Introduction
DeepBurning is an end-to-end automatic neural network accelerator design tool for specialized learning tasks. It provides a unified deep learning acceleration solution to high-level application designers without dealing with the model training and hardware accelerator tuning. You can refer to DeepBurning homepage for more details.

DeepBurning mainly includes the following four parts:
- YOSO, search for the optimized neural network architecture and the NPU configuration
- Model-zoo, pre-compiled neural network instructions
README: Instruction.hex includes the compiled instructions of the neural network data-in.coe, weights.coe, bias.coe are the formatted data files of input, weight and bias. 
- Zynq-prj: Pre-built zynq project on ZC706 and MZ7100. 
README: We have the NPU core deployed on ZC706 and MZ7100 respectively. We have both the hardware project files and relevant linux booting files included. It is a good start to deploy NPU for your own application.

```
Zynq-prj
	-ZC706
		-hw-prj
		-boot
			-rootfs
			-kernel
			-driver
			-app-example
	-MZ7100
-hw-prj
		-boot
			-rootfs
			-kernel
			-driver
			-app-example
```

### NPU-IP: NPU ip core (netlist)
It is a general NPU core that supports almost all the main-stream neural network models. It can be further customized for specific learning tasks and run at higher speed and less resource overhead. More details about the IP can be found in xxx.
