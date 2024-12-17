## *Opara*

*Opara* is a lightweight and resource-aware DNN Operator parallel scheduling framework to accelerate the execution of DNN inference on GPUs. Specifically, *Opara* first employs CUDA Graph and CUDA Streams to automatically parallelize the execution of multiple DNN operators. It further leverages the resource requirements of DNN operators to judiciously adjust the operator launch order on GPUs to expedite DNN inference.

## System overview of *Opara*

*Opara* comprises four components including Model Profiler, Operator Launcher, Stream Allocator, and Graph Capturer. As illustrated in  the subsequent figure, *Opara* takes DNN models and input tensors (i.e., inference data) from users. According to the operator dependencies in the DAG of DNN models, the Stream Allocator first employs a stream allocation algorithm to determine which stream the operators should be allocated to. The Model Profiler then gathers the resource requirements of each operator during the model profiling process. With such resource requirements of operators, the Operator Launcher further employs a resource-aware operator launch algorithm to optimize the operator launch order on GPUs. Finally, the Graph Capturer generates a parallelized CUDA Graph by combining the stream allocation plan and operator launch order, thereby enabling efficient DNN inference on GPUs.
![overview](https://github.com/icloud-ecnu/Opara/blob/main/figures/overview.png?raw=true)

## Installation

```shell
git clone https://github.com/OparaSys/Opara.git
cd Opara
pip install -r requirements.txt
```

## Usage

The subsequent code snippet illustrates the utilization of *Opara* to expedite model inference. It requires the provision of your model and the corresponding input tensors. Then you can utilize the interface ```GraphCapturer.capturer(inputs, model)```, which returns a callable. Finally, feed the callable with  input tensors that serves as a parameter to yield the inference outcome.
```shell
import torch
import torchvision
from Opara import GraphCapturer

model = torchvision.models.googlenet().eval()
model = model.to(device="cuda:0")
x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32).to(device="cuda:0")
inputs = (x,)
# Submit DNN model and input tensors as two parameters to instantiate a model execution with parallel operator execution.
Opara = GraphCapturer.capturer(inputs, model)
output = Opara(*inputs)
```

## Example

We provide a Python script that measures the performance of native PyTorch, sequential CudaGraph, and *Opara*. Execute the following command to generate the corresponding output.
```shell
python examples/googlenet_example.py
```
output:
```shell
Time of native PyTorch:        3.587766415278117 ms std: 0.05031060025425075
Time of sequential CUDA Graph: 1.8679669356346131 ms std: 0.009087139973587288
STAGE:2023-08-28 19:19:41 49050:49050 ActivityProfilerController.cpp:311] Completed Stage: Warm Up
STAGE:2023-08-28 19:19:41 49050:49050 ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2023-08-28 19:19:41 49050:49050 ActivityProfilerController.cpp:321] Completed Stage: Post Processing
Time of Opara:                 1.11034135492642721 ms std: 0.0088255187530593
output of PyTorch == output of Opara: True     Absolute difference: tensor(0., device='cuda:0')
```

## Publication
### *Opara* Journal Version:
[1] Aodong Chen, Fei Xu, Li Han, Yuan Dong, Li Chen, Zhi Zhou, and Fangming Liu, "[Opara: Exploiting Operator Parallelism for Expediting DNN Inference on GPUs](https://ieeexplore.ieee.org/document/10707307)," IEEE Transactions on Computers, 2025, 71(1): 325-333. DOI: 10.1109/TC.2024.3475589.

Our paper has been published in IEEE Transactions on Computers, and we welcome any citations of our work as below.
```
@article{chen2025opara,
  title={Opara: Exploiting Operator Parallelism for Expediting DNN Inference on GPUs},
  author={Chen, Aodong and Xu, Fei and Han, Li and Dong, Yuan and Chen, Li and Zhou, Zhi and Liu, Fangming},
  journal={IEEE Transactions on Computers},
  volume={71},
  number={1},
  pages={325--333}
  year={2025},
  publisher={IEEE}
}
```


### *Opara* Conference Version:
[2] Aodong Chen, Fei Xu, Yuan Dong, Li Chen, Zhi Zhou, and Fangming Liu, “[Opara: Exploring Operator Parallelism for Expediting DNN Inference on GPUs](https://github.com/icloud-ecnu/Opara/blob/main/pdf/ccfsys-opara.pdf)," in: Proc. of CCFSys, Nanchang, China, August 4-5, 2023. (**DPCS Best Student Paper Award**)
