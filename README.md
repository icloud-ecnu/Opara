# ${Opara}$

${Opara}$ is a lightweight and resource-aware DNN Operator parallel scheduling framework to accelerate the execution of DNN inference on GPUs. Specifically, ${Opara}$ first employs CUDA Graph and CUDA Streams to automatically parallelize the execution of multiple DNN operators. It further leverages the resource requirements of DNN operators to judiciously adjust the operator launch order on GPUs to expedite DNN inference.

# System overview of ${Opara}$

${Opara}$ comprises four components including Model Profiler, Operator Launcher, Stream Allocator, and Graph Capturer. As illustrated in  the subsequent figure, ${Opara}$ takes DNN models and input tensors (i.e., inference data) from users. According to the operator dependencies in the DAG of DNN models, the Stream Allocator first employs a stream allocation algorithm to determine which stream the operators should be allocated to. The Model Profiler then gathers the resource requirements of each operator during the model profiling process. With such resource requirements of operators, the Operator Launcher further employs a resource-aware operator launch algorithm to optimize the operator launch order on GPUs. Finally, the Graph Capturer generates a parallelized CUDA Graph by combing the stream allocation plan and operator launch order, thereby enabling efficient DNN inference on GPUs.
![overview](https://github.com/icloud-ecnu/Opara/blob/main/figures/overview.png?raw=true)

# Installation

```shell
git clone https://github.com/OparaSys/Opara.git
cd Opara
pip install -r requirements.txt
```

# Usage

The subsequent code snippet illustrates the utilization of ${Opara}$ to expedite model inference. It requires the provision of your model and the corresponding input tensors. Then you can utilize the interface ```GraphCapturer.capturer(inputs, model)```, which returns a callable. Finally, feed the callable with  input tensors that serves as a parameter to yield the inference outcome.
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

# Example

We provide a Python script that measures the performance of native PyTorch, sequential CudaGraph, and ${Opara}$. Execute the following command to generate the corresponding output.
```shell
python examples/googlenet_example.py
```
output:
```shell
Time of native PyTorch:        3.697766415278117 ms std: 0.3182025326972793
Time of sequential CUDA Graph: 1.9705877343813578 ms std: 0.135356647385814
STAGE:2023-06-21 10:50:27 38128:38128 ActivityProfilerController.cpp:311] Completed Stage: Warm Up
STAGE:2023-06-21 10:50:27 38128:38128 ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2023-06-21 10:50:27 38128:38128 ActivityProfilerController.cpp:321] Completed Stage: Post Processing
Time of Opara:                 1.161413383483887 ms std: 0.03592966765019866
```
