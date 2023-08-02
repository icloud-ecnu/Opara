import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import numpy as np
from Opara import GraphCapturer

def run_torch_model(model, inputs, iterations, warm_ups):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    # model = torch.compile(model)
    # Warmup steps
    for _ in range(warm_ups):
        model(*inputs) # don't record time
    for i in range(iterations):
        # x_torch = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32).to(device="cuda:0")
        flush_cache()
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        y = model(*inputs)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    std = np.std(times)
    output_str = ('Time of native PyTorch:', str(np.mean(times)) + ' ms', "std: " + str(std))
    print('{:<30} {:<20} {:<20}'.format(*output_str))
    return y

def run_sequence_graph(symbolic_traced, inputs, iterations, warm_ups, start_index, end_index):
    with torch.no_grad():
        for i in range(warm_ups):
            symbolic_traced(*inputs)
            # my_interpreter.run(inputs)

    with torch.no_grad():
        g1 = torch.cuda.CUDAGraph()

        with torch.cuda.graph(g1):
            out_torch_graph_without_stream = symbolic_traced(*inputs)

        time_list = []
        torch.cuda.synchronize()
        for i in range(iterations):
            start = torch.cuda.Event(enable_timing=True)  # the times
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            g1.replay()
            end.record()
            end.synchronize()
            tim = start.elapsed_time(end)
            time_list.append(tim)
        # print(time_list[start_index:end_index])
        average_time = np.mean(time_list[start_index: end_index])
        std = np.std(time_list[start_index: end_index])
        output_str = ('Time of sequential CUDA Graph:', str(average_time) + ' ms', "std: " + str(std))
        print('{:<30} {:<20} {:<20}'.format(*output_str))
        return out_torch_graph_without_stream


def run_parallel_graph(Opara, inputs, iterations, warm_ups, start_index, end_index):
    time_list = []

    for i in range(iterations):
        flush_cache()
        torch.cuda._sleep(1_000_000)
        start = torch.cuda.Event(enable_timing=True)  # the times
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = Opara(*inputs)
        end.record()
        end.synchronize()
        tim = start.elapsed_time(end)
        time_list.append(tim)
    average_time = np.mean(time_list[start_index: end_index])
    std = np.std(time_list[start_index: end_index])
    output_str = ('Time of Opara:', str(average_time) + ' ms', "std: " + str(std) )
    print('{:<30} {:<20} {:<20}'.format(*output_str))
    return output

cache = torch.empty(int(4 * (1024 ** 2)), dtype=torch.int8, device='cuda')
def flush_cache():
    cache.zero_()

if __name__ == '__main__':
    warm_ups = 10
    iterations = 30
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32).to(device="cuda:0")
    model = torchvision.models.googlenet().eval()

    inputs = (x,)
    model = model.to(device="cuda:0").eval()
    y = run_torch_model(model, inputs, iterations, warm_ups)

    run_sequence_graph(model, inputs, iterations, warm_ups, 0, 300)

    Opara = GraphCapturer.capturer(inputs, model)
    output = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, 300)
    res = output[0]
    if res.dtype == torch.float16:
        res = res.float()
    print("output of PyTorch == output of Opara:", torch.allclose(y, res,rtol=1e-05,atol=1e-05,equal_nan =False), end='     ')
    print('Absolute difference:', torch.max(torch.abs(y.detach() - res.detach())))
