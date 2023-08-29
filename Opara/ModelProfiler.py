import torch
from torch.fx import symbolic_trace
import torch.profiler
from torch.fx import Interpreter
import torch._dynamo.eval_frame
import os
    

def profile(symbolic_traced, inputs, path):

    interpreter = Interpreter(symbolic_traced)
    
    
    def trace_handler(p):
        p.export_chrome_trace(path)
    
    
    with torch.profiler.profile(
        on_trace_ready=trace_handler,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        # with_flops=True
    ) as p:
        for i in range(1):
            out_torch = interpreter.run(*inputs)
            # out_torch = model(*inputs)
            p.step()
    return 

    
