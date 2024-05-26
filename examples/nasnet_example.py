import torch
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph


if __name__ == '__main__':
    warm_ups = 100
    iterations = 300

    import pretrainedmodels
    model_name = 'nasnetalarge'
    x = torch.randint(low=0, high=256, size=(1, 3, 331, 331), dtype=torch.float32).to(device="cuda:0")
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

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
    print("Memory used by PyTorch:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

    

    
