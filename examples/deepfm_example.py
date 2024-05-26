import torch
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph, run_torch_trt_model
from NCF import DeepFM


if __name__ == '__main__':
    warm_ups = 100
    iterations = 300

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cate_fea_nuniqs = [100*(i+1) for i in range(32)] 
    nume_fea_size = 16  
    model = DeepFM(cate_fea_nuniqs, nume_fea_size, emb_size=8, hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]).to(device)

    batch_size = 1
    X_sparse = torch.randint(0, 100, (batch_size, len(cate_fea_nuniqs))).to(device)  # 生成随机的类别特征输入
    X_dense = torch.rand(batch_size, nume_fea_size).to(device)  # 生成随机的数值特征输入

    inputs = (X_sparse, X_dense)

    model = model.to(device="cuda:0").eval()
    y = run_torch_model(model, inputs, iterations, warm_ups)
    run_sequence_graph(model, inputs, iterations, warm_ups, 0, 300)
    run_torch_trt_model(model, inputs, iterations, warm_ups)
    Opara = GraphCapturer.capturer(inputs, model)
    output = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, 300)
    res = output[0]
    if res.dtype == torch.float16:
        res = res.float()
    print("output of PyTorch == output of Opara:", torch.allclose(y, res,rtol=1e-05,atol=1e-05,equal_nan =False), end='     ')
    print('Absolute difference:', torch.max(torch.abs(y.detach() - res.detach())))
    print("Memory used by PyTorch:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")


    

    
