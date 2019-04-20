import torch

print("pytorch 导出静态图")

N, D_in, H, D_out = 64, 1000, 100, 10

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

dummy_input = torch.randn(N, D_in)
"""
After exporting to ONNX, can run the PyTorch model in Caffe2
"""
torch.onnx.export(model,
                  dummy_input,
                  'model.proto',
                  verbose=True) # 打印模型