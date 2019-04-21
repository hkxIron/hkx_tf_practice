import torch
import torchvision
import numpy as np
np.random.seed(0)
#tf.random.set_random_seed(0)
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
"""
下面的代码在torch 0.4版本中可以运行
"""


"""
h = X*w1
h_relu = relu(h)
y_pred = h_relu * w2
loss = (y_pred - y)**2
"""
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6
iter = 500
print_iter = 100
device = torch.device('cpu')

def nn_from_numpy():

    """
    PyTorch Tensors are just like numpy arrays, but they can run on GPU.
    """
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)
    print("nn from numpy")

    for t in range(iter):
        """
        
        h = X*w1
        h_relu = relu(h)
        y_pred = h_relu * w2
        loss = (y_pred - y)**2
        """
        h = x@w1
        h_relu = np.clip(h, a_min=0, a_max=None) # [N, D_in]
        y_pred = h_relu@w2
        loss = np.sum((y_pred - y)**2)
        if t%print_iter == 0:
            print("iter:{} loss:{}".format(t, loss))

        grad_y_pred = 2.0 * (y_pred - y) # [N, D_out]
        grad_w2 = h_relu.T@grad_y_pred # [H, D_out],
        grad_h_relu = grad_y_pred@w2.T # [N, D_in]
        # dL/dh= dL/d_h_relu*d_h_relu/dh
        # d_h_rele/dh= 1 if h>0
        #              0 if h<=0
        grad_h = np.copy(grad_h_relu)
        grad_h[h < 0] = 0
        grad_w1 = x.T@grad_h

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

nn_from_numpy()

def nn_from_torch_wich_manual_gradient():

    """
    PyTorch Tensors are just like numpy arrays, but they can run on GPU.
    """
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    w1 = torch.randn(D_in, H, device=device)
    w2 = torch.randn(H, D_out, device=device)
    print("nn from torch with manual gradient")

    for t in range(iter):
       """
       PyTorch Tensor API looks almost exactly like numpy!
       
       h = X*w1
       h_relu = relu(h)
       y_pred = h_relu * w2
       loss = (y_pred - y)**2
       """
       h = x.mm(w1)
       h_relu = h.clamp(min=0) # [N, D_in]
       y_pred = h_relu.mm(w2)
       loss = (y_pred - y).pow(2).sum()
       if t%print_iter == 0:
           print("iter:{} loss:{}".format(t, loss))

       grad_y_pred = 2.0 * (y_pred - y) # [N, D_out]
       grad_w2 = h_relu.t().mm(grad_y_pred) # [H, D_out],
       grad_h_relu = grad_y_pred.mm(w2.t()) # [N, D_in]
       # dL/dh= dL/d_h_relu*d_h_relu/dh
       # d_h_rele/dh= 1 if h>0
       #              0 if h<=0
       grad_h = grad_h_relu.clone()
       grad_h[h < 0] = 0
       grad_w1 = x.t().mm(grad_h)

       w1 -= learning_rate * grad_w1
       w2 -= learning_rate * grad_w2

nn_from_torch_wich_manual_gradient()


def nn_fromm_pytorch():
    device = torch.device('cpu')

    """
    We will not want gradients (of loss) with respect to data
    """
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    """
    Creating Tensors with requires_grad=True enables autograd
    """
    w1 = torch.randn(D_in, H, device=device, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, requires_grad=True)
    print("nn from torch")

    for t in range(iter):
        """
        Operations on Tensors with requires_grad=True cause PyTorch to build a computational graph
        
        Forward pass looks exactly the same as before, but we don’t need to track intermediate values - PyTorch keeps track of them for us in the graph
        """
        # 由于不需要手工梯度,因此只用一个变量
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        if t%print_iter == 0:
            print("iter:{} loss:{}".format(t, loss))

        """
        Compute gradient of loss with respect to w1 and w2
        
        想想caffe中的backforward()
        """
        loss.backward()

        """
        Make gradient step on weights, then zero them. Torch.no_grad means “don’t build a computational graph for this part”
        """
        with torch.no_grad():
            w1 -= learning_rate * w1.grad # 更新梯度
            w2 -= learning_rate * w2.grad

            """
            PyTorch methods that end in underscore modify the Tensor in-place; methods that don’t return a new Tensor
            """
            w1.grad.zero_()
            w2.grad.zero_()


nn_fromm_pytorch()


"""
Define your own autograd functions by writing forward and backward functions for Tensors

Very similar to modular layers in A2(caffe2?)! Use ctx object to “cache” values for the backward pass, just
like cache objects from A2. 

Define a helper function to make it easy to use the new function
"""
def auto_grad_function_pytorch():

    class MyReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_y):
            x, = ctx.saved_tensors
            grad_input = grad_y.clone()
            grad_input[x < 0] = 0
            return grad_input

    """
    In practice you almost never need to define new autograd functions!
    Only do it when you need custom backward. In this case we can just use a normal Python function
    """
    def my_relu(x):
        return MyReLU.apply(x)


    device = torch.device('cpu')

    """
    We will not want gradients (of loss) with respect to data
    """
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    """
    Creating Tensors with requires_grad=True enables autograd
    """
    w1 = torch.randn(D_in, H, device=device, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, requires_grad=True)
    print("auto grad function for pytorch")

    for t in range(iter):
        """
        Can use our new autograd function in the forward pass
        """
        y_pred = my_relu(x.mm(w1)).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        if t%print_iter == 0:
            print("iter:{} loss:{}".format(t, loss))

        """
        Compute gradient of loss with respect to w1 and w2
        """
        loss.backward()


        """
        Make gradient step on weights, then zero them. Torch.no_grad means “don’t build a computational graph for this part”
        """
        with torch.no_grad():
            w1 -= learning_rate*10000 * w1
            w2 -= learning_rate*10000 * w2

            """
            PyTorch methods that end in underscore modify the Tensor in-place; methods that don’t return a new Tensor
            """
            w1.grad.zero_()
            w2.grad.zero_()

auto_grad_function_pytorch()


def nn_with_high_level_apis():
    device = torch.device('cpu')
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    print("nn with high level apis ...")

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )

    for t in range(iter):
        """
        Forward pass: feed data to model, and compute loss
        """
        y_pred = model(x)
        """
        torch.nn.functional has useful helpers like loss functions
        
        Backward pass: compute gradient with respect to all model weights (they have requires_grad=True)
        """
        loss = torch.nn.functional.mse_loss(y_pred, y)

        if t%print_iter == 0:
            print("iter:{} loss:{}".format(t, loss))
        loss.backward()

        """
        Make gradient step on each model parameter (with gradients disabled)
        """
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate *10000* param.grad

        model.zero_grad()

nn_with_high_level_apis()

def nn_with_high_level_apis_adam():
    device = torch.device('cpu')
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    print("nn with adam ...")

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*100)
    for t in range(iter):
        """
        Forward pass: feed data to model, and compute loss
        """
        y_pred = model(x)
        """
        torch.nn.functional has useful helpers like loss functions
        
        Backward pass: compute gradient with respect to all model weights (they have requires_grad=True)
        """
        loss = torch.nn.functional.mse_loss(y_pred, y)

        if t%print_iter == 0:
            print("iter:{} loss:{}".format(t, loss))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

nn_with_high_level_apis_adam()


def use_new_modules():

    class TwoLayerNet(torch.nn.Module):
        """
        Initializer sets up two children (Modules can contain modules)
        """
        def __init__(self, D_in, H, D_out):
            super(TwoLayerNet, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)

        """
        Define forward pass using child modules.
        No need to define backward - autograd will handle it
        """
        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)

    print("nn with new modules ...")

    """
    construct and train an instance of our model
    """
    model = TwoLayerNet(D_in, H, D_out)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

    for t in range(iter):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)

        if t%print_iter == 0:
            print("iter:{} loss:{}".format(t, loss))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

use_new_modules()


def mix_and_custom_module_and_seq_container():

    class ParallelBlock(torch.nn.Module):
        def __init__(self, D_in, D_out):
            super(ParallelBlock, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, D_out)
            self.linear2 = torch.nn.Linear(D_in, D_out)

        def forward(self, x):
            h1 = self.linear1(x)
            h2 = self.linear2(x)
            return (h1 * h2).clamp(min=0)

    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)

    print("mix custom module and seq container ...")

    """
    Stack multiple instances of the component in a sequential
    """
    model = torch.nn.Sequential(
        ParallelBlock(D_in, H),
        ParallelBlock(H, H),
        torch.nn.Linear(H, D_out)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

    for t in range(iter):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)

        if t%print_iter == 0:
            print("iter:{} loss:{}".format(t, loss))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

mix_and_custom_module_and_seq_container()

#alexnet = torchvision.models.alexnet(pretrained=True)
#print("alexnet", alexnet)


def test_data_loader():
    from torch.utils.data import TensorDataset, DataLoader
    class TwoLayerNet(torch.nn.Module):
        """
        Initializer sets up two children (Modules can contain modules)
        """
        def __init__(self, D_in, H, D_out):
            super(TwoLayerNet, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)

        """
        Define forward pass using child modules.
        No need to define backward - autograd will handle it
        """
        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)

    loader = DataLoader(TensorDataset(x,y), batch_size=8)
    model = TwoLayerNet(D_in, H, D_out)

    print("test data loader...")

    """
    
    Building the graph and computing the graph happen at the same time.
    Seems inefficient, especially if we are building the same graph over and over again...
    """
    model = TwoLayerNet(D_in, H, D_out)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

    """
    Throw away the graph, backprop path, and rebuild it from scratch on every iteration
    """
    for t in range(iter):
        """
        Iterate over loader to form minibatches
        """
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_batch)

            if t%print_iter == 0:
                print("iter:{} loss:{}".format(t, loss))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

test_data_loader()


def static_graph():
    desc = """
    Alternative: Static graphs
    Step 1: Build computational graph
    describing our computation
    (including finding paths for
    backprop)
    Step 2: Reuse the same graph on
    every iteration
    
    graph = build_graph()
    for x,y in loader:
        run_graph(graph, x, y)
        
    """
    print(desc)
