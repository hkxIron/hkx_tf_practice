import numpy as np

Ni,Nk = 2, 4
Nj = 3

def matrix_equal(A, B):
    return np.sum(np.abs(A - B)) == 0

def matrix_multiply():
    A = np.arange(0, Ni*Nk).reshape((Ni, Nk))
    B = np.arange(0, Nk*Nj).reshape((Nk, Nj))

    C = np.einsum("ik,kj->ij", A, B)
    print("A:\n", A)
    print("B:\n", B)
    print("C:\n", C)

    #  8. Summation Indices
    C_manual = np.empty((Ni, Nj), dtype=np.int32)
    for i in range(Ni):
        for j in range(Nj):
            total = 0
            for k in range(Nk):
                total += A[i, k] * B[k, j]

            C_manual[i, j] = total
    assert matrix_equal(C, C_manual)

    print("C_manual:\n", C_manual)

def matrix_diagonal_extraction():
    d = np.empty((Ni), dtype=np.int32)
    M = np.arange(0, Ni*Ni).reshape((Ni, Ni))
    print("M:\n", M)
    for i in range(Ni):
        total = 0
        total += M[i, i]
        d[i] = total
    print("d:\n", d)
    d_diag = np.diagonal(M)
    print("d_diag:\n", d_diag)
    d_esin = np.einsum("ii->i", M)
    d_esin = np.einsum("ii->i ", M) # 是否有空格,都是一样的
    d_esin = np.einsum("ii-> i", M)
    print("d_esin:\n", d_esin)
    assert matrix_equal(d_esin, d_diag)


def matrix_trace():
    Tr = 0  # Scalar! Has dimension 0 and no indexes
    total = 0
    M = np.arange(0, Ni*Ni).reshape((Ni, Ni))
    print("M:\n", M)
    for i in range(Ni):
        total += M[i, i]
    Tr = total

    trace_np = np.trace(M)
    trace_ein = np.einsum("ii->", M)
    print("trace_np:\n", trace_np)
    print("trace_ein:\n", trace_ein)

def quadratic_form():
    Ns, Nt = 3,3
    v = np.arange(0, Ns)
    x = 0
    total = 0
    M = np.arange(0, Ns*Nt).reshape((Ns, Nt))
    print("v:\n", v)
    print("M:\n", M)
    for s in range(Ns):
        for t in range(Nt):
            total += v[s] * M[s, t] * v[t]
    qua_manual = total
    qua_np = v@M@v
    qua_ein = np.einsum("s,st,t->", v, M, v)
    print("qua_manual:\n", qua_manual)
    print("qua_np:\n", qua_np)
    print("qua_ein:\n", qua_ein)

def batch_outer_product():
    NB = 2
    ni = 3
    nj = 4
    P = np.arange(0, NB*ni).reshape((NB, ni))
    Q = np.arange(0, NB*nj).reshape((NB, nj))

    R = np.zeros((NB, ni, nj), dtype=np.int32)
    for B in range(NB):
        for i in range(Ni):
            for j in range(Nj):
                total = 0
                total += P[B, i] * Q[B, j]
                R[B, i, j] = total

    outer_manual = R
    outer_ein = np.einsum("Bi,Bj->Bij", P, Q)
    print("P:\n", P)
    print("Q:\n", Q)
    print("outer_manual:\n", outer_manual)
    print("outer_ein:\n", outer_ein)
    #assert matrix_equal(outer_manual, outer_ein)

def softmax(x):
    x_ = np.exp(x)
    return x_/np.sum(x_)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def mlp():
    # 15: MLP Backprop done easily (stochastic version).
    #     h = sigmoid(Wx + b)
    #     y = softmax(Vh + c)
    Ni = 784
    Nh = 500
    No = 10

    np.random.seed(0)

    W = np.random.normal(size=(Nh, Ni))  # Nh x Ni
    b = np.random.normal(size=(Nh,))  # Nh
    V = np.random.normal(size=(No, Nh))  # No x Nh
    c = np.random.normal(size=(No,))  # No

    # Load x and t...
    #x, t = train_set[k]
    x, t = np.random.random((Ni)), np.random.random((No))

    # With a judicious, consistent choice of index labels, we can
    # express fprop() and bprop() extremely tersely; No thought
    # needs to be given about the details of shoehorning matrices
    # into np.dot(), such as the exact argument order and the
    # required transpositions.
    #
    # Let
    #
    #     'i' be the input  dimension label.
    #     'h' be the hidden dimension label.
    #     'o' be the output dimension label.
    #
    # Then

    # Fprop
    ha = np.einsum("hi, i -> h", W, x) + b
    h = sigmoid(ha)
    ya = np.einsum("oh, h -> o", V, h) + c
    y = softmax(ya)

    # Bprop
    dLdya = y - t
    dLdV = np.einsum("h , o -> oh", h, dLdya)
    dLdc = dLdya
    dLdh = np.einsum("oh, o -> h ", V, dLdya)
    dLdha = dLdh * dsigmoid(ha)
    dLdW = np.einsum("i,  h -> hi", x, dLdha)
    dLdb = dLdha

    print("dLdb shape:", dLdb.shape)

def batch_mlp():
    # 16: MLP Backprop done easily (batch version).
    #     But we want to exploit hardware with a batch version!
    #     This is trivially implemented with simple additions
    #     to np.einsum's format string, in addition to the usual
    #     averaging logic required when handling batches. We
    #     implement even that logic with einsum for demonstration
    #     and elegance purposes.
    batch_size = 128
    Ni = 784
    Nh = 500
    No = 10

    np.random.seed(0)

    W = np.random.normal(size=(Nh, Ni))  # Nh x Ni
    b = np.random.normal(size=(Nh,))  # Nh
    V = np.random.normal(size=(No, Nh))  # No x Nh
    c = np.random.normal(size=(No,))  # No

    x, t = np.random.random((batch_size, Ni)), np.random.random((batch_size, No))

    # Let
    #     'B' be the batch  dimension label.
    #     'i' be the input  dimension label.
    #     'h' be the hidden dimension label.
    #     'o' be the output dimension label.
    #
    # Then

    # Fprop
    ha = np.einsum("hi, Bi -> Bh", W, x) + b # 空格并没有什么作用
    h = sigmoid(ha)
    ya = np.einsum("oh, Bh -> Bo", V, h) + c
    y = softmax(ya)

    # Bprop
    dLdya = y - t
    dLdV = np.einsum("Bh, Bo -> oh", h, dLdya) / batch_size
    dLdc = np.einsum("Bo     -> o ", dLdya) / batch_size
    dLdh = np.einsum("oh, Bo -> Bh", V, dLdya)
    dLdha = dLdh * dsigmoid(ha)
    dLdW = np.einsum("Bi, Bh -> hi", x, dLdha) / batch_size
    dLdb = np.einsum("Bh     -> h ", dLdha) / batch_size
    print("dLdb shape:", dLdb.shape)

matrix_multiply()
matrix_diagonal_extraction()
matrix_trace()
quadratic_form()
batch_outer_product()
mlp()
batch_mlp()
