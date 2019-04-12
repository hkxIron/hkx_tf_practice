#coding:utf-8
"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
import code

class LSTM:
  
  @staticmethod
  def init(input_size, hidden_size, fancy_forget_bias_init = 3):
    """
    input_size:输入的向量维度 ,hidden_size:隐向量的维度
    Initialize parameters of the LSTM (both weights and biases in one matrix) 
    One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
    """
    # +1 for the biases, which will be the first row of WLSTM
    # W_gx*x(t)+W_gh*h(t-1)+b_g,即input_size+hidden_size+1
    # 4*hidden_size:输入门,输出门,忘记门,输入数据,input:10,hidden:4
    # 如果是单个门,维度为:15*4,而现在是3个门+gate,则 15*(4*4)=15*16,gate 是cell_candidate
    WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) / np.sqrt(input_size + hidden_size) #15*16
    WLSTM[0,:] = 0 # initialize biases to zero,15*16
    if fancy_forget_bias_init != 0:
      # forget gates get little bit negative bias initially to encourage them to be turned off
      # remember that due to Xavier initialization above, the raw output activations from gates before
      # nonlinearity are zero mean and on order of standard deviation ~1
      WLSTM[0,hidden_size:2*hidden_size] = fancy_forget_bias_init
    return WLSTM
  
  @staticmethod
  def forward(X, WLSTM, c0 = None, h0 = None):
    """
    X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
    """
    n,b,input_size = X.shape #n序列长度,b:batch长度,input_size:输入的x的维度,(1, 3, 10)
    d = WLSTM.shape[1]/4 #d:hidden size,WLSTM:15*16,d=4
    if c0 is None: c0 = np.zeros((b,d)) #c0:之前的细胞状态,3*4
    if h0 is None: h0 = np.zeros((b,d)) #h0:之前的隐藏状态,3*4
    
    # Perform the LSTM forward pass with X as the input
    xphpb = WLSTM.shape[0] # x plus h plus bias, lol, 即 W_gx*x(t)+W_gh*h(t-1)+b_g
    Hin = np.zeros((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM,1*3*15
    Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content), sequence_length*batch_size*hidden_size
    IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
    IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
    C = np.zeros((n, b, d)) # cell content
    Ct = np.zeros((n, b, d)) # tanh of cell content
    for t in range(n): #序列的长度,比如长度为5的序列
      # concat [x,h] as input to the LSTM
      prevh = Hout[t-1] if t > 0 else h0 # batch_size*hidden_size,3*4
      Hin[t,:,0] = 1 # bias
      Hin[t,:,1:input_size+1] = X[t] #3*10
      Hin[t,:,input_size+1:] = prevh #3*4
      # compute all gate activations. dots: (most work is this line)
      IFOG[t] = Hin[t].dot(WLSTM) #Hin[t]:(3*15), WLSTM:15*16 ->3*16,3即为3通道,3通道数据相互独立,但它们共享W参数
      # non-linearities
      IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates,对所有的门进行sigmoid变换,input,forget,output
      IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh,对gate进行tanh变换
      # compute the cell activation
      prevc = C[t-1] if t > 0 else c0
      C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc #input_gate*gate+forget_gate*c_prev -> 新细胞状态
      Ct[t] = np.tanh(C[t]) 
      #output_door*tanh(C_t)
      Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t] #输出门*新细胞状态

    cache = {}
    cache['WLSTM'] = WLSTM
    cache['Hout'] = Hout
    cache['IFOGf'] = IFOGf
    cache['IFOG'] = IFOG
    cache['C'] = C
    cache['Ct'] = Ct # 5*3*4,即 序列中每个元素输出后的序列的细胞状态
    cache['Hin'] = Hin #记录序列中每个元素输入的(bias,xt,ht),shape为(5,3,15)
    cache['c0'] = c0
    cache['h0'] = h0

    # return C[t], as well so we can continue LSTM with prev state init if needed
    return Hout, C[t], Hout[t], cache
  
  @staticmethod
  def backward(dHout_in, cache, dcn = None, dhn = None): 
    #dHout_in:5*3*4,序列中每次输出的h
    WLSTM = cache['WLSTM'] #15*16
    Hout = cache['Hout'] #5*3*4
    IFOGf = cache['IFOGf'] #5*3*16
    IFOG = cache['IFOG'] #5*3*16
    C = cache['C'] #5*3*4
    Ct = cache['Ct'] #5*3*4
    Hin = cache['Hin']
    c0 = cache['c0']
    h0 = cache['h0']
    n,b,d = Hout.shape
    input_size = WLSTM.shape[0] - d - 1 # -1 due to bias,
 
    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros((n,b,input_size))
    dh0 = np.zeros((b, d))
    dc0 = np.zeros((b, d))
    dHout = dHout_in.copy() # make a copy so we don't have any funny side effects,dHout即dHt
    if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
    if dhn is not None: dHout[n-1] += dhn.copy()
    for t in reversed(range(n)):#从t到t-1,t-2,...0
      tanhCt = Ct[t] #3*4,最后一次细胞状态
      dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]#输出门的梯度,tanhCt:3*4,dHout[t]:3*4,细胞状态*输出门,Ht=Ot*tanh(Ct),则 dOt=tanh(Ct)*dHt
      # backprop tanh non-linearity first then continue backprop
      dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t]) #注意,这里必须是+=, 因为在上一次(t+1)时刻时,dC[t]就被赋过值了
 
      if t > 0:
        dIFOGf[t,:,d:2*d] = C[t-1] * dC[t] #dft,即对forget_door的梯度
        dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
      else:
        dIFOGf[t,:,d:2*d] = c0 * dC[t] #dft
        dc0 = IFOGf[t,:,d:2*d] * dC[t] #
      dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t] #input_door,dit
      dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t] #dC_prev, IFOGF[t,:,:d] input_door
      
      # backprop activation functions
      dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:] #dC_prev
      y = IFOGf[t,:,:3*d]
      dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d] #input_door,forget_door,output_door
 
      # backprop matrix multiply
      dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t]) #由于每个序列内都有新的增量更新过来,所以也必须是加
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())
 
      # backprop the identity transforms into Hin
      dX[t] = dHin[t,:,1:input_size+1]  # bias+Xt
      if t > 0:
        dHout[t-1,:] += dHin[t,:,input_size+1:] #隐层更新,由于只更新一次,所以这里改成dHout[t-1,:]=也影响不大
      else:
        dh0 += dHin[t,:,input_size+1:]
 
    return dX, dWLSTM, dc0, dh0



# -------------------
# TEST CASES
# -------------------



def checkSequentialMatchesBatch():
  """ check LSTM I/O forward/backward interactions """

  n,b,d = (5, 3, 4) # sequence length:5, batch size:3, hidden size:4
  input_size = 10
  WLSTM = LSTM.init(input_size, d) # input size, hidden size,15*16
  X = np.random.randn(n,b,input_size) #seqLength*batch*inputSize,5*3*10
  h0 = np.random.randn(b,d) #之前的输出,3*4
  c0 = np.random.randn(b,d) #之前的细胞状态,3*4

  # sequential forward
  cprev = c0
  hprev = h0
  caches = [{} for t in range(n)]
  Hcat = np.zeros((n,b,d))
  for t in range(n):
    xt = X[t:t+1] #1*3*10,同于X[t:t+1,:,:],1*3*10
    _, cprev, hprev, cache = LSTM.forward(xt, WLSTM, cprev, hprev) #forward返回: Hout, C[t], Hout[t], cache
    caches[t] = cache
    Hcat[t] = hprev

  # sanity check: perform batch forward to check that we get the same thing
  H, _, _, batch_cache = LSTM.forward(X, WLSTM, c0, h0) # X:5*3*10,batch_cache keys:['Hout', 'C', 'h0', 'IFOG', 'WLSTM', 'IFOGf', 'c0', 'Hin', 'Ct']
  assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!' #检查两个向量在一定的误差范围之内,是否逐元素相同,H:5*3*4

  # eval loss
  wrand = np.random.randn(*Hcat.shape) # (5,3,4)
  loss = np.sum(Hcat * wrand) #(5,3,4).*(5,3,4),逐元素做点乘,即各time_step的loss求和
  dH = wrand

  # get the batched version gradients
  BdX, BdWLSTM, Bdc0, Bdh0 = LSTM.backward(dH, batch_cache)

  # now perform sequential backward
  dX = np.zeros_like(X)
  dWLSTM = np.zeros_like(WLSTM)
  dc0 = np.zeros_like(c0)
  dh0 = np.zeros_like(h0)
  dcnext = None
  dhnext = None
  for t in reversed(range(n)):
    dht = dH[t].reshape(1, b, d)
    dx, dWLSTMt, dcprev, dhprev = LSTM.backward(dht, caches[t], dcnext, dhnext)
    dhnext = dhprev
    dcnext = dcprev

    dWLSTM += dWLSTMt # accumulate LSTM gradient
    dX[t] = dx[0]
    if t == 0:
      dc0 = dcprev
      dh0 = dhprev

  # and make sure the gradients match
  print('Making sure batched version agrees with sequential version: (should all be True)')
  print(np.allclose(BdX, dX))
  print(np.allclose(BdWLSTM, dWLSTM))
  print(np.allclose(Bdc0, dc0))
  print(np.allclose(Bdh0, dh0))
  

def checkBatchGradient():
  """ check that the batch gradient is correct """

  # lets gradient check this beast
  n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d) # input size, hidden size
  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)

  # batch forward backward
  H, Ct, Ht, cache = LSTM.forward(X, WLSTM, c0, h0)
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
  dH = wrand
  dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, cache)

  def fwd():
    h,_,_,_ = LSTM.forward(X, WLSTM, c0, h0)
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-5
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [X, WLSTM, c0, h0]
  grads_analytic = [dX, dWLSTM, dc0, dh0]
  names = ['X', 'WLSTM', 'c0', 'h0']
  for j in range(len(tocheck)):
    mat = tocheck[j]
    dmat = grads_analytic[j]
    name = names[j]
    # gradcheck
    for i in range(mat.size):
      old_val = mat.flat[i]
      mat.flat[i] = old_val + delta
      loss0 = fwd()
      mat.flat[i] = old_val - delta
      loss1 = fwd()
      mat.flat[i] = old_val

      grad_analytic = dmat.flat[i]
      grad_numerical = (loss0 - loss1) / (2 * delta)

      if grad_numerical == 0 and grad_analytic == 0:
        rel_error = 0 # both are zero, OK.
        status = 'OK'
      elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
        rel_error = 0 # not enough precision to check this
        status = 'VAL SMALL WARNING'
      else:
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        status = 'OK'
        if rel_error > rel_error_thr_warning: status = 'WARNING'
        if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

      # print stats
      print('%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
            % (status, name, np.unravel_index(i, mat.shape), old_val, grad_analytic, grad_numerical, rel_error))


if __name__ == "__main__":

  checkSequentialMatchesBatch()
  raw_input('check OK, press key to continue to gradient check')
  checkBatchGradient()
  print('every line should start with OK. Have a nice day!')