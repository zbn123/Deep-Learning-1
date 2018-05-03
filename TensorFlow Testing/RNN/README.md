# RNN实践

## 一、初识RNNCell

RNNCell是一个抽象类，通常我们是通过它的两个子类BasicRNNCell和BasicLSTMCell来搭建相应的网络。

- 每个RNNCell都有一个call方法，使用方法是：

```python
(output,netx_state) = call(input,state) # 得到下一状态
```

![](https://static.leiphone.com/uploads/new/article/740_740/201709/59a9264ecd98b.jpg?imageMogr2/format/jpg/quality/90)

每调用一次``call()``方法，就相当于在时间上“推进了一步”，这就是RNNCell的基本功能。

- 除了``call()``方法之外，RNNCell还有两个类属性十分重要：

  - state_size # 隐层大小
  - output_size # 输出大小

  将一个batch送入模型计算，设输入的数据shape为(batch_size,input_size)，那么计算时得到的隐层状态就是(batch_size,state_size)，输出就是(batch_size,output_size)。

- zero_state(batch_size,np.float32) 使用此方法，能够得到一个shape为(batch_size,state_size)的全0初始状态



## 二、初识BasicRNNCell

```python
import tensorflow as tf
import numpy as np

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # 指定state_size为128
inputs = tf.placeholder(np.float32,shape(32,100)) # 指定输入的(batch_size,input_size)
h0 = cell.zero_state(32,np.float32) # 通过zero_state得到一个全0的初始状态(batch_size,state_size)
output,h1 = cell.call(inputs,h0)
print(hl.shape) #(32,128)
```



## 三、初识BasicLSTMCell

BasicLSTMCell包含两个隐层状态：h和c。对应的隐层就是一个Tuple。每个都是(batch_size,state_size)的形状。

```python
import tensorflow as tf
import numpy as np
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.palceholder(np.float,shape(32,100))
h0 = lstm_cell.zero_state(32,np.float32) # 通过zero_state得到一个全0的初始状态(batch_size,state_size)
output,h1 = lstm_cell.call(inputs,h0)
print(h1.h) # (32,128)
print(h1.c) # (32,128)
```



## 四、用tf.nn.dyamic_rnn实现多次调用call()

``tf.nn.dyamic_rnn``可以通过{h0,x1,x2,...,xn}直接得到{h1,h2,...,hn}。

输入的数据shape=(batch_size,time_steps,input_size)，其中time_steps表示序列本身的长度，比如在char RNN中，长度为10的句子对应的time_steps就等于10。最后的input_size就表示输入数据单个序列序列单个时间维度生固有的长度。另外我们定义好了一个RNNCell，调用该RNNCell的call函数time_steps次，对应的代码就是：

```python
'''
inputs: shape=(batch_size,time_steps,input_size)
cell: RNNCell
initial_state: shape=(batch_size,state_size)
'''
outputs,state = tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)
# shape(batch_size,time_steps,cell.output_size)
```



## 五、用MultiRNNCell堆叠RNNCell

在TensorFlow中可以使用```tf.nn.rnn_cell.MultiRNNCell```函数堆叠多层RNN。

```python
import tensorflow as tf
import numpy as np

def get_a_cell():
	return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range (3)]) # 3层RNN
# state_size 是每层128
inputs = tf.placeholder(np.float32,shape=(32,100))
h0 = cell.zero_state(32,np.float32)
output, h1 = cell.call(inputs,h0)
print(h1) # tuple中含有三个32*128的向量
```

也可以使用tf.nn.dynamic_rnn来一次运行多步。



## 六、关于output

先来看段BasicRNNCell的call函数源码：

```python
def call(self, inputs, state):
   """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
   output = self._activation(_linear([inputs, state], self._num_units, True))
   return output, output
```

**这句“return output, output”说明在BasicRNNCell中，output其实和隐状态的值是一样的。因此，我们还需要额外对输出定义新的变换，才能得到图中真正的输出y。**由于output和隐状态是一回事，所以在BasicRNNCell中，state_size永远等于output_size。

再来看一下BasicLSTMCell的call函数源码：

```python
new_c = (
   c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))

new_h = self._activation(new_c) * sigmoid(o)

if self._state_is_tuple:
 new_state = LSTMStateTuple(new_c, new_h)

else:
 new_state = array_ops.concat([new_c, new_h], 1)

return new_h, new_state
```

我们只需要关注self._state_is_tuple == True的情况，因为self._state_is_tuple == False的情况将在未来被弃用。返回的隐状态是new_c和new_h的组合，而output就是单独的new_h。如果我们处理的是分类问题，那么我们还需要对new_h添加单独的**Softmax层**才能得到最后的分类概率输出。



### 参考文献

[1] TensorFlow中RNN实现的正确打开方式