# 关于tensorflow的一些函数

## 一、tf.flags

```python
import tensorflow as tf
FLAGS = tf.flags.FLAGS # flags是一个文件：flags.py，用于处理命令行参数的解析工作

# 调用tf.flags内部的DEFINE_类函数来定制解析规则
tf.flags.DEFINE_string('name','default','name of the model')
tf.flags.DEFINE_integer('n_seqs',100,'number of seqs in one batch')

# FLAGS是一个对象，保存了解析后的命令行参数
FLAGS = flags.FLAGS

def main(_):
    FLAGS.para_name #调用命令行输入的参数
    
# 使用这种方式保证了，如果此文件被其它文件import的时候，不会执行main中的代码
if __name__ == "__main__": 
    tf.app.run() # 解析命令行参数，调用main函数 main(sys.argv)命令行输入的参数
```

```
# 命令行调用，不传参数的话，会使用默认值
~/ python script.py --para_name_1=name --para_name_2=name2
```



## 二、Gradients

tf中有一个计算梯度的函数：```tf.gradients(ys,xs)```。其中，```xs```中的```x```必须要与```ys```相关，不相关的话，会报错。

```python
#wrong
import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

res = tf.matmul(w1, [[2],[1]]) # res只与w1相关

grads = tf.gradients(res,[w1,w2])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    re = sess.run(grads) # 报错：TypeError: Fetch argument None has invalid type
    print(re) 
```

`tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)`

Constructs symbolic partial derivatives of `ys` w.r.t. x in `xs`.

`ys` and `xs` are each a `Tensor` or a list of tensors. `grad_ys` is a list of `Tensor`, holding the gradients received by the `ys`. The list must be the same length as `ys`.

```python
import tensorflow as tf

w1 = tf.get_variable('w1', shape=[3])
w2 = tf.get_variable('w2', shape=[3])

w3 = tf.get_variable('w3', shape=[3])
w4 = tf.get_variable('w4', shape=[3])

z1 = w1 + w2+ w3
z2 = w3 + w4

grads = tf.gradients([z1, z2], [w1, w2, w3, w4], grad_ys=[tf.convert_to_tensor([2.,2.,3.]),
                                                          tf.convert_to_tensor([3.,2.,4.])])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(grads))

# 输出
#[array([ 2.,  2.,  3.],dtype=float32),
# array([ 2.,  2.,  3.], dtype=float32), 
# array([ 5.,  4.,  7.], dtype=float32), 
# array([ 3.,  2.,  4.], dtype=float32)]
```



**阻挡节点`BP`的梯度**：一个`节点`被 `stop`之后，这个节点上的梯度，就无法再向前`BP`了。由于`w1`变量的梯度只能来自`a`节点，所以，计算梯度返回的是`None`。

```python
import tensorflow as tf

w1 = tf.Variable(2.0)
w2 = tf.Variable(2.0)

a = tf.multiply(w1, 3.0)
a_stoped = tf.stop_gradient(a)

# b=w1*3.0*w2
b = tf.multiply(a_stoped, w2)
gradients = tf.gradients(b, xs=[w1, w2])
print(gradients)
# 输出
#[None, <tf.Tensor 'gradients/Mul_1_grad/Reshape_1:0' shape=() dtype=float32>]
```

**高阶导数的实现**

```python
import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant(1.)
    b = tf.pow(a, 2)
    grad = tf.gradients(ys=b, xs=a) # 一阶导
    print(grad[0])
    grad_2 = tf.gradients(ys=grad[0], xs=a) # 二阶导
    grad_3 = tf.gradients(ys=grad_2[0], xs=a) # 三阶导
    print(grad_3)

with tf.Session() as sess:
    print(sess.run(grad_3))
```

**Note: 有些 op，tf 没有实现其高阶导的计算，例如 tf.add …, 如果计算了一个没有实现 高阶导的 op的高阶导， gradients 会返回 None。**

