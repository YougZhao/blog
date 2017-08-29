---
layout: post
title: "MNIST数据集与Convolutional Neural Network"
author: "Yang ZHAO"
---

### MNIST数据集

MNIST是一个非常有名的数据集，对于神经网络的学习来说，它就是我们所需要做的第一个“Hello World”，这篇说明文件将联系Convolutional Neural Network与MNIST一起使用。使用一个CNN区分数字0~9。但是在之前，会对CNN的大致原理做一个说明。

### 卷积神经网络

卷积神经网络（Convolutional Neural Network）是现在机器学习非常火爆的一个领域，现在有大量的企业开始将神经网络用作服务的核心，可以将神经网络用于自动标注算法、图片搜索、商品推荐、个性化主页推送。

>CNN的适用性为对于有标签的数据的分类或是识别，它可以方便地将数据进行处理后学习到各个不同标签的内在模式，精确度也比较高。它主要是通过卷积，池化等一系列计算将输入事物的特点抽取出来，并在神经网络中通过反向传播参数自动调整权重，并对事物进行判断和分类。其实卷积神经网络是将人类无法描述出来的概念和逻辑通过模式和特征的抽取，教给机器进行学习，这个学习的过程就是一个权值的调整，并最后输出结果。图片是最适用于神经网络的学习的类别，对于数据含量较小的向量，用经典的分类器在速度上面可能会比神经网络好。因为CNN需要的计算资源非常大，所以在计算大量数据的时候，可能需要用到hadoop的分布式gpu计算，成本比较高。

我们需要三个基本的元素来定义一个基本的卷积网络

**1.卷积层**

在这一层中，假设我们有一个6\*6的图像。我们定义一个权值矩阵，用来从图像中提取一定的特征。

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_1.gif)

我们把权值初始化成一个3\*3的矩阵。这个权值现在应该与图像结合，所有的像素都被覆盖至少一次，从而来产生一个卷积化的输出。上述的429，是通过计算权值矩阵和输入图像的3\*3高亮部分以元素方式进行的乘积的值而得到的。

现在6\*6的图像转换成了4\*4的图像。想象一下权值矩阵就像用来刷墙的刷子。首先在水平方向上用这个刷子进行刷墙，然后再向下移，对下一行进行水平粉刷。当权值矩阵沿着图像移动的时候，像素值再一次被使用。实际上，这样可以使参数在卷积神经网络中被共享。

下面我们以一个真实图像为例。

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_2.png)

权值矩阵在图像里表现的像一个从原始图像矩阵中提取特定信息的过滤器。一个权值组合可能用来提取边缘（edge）信息，另一个可能是用来提取一个特定颜色，下一个就可能就是对不需要的噪点进行模糊化。

先对权值进行学习，然后损失函数可以被最小化，类似于多层感知机（MLP）。因此需要通过对参数进行学习来从原始图像中提取信息，从而来帮助网络进行正确的预测。当我们有多个卷积层的时候，初始层往往提取较多的一般特征，随着网络结构变得更深，权值矩阵提取的特征越来越复杂，并且越来越适用于眼前的问题。

_**步长（stride）** 和 **边界（padding）**的概念:_

像我们在上面看到的一样，过滤器或者说权值矩阵，在整个图像范围内一次移动一个像素。我们可以把它定义成一个超参数（hyperparameter），从而来表示我们想让权值矩阵在图像内如何移动。如果权值矩阵一次移动一个像素，我们称其步长为 1。下面我们看一下步长为 2 时的情况。

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_3.gif)

可以看见当我们增加步长值的时候，图像的规格持续变小。在输入图像四周填充 0 边界可以解决这个问题。我们也可以在高步长值的情况下在图像四周填加不只一层的 0 边界。

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_4.png)

我们可以看见在我们给图像填加一层 0 边界后，图像的原始形状是如何被保持的。由于输出图像和输入图像是大小相同的，所以这被称为same padding

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_5.gif)

这就是 same padding（意味着我们仅考虑输入图像的有效像素）。中间的 4*4 像素是相同的。这里我们已经利用边界保留了更多信息，并且也已经保留了图像的原大小。

_**多过滤** 与 **激活图**:_

需要记住的是权值的纵深维度（depth dimension）和输入图像的纵深维度是相同的。权值会延伸到输入图像的整个深度。因此，和一个单一权值矩阵进行卷积会产生一个单一纵深维度的卷积化输出。大多数情况下都不使用单一过滤器（权值矩阵），而是应用维度相同的多个过滤器。

每一个过滤器的输出被堆叠在一起，形成卷积图像的纵深维度。假设我们有一个 32\*32\*3 的输入。我们使用 5\*5\*3，带有 valid padding 的 10 个过滤器。输出的维度将会是 28\*28\*10。

如下图所示：

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_6.png)

激活图是卷积层的输出。

**2.池化层**

有时图像太大，我们需要减少训练参数的数量，它被要求在随后的卷积层之间周期性地引进池化层。池化的唯一目的是减少图像的空间大小。池化在每一个纵深维度上独自完成，因此图像的纵深保持不变。池化层的最常见形式是最大池化。

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_7.png)

在这里，我们把步幅定为 2，池化尺寸也为 2。最大化执行也应用在每个卷机输出的深度尺寸中。正如你所看到的，最大池化操作后，4\*4 卷积的输出变成了 2\*2。

让我们看看最大池化在真实图片中的效果如何。

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_8.png)

正如看到的，我们卷积了图像，并最大池化了它。最大池化图像仍然保留了汽车在街上的信息。如果你仔细观察的话，你会发现图像的尺寸已经减半。这可以很大程度上减少参数。同样，其他形式的池化也可以在系统中应用，如平均池化和 L2 规范池化。

_**输出维度**_

理解每个卷积层输入和输出的尺寸可能会有点难度。以下三点或许可以让你了解输出尺寸的问题。有三个超参数可以控制输出卷的大小。

1. 过滤器数量-输出卷的深度与过滤器的数量成正比。请记住该如何堆叠每个过滤器的输出以形成激活映射。激活图的深度等于过滤器的数量。
2. 步幅（Stride）-如果步幅是 1，那么我们处理图片的精细度就进入单像素级别了。更高的步幅意味着同时处理更多的像素，从而产生较小的输出量。
3. 零填充（zero padding）-这有助于我们保留输入图像的尺寸。如果添加了单零填充，则单步幅过滤器的运动会保持在原图尺寸。
我们可以应用一个简单的公式来计算输出尺寸。输出图像的空间尺寸可以计算为（[W-F+2P]/S）+1。在这里，W 是输入尺寸，F 是过滤器的尺寸，P 是填充数量，S 是步幅数字。假如我们有一张 32\*32\*3 的输入图像，我们使用 10 个尺寸为 3\*3\*3 的过滤器，单步幅和零填充。
那么 W=32，F=3，P=0，S=1。输出深度等于应用的滤波器的数量，即 10，输出尺寸大小为 ([32-3+0]/1)+1 = 30。因此输出尺寸是 30\*30\*10。

**3. 输出层**

在多层卷积和填充后，我们需要以类的形式输出。卷积和池化层只会提取特征，并减少原始图像带来的参数。然而，为了生成最终的输出，我们需要应用全连接层来生成一个等于我们需要的类的数量的输出。仅仅依靠卷积层是难以达到这个要求的。卷积层可以生成 3D 激活图，而我们只需要图像是否属于一个特定的类这样的内容。输出层具有类似分类交叉熵的损失函数，用于计算预测误差。一旦前向传播完成，反向传播就会开始更新权重与偏差，以减少误差和损失。

正如你所看到的，CNN 由不同的卷积层和池化层组成。让我们看看整个网络是什么样子：

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_9.png)

我们将输入图像传递到第一个卷积层中，卷积后以激活图形式输出。图片在卷积层中过滤后的特征会被输出，并传递下去。
每个过滤器都会给出不同的特征，以帮助进行正确的类预测。因为我们需要保证图像大小的一致，所以我们使用同样的填充（零填充），否则填充会被使用，因为它可以帮助减少特征的数量。随后加入池化层进一步减少参数的数量。
在预测最终提出前，数据会经过多个卷积和池化层的处理。卷积层会帮助提取特征，越深的卷积神经网络会提取越具体的特征，越浅的网络提取越浅显的特征。
如前所述，CNN 中的输出层是全连接层，其中来自其他层的输入在这里被平化和发送，以便将输出转换为网络所需的参数。随后输出层会产生输出，这些信息会互相比较排除错误。损失函数是全连接输出层计算的均方根损失。随后我们会计算梯度错误。错误会进行反向传播，以不断改进过滤器（权重）和偏差值。一个训练周期由单次正向和反向传递完成。

### 实例代码

接下来是一个CNN的实例。
首先将需要的library导入，其中就包括MINST_data，其中的one_hot代表将数据的标签用[0,0,0,1,0,0,0,0,0,0]的形式来代表3。

{% highlight python %}
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from PIL import Image
{% endhighlight %}

接下来是创建两个placeholder，因为我们的图片是28*28的大小，将这一张图片变为1维后就是784。因为数据量时不定的，所以我们先用None代替，run的时候再进行赋值。所以我们的x的数据就是[None，784]，因为之前我们提到了y的标签是用one_hot的形式表示的，所以y有10维，数据量不定，得到y的数据为[None，10]

{% highlight python %}
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
{% endhighlight %}

接下来是通过函数创建两个可以调整的variable，w其实就是我们的filter，这个filter 的size我们可以以参数shape自己传进去，让其自己随机生成n个5*5，标准差为0.1，通道数量为1的filter。此处的通道数量指的是图片的色彩通道，一般彩色图片有三个通道R,G,B，MNIST中的黑白图片只有一个通道。

{% highlight python %}
#这个函数是随机创建n个filter，传入的shape参数决定了这个filter的大小，通道数，和数量n
#此外还能自己决定这个filter的标准差。
def weight_variable(shape):
    initial = tf.truncated_normal(shape = shape, stddev=0.1)
    return tf.Variable(initial)

#这个函数是随机创建n个bias（小偏差）
#这个模型中的权重在初始化时应该加入少量的噪声来打破对称
#性以及避免0梯度。由于我们使用的是ReLU神经元，因此比较
#好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
{% endhighlight %}

创建两个卷积和池化的函数

{% highlight python %}
#进行卷积的函数，4个1分别为用多少个filter，filter高，filter宽，filter的通道数
#padding=“same”指的是卷积过后输出的图片还是和原来大小一样
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
                           #[batch, height, width, channels]

#进行池化，每2*2的块中，选出一个最大的，所以是max_pool
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
{% endhighlight %}

接下来就是直接设置flow了，我们可以看到打出的三个流程中的tensor的形状，其中的问号代表不定，这是由输入数据的个数决定的。

{% highlight python %}
#先创建32个5*5，通道数为1的filter
W_conv1 = weight_variable([5, 5, 1, 32])
#创建少量的噪声来避免0梯度
b_conv1 = bias_variable([32])
#将n个28\*28的单通道图片输入，并变成n\*784的矩阵
x_image = tf.reshape(x, [-1, 28, 28, 1])
print x_image
#进行第一次卷积操作，对每个图片进行每个filter的卷积，并加上噪声
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print h_conv1
#进行第一次池化操作
h_pool1 = max_pool_2x2(h_conv1)
print h_pool1
{% endhighlight %}

将第一次操作的结果放入第二层。，再进行一次操作，次filter有64个了。

{% highlight python %}
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print h_conv2
h_pool2 = max_pool_2x2(h_conv2)
print h_pool2
{% endhighlight %}

其中有一个ReLU function，是为了让梯度能够更好地传递

![pic]({{ site.url }}/assets/2017-08-28-MNIST数据集与Convolutional-Neural-Network_10.png)

进过这两层后，我们将输出放入全连接层。

{% highlight python %}

{% endhighlight %}