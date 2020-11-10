# 介绍 recall 知识

import paddle 
import numpy 

# 将Recall类实例化

recall_metric = paddle.fluid.metrics.Recall()

# 先生成我们的标签

labels = numpy.array([[1], [1]]).astype(numpy.int32)

# 再生成我们的预测值

predict = numpy.array([[0.7], [0.1]]).astype(numpy.float32)

# TP表示预测为正类（Positive），实际是正类 （True)，如第一个标签和预测值的行为

# FN表示预测为反类（Negative），实际是正类 (False)，如第二个标签和预测值的行为

recall_metric.update(predict, labels)

# recall 的计算公式为 TP / （TP+FN）,在这个例子，TP和FN均为1

recall_1 = recall_metric.eval()

# 根据公式，recall_1 = 1 / (1 + 1) = 0.5

recall_1

# 新建一个标签

labels2 = numpy.array([[1], [0], [0]]).astype(numpy.int32)

# 新建一个预测值

predict2 = numpy.array([[0.7], [0.6], [0.8]]).astype(numpy.float32)

# 预测值全都预测为正类，而标签只有一个为正类，因此 TP = 1

# 由于没有预测为反类的，所以根据定义，FN = 0

# 更新recall值

recall_metric.update(predict2, labels2)

# 计算Recall值

recall_2 = recall_metric.eval()

# 注意recall值是一个更新的过程，因此前面两次的TP总和为2，FN总和为1，根据公式，TP / (TP+FN) = 0.66

recall_2


