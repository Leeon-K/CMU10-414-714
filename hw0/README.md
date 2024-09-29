# Homework 0
Public repository and stub/testing code for Homework 0 of 10-714.
## 需要实现softmax regression
- [ ] 实现num_add，parse_mnist：

    遇到问题: parse_mnist时归一化，python 先对arr.astype('float32')；再/255 就过了，如果不.astype('float32')，默认是fp64，有误差

- [x] 实现nn_epoch

- [ ] 实现softmax_regression_epoch_cpp 中间遇到nan 的问题；

    前面几个数据不是nan 中间变成nan；发现是 inplace进行 matmul，Z公用一块内存，但没有给 Z 置零，初始化的问题；
    精度微小差异 +- 0.002；发现是地址取值 问题：常量指针不能改变赋值，但编译器没报错