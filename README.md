# LeNet
Implement the LeNet using tensorflow to recognize handwritten number. Training with MNIST. 
Some modifications here
1. Training with MNIST set with image size 28 * 28. To match the size of LeNet, the first convolution layer applied padding.
2. Using Relu instead of Sigmod as activation function.
3. Applied dropout in the FC layer.

This net can get 99.1% correct rate on MNIST test set.
