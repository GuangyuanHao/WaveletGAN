# Wavelet Generative Adversarial Networks (WaveletGAN)
  This projects is under research. It is based on BEGAN to solve how to make a model learn to generate images of which different parts have different definition of images.

  This model's novel points: 
  1. Two discriminators "teach" one generator. 
  2. Low-pass wavelet filter and a mask, and anther mask are put in front of first layers of two discriminators respectively, so the discriminators' ability is limited and they can also "teach" the generator different knowledge. One "teaches" generator to generate one low-definition part of an image and anther "teaches" generator to generate anther high-definition part of an image. 
  
# Current Results on CelebA

Generated faces' different parts have different definition of images. Middle part is clear than the rest. But there are problems that eyes and noses are like similar among faces and the background is too dark. The project needs to be improved further.

![1](https://github.com/GuangyuanHao/WaveletGAN/raw/master/results/samples.png)
