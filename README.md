# Wavelet Generative Adversarial Networks (WaveletGAN)
  This projects is under research. It is based on BEGAN to solve how to make a model learn to generate images of which different parts have different definition of images.

  This model's novel points: 1. Two discriminators "teach" one generator. 2. Wavelet filters are put in front of first layers of two discriminators respectively, so the discriminators' ability is limited and also they can "teach" the generator different knowledge. One "teach" generator to generate one high-defintion part of image and anther "teach" generator to generate anther low-defintion part of a image. 
  
# Current Results on CelebA

Generated faces' different parts have different definition of images. Middle part is clear than the rest. But there is a problem that eyes and noses are like similar among faces. The project needs be imporved further.

![1](https://github.com/GuangyuanHao/WaveletGAN/raw/master/results/samples.png)
