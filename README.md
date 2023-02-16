# MNIST GAN Project

## Introduction

In this project I explore four diffrent  approaches to generate MNIST image with general adversial network.

1. Vanilla

2. Conditional

3. Deep Convolution

4. Conditional Deep Convolution

Projects is written on PyTorch framework. IDE used in project is SPYDER. In all case latent space dimension is equal 128.

## Vanilla approach

This is the simplest way to building GAN - both discriminator and generator are based on dense layers, with one hidden layer, BCELoss and Adam optimizer.

![100 epochs for vanilla GAN - Discriminator and Generator Based only on dense layer](https://github.com/KordianChi/MNIST_GAN/blob/main/results/vanilla_gan_result.gif)

## Conditional approach

In this approach, discriminator and generator are based on dense layers, but with additional embedding layer. Embedding layer allows to take label info. How you can see conditional model convergence is stable.

![100 epochs for vanilla GAN - additional embedding layer](https://github.com/KordianChi/MNIST_GAN/blob/main/results/conditional_gan_result.gif)

## Deep convolution approach

Convolution layers are first choice for neural network working with images. Discriminator is based on free layers of 2d convolution layers, and block of dense layers for classifictation. Generator used transposed convolution layers.

![100 epochs for vanilla GAN - additional embedding layer](https://github.com/KordianChi/MNIST_GAN/blob/main/results/deep_conv_gan_result.gif)

## Conditional deep convolution approach

The last approach combines advantages conditional and deep convolution GAN. Convergence is stable and credible. Model is based on convolution layers with embedding layer.

![100 epochs for vanilla GAN - additional embedding layer](https://github.com/KordianChi/MNIST_GAN/blob/main/results/conditional_deep_conv_gan_result.gif)

## Summary

This project was for my self-education about advanced neural network architecture like GAN. Second it was my first big project with PyTorch framework.
