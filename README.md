# MNIST GAN Project

## Introduction

In this project I explore four diffrent  approaches to generate MNIST image with general adversial network.

1. Vanilla

2. Conditional

3. Deep Convolution

4. Conditional Deep Convolution

Projects is written on PyTorch framework. IDE used in project is SPYDER.

## Vanilla approach

This is the simplest way to building GAN - both discriminator and generator are based on dense layers, with one hidden layer, BCELoss and Adam optimizer.

! [100 epochs for vanilla GAN - Discriminator and Generator Based only on dense layer](https://github.com/KordianChi/MNIST_GAN/blob/main/results/vanilla_gan_result.gif)
