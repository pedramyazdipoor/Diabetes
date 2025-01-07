## Gestational Diabetes Diagnosis

This project is a full implementation of this [article](https://doi.org/10.1016/j.jocs.2017.07.015).
In this project I implemented a radial basis function neural network from scratch with python. 
The weights of this neural network is calculated through interpolation rather than using gradient descent algorithm.

![image](https://github.com/user-attachments/assets/c0acb15a-bae4-47d1-9174-f10d2c34ef95)

The neurons in this network can be represented by a cluster of data points. The article suggested to do clustering on our dataset using K-Means;
However, it is essential to know the label of each centroid because of interpolation process. So I used another algorithm called PAM and also tested its
modified versions to do clustering faster. In contrast to K-Means, the centroid is chosen out of data points and it really exists!!
