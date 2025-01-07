## Gestational Diabetes Diagnosis

This project is a full implementation of this [article](https://doi.org/10.1016/j.jocs.2017.07.015).\
In this project I implemented a radial basis function neural network from scratch with python. 

Available features are Number of pregnancies, Plasma glucose concentration, Diastolic blood pressure, Triceps skin fold thickness,
2-H serum insulin, Body mass index, Diabetes pedigree function and Age.

The weights of this neural network is calculated through interpolation rather than using gradient descent algorithm.

![image](https://github.com/user-attachments/assets/bff3169b-4961-49f2-8b98-84c12d2aa7b4)


The neurons in this network can be represented as clusters of data points. The article suggested to do clustering on our dataset using K-Means;
However, it is essential to know the label of each centroid because of the interpolation process. So I used another algorithm called PAM and also tested its
modified versions to do clustering faster. In contrast to K-Means, the centroid is chosen out of data points and it really exists!!
The [Faster PAM](https://arxiv.org/abs/1810.05691) algorithm was proved to be 93 times faster in comparison to regular PAM in my experiments which is awesome!!!

![image](https://github.com/user-attachments/assets/b903daaf-7fad-4273-806d-af8136c4069f)


The phi matrix was normalized like below in order to have more coverage over the multi-dimensional space:

![image](https://github.com/user-attachments/assets/e69c2909-b754-46ce-9bad-b82c86b0378a)

The interpolation process is done after clustering and calculation of phi matrix(and of course knowing Z which is the labels of the centroids):

![image](https://github.com/user-attachments/assets/d206a4bb-edde-4604-8240-10c2be0546de)

With this method weights are calculated all at once thanks to linear algebra.

### Results
I could get these results in the best scenario after applying PCA algorithm which are approximately same as the article:

accuracy---82.75\
Precision--81.32\
Recall-----78.4\
FPR--------8.97
