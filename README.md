# Final-Project
We are interested in bird image classification problems. The input of our algorithm is a bird image. Then we will consider using image classification neural networks like ResNet or RNN to classify which species the bird is.
# Dataset
Data set of 315 bird species.45980 training images, 1575 test images(5 images per species) and 1575 validation images (5 images per species.Each figure has 224 × 224 resolution. For preprocessing, we want to use PCA to decrease the dimensions of each data sample so that we can make data more centralized and decrease noise. In addition, if our model meet underfittting problem, we use shifted graphs to augmenting the total size of data. Ideally, we will do the unsupervised task by dropping labels and implement clustering on the data set.

## 1.1 Must accomplish
1. feature engineering of input image
We have processed and structured PCA which is used to reduce dimensions for original data set.
2. basic CNN model of classification
We have built basic CNN model for test data set.
3. training codes
Parameter tuning codes need to be debugged.
