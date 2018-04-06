mnist_train_small.csv contains 30,000 pixel representations of hand-written digits. Each row contains information for a single digit. The first column has a value between 0 and 9 to indicate the digit. The remaining columns are 784 pixels corresponding to a 28x28 grid rendering of the hand-written digit.

commonDIrection function extracts the most common direction of variance across all data points

dataProjection takes in the 784 pixel data points from a single digit xi and the unit vector component uq and returns the coefficient zqi

componentRemoval function takes in a data vector xi , a component uq, and the corresponding co-efficient zqi and returns the updated data vector xi with the component uq removed. 

findComponents function takes in a matrix containing all the hand- written image data (30000 digits x 784 pixels) and a positive integer n specifying how many components to learn.



