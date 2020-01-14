# Handwriting generation

An implementation of Alex Graves paper on handwriting generation https://arxiv.org/pdf/1308.0850.pdf.

A first version was written using tensorflow (version 1.11.0) followed by pytorch (version 1.3.1).

Both codes are structures as follow: 
- part 1 : preparing the data to feed the neural network
- part 2 : defining the model structure and training it with the prepared data
- part 3 : using the trained parameters to generate handwriting strokes.
                                     
                                     
# Data structure

Two types of data are fed to the NN. 


A corpus of sentences:
  - Transformed into a one hot tensor of shape (m, U, character_number).
  - 'm' is the number of sentences, 'U' the length of the longest sentence and 'character_number' the number of unique character.
  - For 'm = 0', it is the first sentence with each one of his character as a one hot vector. 
  - If the length of the sentence is less than 'U', the sentence is padded with vector of zeros until its length reaches 'U'.

Each sentence is paired with an array of shape (T,3):
  - Representing the sentence as a handwritten sentence. 
  - Each line represents a stroke, first column with value 0 or 1 represents whether the stroke is contiguous or not. 
  - Second and third columns represent the coordinates of the strokes relatively to the previous stroke. 
  - All strokes are regrouped as a tensor of shape (m, T, 3) with 'T' the number of stroke of the longest handwritten sentence. 
  - All other array of stroke are padded with vector of zeros until their number of strokes reaches T.
  - Values are normalized with mean = 0 and std = 1.


This tensor of strokes is fed to the NN as the main input, the tensor of texts is fed as a helper to the learning process and the NN given the stroke at T = t will try to predict the stroke at T = t+1.


# First results

Following are a few random outputs by the testing model (in the tensorflow folder) with 2 lstm layer, fed with strokes of maximum 600 timesteps during 10 epochs. 

![asd](https://user-images.githubusercontent.com/34350063/49361900-1f3a7280-f718-11e8-9ab2-3d94b305f044.png)

![as](https://user-images.githubusercontent.com/34350063/49361918-28c3da80-f718-11e8-9cb8-84cc956c6937.png)

![asq](https://user-images.githubusercontent.com/34350063/49361931-311c1580-f718-11e8-9588-77a97134267b.png)


# Error analysis

- Most of the generated examples have strokes going from left to right which is a good point.
- Investing on increasing the number of training examples (currently with 2500 examples) to check results.

# Recent results with pytorch
- A random stroke generation result
![Screenshot from 2019-12-15 17-57-46](https://user-images.githubusercontent.com/34350063/70866013-83ffee00-1f64-11ea-9f74-1b0d1098fda0.png)
