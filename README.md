# Handwriting generation

An implementation of Alex Graves paper on handwriting generation https://arxiv.org/pdf/1308.0850.pdf.

A first version was written using tensorflow (version 1.11.0) followed by a keras version (2.2.4), keras being more computational costly though.

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


This tensor of strokes is fed to the NN as the main input, the tensor of texts is fed as a helper to the learning process and the NN given the stroke at T = t will try to predict the stroke at T = t+1.


# First results

Following are a few random outputs by the testing model with 2 lstm layer, only fed with strokes of maximum 600 timesteps during 10 epochs.

![img1](https://user-images.githubusercontent.com/34350063/49277982-71805700-f4be-11e8-8255-9b6124fb7d46.png)

![img2](https://user-images.githubusercontent.com/34350063/49278104-c15f1e00-f4be-11e8-82ad-57f6443d78da.png)

![img3](https://user-images.githubusercontent.com/34350063/49278155-dfc51980-f4be-11e8-8d2a-a9c07feaa492.png)

# Error analysis

- The end of stroke probability is not learned, there are as much 1 than 0 in the fisrt columns of the outputs
- Investing on re training with a basic loss function to compare results
