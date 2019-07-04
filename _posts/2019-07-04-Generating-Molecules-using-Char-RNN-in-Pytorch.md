---
layout: post
title:  "Generating Molecules using CharRNN "
date: 2019-06-27
comments: True
mathjax: True
---
<h2>Generating Molecules using CharRNN </h2>

Before you dig into details of Recurrent Neural networks, if you are a Beginner I suggest you to read about RNN.
Note: To go through the article, you must have basic knowledge of neural networks and how Pytorch (a deep learning library) works. 
You can refer the some articles to understand these concepts:
In this post ,I am implementing a RNN model with Pytorch to generate SMILES.

Now in this we will learn-
 <ol>
 <li>Why/what are Recurrent Neural Networks?</li>
 <li> Character level RNN model</li>
 <li>RNN for Molecules (SMILES) Generation</li>
 <li>Generating SMILES using RNN's</li>
 </ol>
 <b>Recurrent Neural Networks-</b>The idea behind RNN's is to make use of sequential information. 
 RNN's are used for sequential data, like audio or sentences, where the order of the data plays an important role.
 What makes Recurrent Networks so special? The core reason that recurrent nets are more exciting is that they allow us to operate over sequences of vectors:
   {% include image.html url="/assets/img/1rnn.png" description="Unrolled RNN Architecture" %}
   
 <b>Character-Level RNN Model:</b> Okay, so we have an idea about what RNNs are, why they are super exciting, and how they work. We’ll now ground this in a fun application: 
 We’ll train RNN character-level RNN models. That is, we’ll give the RNN a huge chunk of data(Smiles representation of molecules)and ask it to model the 
 probability distribution of the next character in the sequence given a sequence of previous characters. 
 This will then allow us to generate new smiles one character at a time.
 By the way, together with this post I am also releasing (code on Github:<a href="https://github.com/bayeslabs/genmol/tree/Sunita/genmol/CharRNN/">Visit this link</a>) that allows you to train char RNN model based on multi-layer LSTMs.
 
<b>RNN for Molecules (SMILES) Generation-</b> In this Post, we want to show that recurrent neural networks can be trained as generative models for molecular structures, 
 similar to statistical language models in natural language processing. 
 We demonstrate that the properties of the generated molecules correlate very well with the properties of the molecules used to train the model.
To connect chemistry with language, it is important to understand how molecules are represented. Usually, they are modeled by molecular graphs, also called Lewis structures in chemistry. In molecular graphs, atoms are labeled nodes. The edges are the bonds between atoms, which are labeled with the bond order (e.g., single, double, or triple).

However, in models for natural language processing, the input and output of the model are usually sequences of single letters, strings or words. We therefore employ the SMILES (Simplified Molecular Input Line Entry System) format are the type of chemical notation that helps us to represent molecules and easy to used by the computers. It is a simple string representation of molecules, which encodes molecular graphs compactly as human-readable strings. SMILES is a formal grammar which describes molecules with an alphabet of characters, for example c and C for aromatic and aliphatic carbon atoms, O for oxygen, and −, =, and # for single, double, and triple bonds (see Figure 2).To indicate rings, a number is introduced at the two atoms where the ring is closed. For example, benzene in aromatic SMILES notation would be c1ccccc1.
{%include image.html url="\assets\img\smiles.png description="Examples of molecule and It's SMILES representation. To correctly create smiles, the model has to learn long-term dependencies, for example, to close rings (indicated by numbers) and brackets." %}


<b>Generating SMILES using RNN's:</b>  I'll be showing you how I implemented my recurrent neural network in Pytorch. I trained it using the ChEMBL smiles Dataset ,which contains 2M smiles,and it is a manually curated database of bio-active drug-like molecules.

 <b> Part 1: Importing libraries and data preprocessing -</b> First, we import <b>pytorch</b>, the deep learning library we'll be using,also <b>import nn </b> (pytorch's neural network library) and <b>torch.nn.functional</b>, which includes non-linear functions like ReLu and sigmoid.
 
Let's load the Data file and name it as text

```python
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

with open('chembl_smiles.txt ', 'r') as f:
    text = f.read()
```

Then we'll create a dictionary out of all the characters and map them to an integer. This will allow us to convert our input characters to their respective integers (char2int) and vice versa (int2char).
```python
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
# Encode the text
encoded = np.array([char2int[ch] for ch in text])
```
Finally, we're going to convert all the integers into one-hot vectors.
```python
# Defining method to encode one hot labels
def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot
```
we will usually want to feed training data in batches to speed up the training process,so defining method to make mini-batches for training.
```python
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
```
<b>Part 2: Building the Model</b> 
First, we're going to check if we can train using the <b>GPU</b>, which will make the training process <b>much quicker</b>. If you don't have a GPU, be forewarned that it will take a much longer time to train. Check out Google Collaboratory or other cloud computing services!
