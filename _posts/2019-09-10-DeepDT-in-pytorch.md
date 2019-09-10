---
layout: post
title:  "Way to predict Drug-target interaction binding affinity using Convolution Neural Network in Drug Discovery Pipeline"
date: 2019-09-10
comments: True
mathjax: True
---
In this blog, I am writing about use of deep learning architecture for identification of drug-target interactions (DTI) 
strength (binding affinity) using character based sequence model.

As the field of drug discovery expands with the discovery of new drugs, re-purposing of existing drugs and identification of novel 
interacting partners for approved drugs is also gaining interest, since identification of novel drug-target (DT) interactions is a
substantial part of the drug discovery process there are many existing model to predict the drug-target interaction based on binary 
classification but in that constrain is the data sets constitutes a major problem, since negative (non-binding ) information is hard 
to find there. This article is about an approach to predict the binding affinities of protein-ligand interactions with deep learning
models using only sequences (1D representations) of proteins and ligand. It is a CNN based deep learning model that employ chemical 
and biological textual sequence information to predict binding affinity 
<b>(Binding affinity provide information on the strength of the interaction between a drug-target (DT) pair)</b>.

<b> Data-sets:</b>
Model evaluation is done on KIBA data-set (kinase inhibitors bio-active data) in which kinase inhibitor bio-activities from different sources were combined.
The KIBA data set originally comprised 229 proteins and 2111 drugs but due to our resource constrains(i.e. for more data to perform it requires GPU environment but we are using CPU environment) we filtered the data-set based on length of ligands and proteins. The constrains of length of every ligand sequence is 50(characters) and for every protein sequence it is 600(characters), now our data-set contains 111 proteins and 810 ligands and corresponding affinity values.

<b>Data Input representation to the model: </b>
We used one-hot encoding that uses binary integers (0, 1) for the ligand (characters) and protein (characters) to represent inputs Since Both SMILES (Len<50) and protein sequences (Len<600) have varying lengths. Hence, in order to create an effective representation form (one-hot encoding), we decided to fixed maximum lengths of 50(characters) for SMILES and 600(characters) for protein sequences in our data-set.
Final one-hot representation of ligands and proteins having dimension of (62x50) and (25x600) respectively. Here 62 and 25 are unique characters from SMILES (ligand) and proteins respectively.
<b>For more details refer this article:</b>:<a> href="https://arxiv.org/abs/1801.10193"</a>

<b>Model Preparation:</b>
In this article we treated protein-ligand interaction prediction as a regression problem by aiming to predict the binding affinity scores and used deep learning architecture, Convolutional Neural Network (CNN).
For this problem set (i.e. for ligands and proteins) we build a CNN-based prediction model that comprises two separate CNN blocks, each of which aims to learn representations from SMILES strings and protein sequences separately. For each CNN block the number of filters is 16 in first and 32 in the second convolutional layer then followed by the max-pooling layer. The final features of the max-pooling layers were concatenated and fed into two FC layers with a dropout layer (rate=0.15), which we named as DeepDTA.
Ø Activation function used: Rectified Linear Unit (ReLU).
Ø Loss function used: Root mean squared error (RMSE).
<b>The proposed model that combines two CNN blocks is illustrated below:</b>
.......................

<b>Training:</b>
For training and loss calculation we used Adam optimizer and RMSE loss function respectively.
for more details visit this github link:-

<b>Conclusion:</b>
when two CNN-blocks that learn representation of proteins and drugs based on raw sequence data are used in conjunction with DeepDTA, the performance is significantly considerable.




