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



