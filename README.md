### Repository of the paper:

## Hide-and-Seek: A template for explainable AI
### by Thanos Tagaris and Andreas Stafylopatis

Abstract:
> Recent breakthroughs in Deep Learning have led to a meteoric rise in the popularity of Neural Networks 
in the scientific world. Their adoption in various sectors of industry, however, has been much slower. 
This can mainly be attributed to the lack of transparency in Neural Networks and the lack of trust this 
entails to their end users. A prevailing disposition regarding interpretability in AI is the so-called 
*Fidelity-Interpretability tradeoff*, which implies that there is a tradeoff between the performance 
of a model and its interpretability. This study explores a framework for creating explainable Neural Networks, 
through the collaborative training of two networks, one trained for classification and one aiming at increasing 
its interpretability. The goal of this technique is to train a model with the highest possible degree of 
interpretability without sacrificing its performance. The experimental study involved image classification 
on 3 datasets: MNIST, CIFAR10 and an ImageNet derivative. Results prove that interpretable Convolutional 
Neural Networks could be trained on all the aforementioned datasets, without reduction in performance. 
Furthermore, evidence indicates that the aforementioned tradeoff is not as steep as related research might 
suggest. 

## Quick start:

## Requirements:

## Idea and Theory:

### Motivation:

The goal was to build a model that can not only classify images, but can indicate which part of the input it takes into account when making a prediction. This will be in the form of an binary mask which has the same shape as the input. In order to get the best possible explanations, we want the mask to *hide* the largest possible part of the input, i.e. keep only a small part of it.

The reasoning behind this is that the model's interpretability is tied with the portion of the input that the model manages to hide. If it hides a small part of the input then its interpretability is smaller. Intuitively, this can be seen in the example below:

> Model1: ***Although** the food **wasn't bad**, we **didn't** have a **good** time*.
> Model2: ***Although** the food **wasn't bad**, we **didn't** have a good time*.
> Model3: *Although the food wasn't bad, we **didn't** have a **good** time*.
>
> Out of these three models the third is the most interpretable, as it underlines only the necessary words that convey the sentiment of the sentence.

### Idea

The main idea is that we have two models, the **hider** and the **seeker**. These two are trained collaboratively to produce the best possible classification performance, while hiding the largest part of the input. 

The hider's goal is to produce a binary input mask that will be used to hide portions of the input. When trained properly, it should identify which parts of the input are unimportant and hide them. The seeker's goal is to classify the masked inputs. 

As stated previously, these two are trained simultaniously to under a loss function with two terms: the classification loss and the size of the input mask (i.e. how many ones we have in the mask). Both need to be minimized.

## Description of experiments and results

## Detailed Guide :

