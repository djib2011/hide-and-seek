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

### Hider

The hider is a model whose output has the same shape of its input but can produce only binary values. This is to be used as a input mask. For our experiments we used an autoencoder-like architecture. The output of this model needs to have a **binary threshold** to ensure this.

#### Deterministic vs Stochastic thresholding

We examined two types of thresholds: a deterministic one and a stochastic.

The first sets all values larger than 0.5 to 1 and the rest to 0. The second does the same based on a probability. For example if it takes an input of 0.7, it will output 1 with a probability of 0.7 and 0.3 otherwise.
The previous layer to both of these needs to be sigmoid-activated to normalize its output to [0, 1]. These units will be referred to as **Binray Deterministic Neurons (BDN)** or **Binary Stochastic Neurons (BSN)** depending on the thresholding technique they employ. 

An issue that arises is how are we going to backpropagate the gradients through the binary layer. For BDNs, one choice is to just ignore the threshold. This is a bit trickier in the case of BSNs, where we need to *estimate* the gradient. More details for this are available in the paper. We examine four different gradient estimators: *Streight-Through-v1 (ST1), Streight-Through-v2 (ST2), Slope-Annealing and REINFORCE*.

### Seeker

The seeker is a regular classifier, that attempts to classify the masked images. For this experiment we used a few CNN architectures.

### Loss

The loss function is comprised of two terms: a cross-entropy loss between the network's prediction and the actual label and a measure of the amount of information that the mask allows to pass through it. For the latter we used the sum of pixels in the mask.

These two terms are, by their nature in competition. By masking a large number of pixels, the classification performance should deteriorate. One question is how would we weight the two terms (i.e. which is more important and by what degree?).
This is regulated by a hyperparameter we call `alpha`.

One choice is to leave this in the hands of the user. By choosing to weight classification more than masking (i.e. `alpha > 0.5`), he is increasing the Fidelity of the classifier. By choosing the oposite he is electing to sacrifice Fidelity in favor of Interpretability.

If the user doesn't want to make this choice, we provide an **adaptive weighting** scheme. This starts off by purely assessing classification performance (i.e. `alpha=1`). When the model stagnates, it decreases the `alpha` by `0.05` and repeats this cycle until either `alpha=0.05` or the maximum number of epoxhs is reached. 

## Description of experiments and results:

Some research questions that arose:

1. Investigate the best initialization conditions:
    - Training from scratch (i.e. *full training*)
    - Pretrain the hider (i.e. *pretrained hider*)
    - Pretrain the seeker (i.e. *pretrained seeker*)
    - Pretrain both the hider and the seeker (i.e. *pretrained both*)
    
2. Investigate the best performing stochastic estimator:
    - Streight-Through-v1 (*ST1*)
    - Streight-Through-v1 (*ST2*)
    - Slope-Annealing (*SA*)
        - `Rate=0.1`
        - `Rate=0.5`
        - `Rate=1.0`
        - `Rate=10.0`
        - `Rate=100.0`
    - *REINFORCE*

3. Investigate the best thresholding technique:
    - Deterministic
    - Stochastic
    
4. Evaluate the HND model on the MNIST dataset.

5. Evaluate the HND model on the CIFAR10 dataset.

6. Evaluate the HND model on the CIFAR100 dataset.


## Detailed Guide :

