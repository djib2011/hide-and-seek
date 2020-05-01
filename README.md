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

To run one of the preset experiments (e.g. the *mnist* one, as defined in the `config` file):

```
python collaborative_training.py --config mnist
```

The above command will train an HNS model from scratch on the MNIST dataset, without a baseline, and store the results and logs under `<...>/mnist/hns/default`. The baseline is required for measuring *Fidelity*, *FIR* and *FII*. To obtain a baseline and measure those metrics, we need to first train the *Seeker* (i.e. the classifier).

```
python pretrain_seeker.py --config mnist 
python baseline.py --config mnist
python collaborative_training.py --config mnist --baseline results/mnist/seeker/default/baseline.txt 
```

The first command pretrains the *Seeker*, evaluates the *Seeker* and stores the baseline performance and the third trains the HNS model. Since the baseline was passed as an argument, the logs will also depict all *Fidelity*-based metrics. The path to the baselien was the default one.

To pretrain the components of the HNS model:

```
python pretrain_seeker.py --config mnist
python pretrain_hider.py --config mnist 
python collaborative_training.py --config mnist --hider_weights weights/mnist/hider/default/best_weights.h5 --seeker_weights weights/mnist/seeker/default/best_weights.h5
```

The first command pretrains the *Seeker*, the second pretrains the *Hider* and the last trains the HNS model with a pretrained *Hider* and *Seeker*. 

To change parameters of the HNS model:

```
python collaborative_training.py --config mnist --stochastic --sa --rate 0.1 --alpha 0.7
```

The above command uses a stochastic threshold, with a *slope-annealing* gradient estimator (with a rate of 0.1) and a constant loss-regulator of 0.7 (i.e. no adaptive weighting).


## Requirements:

The development and initial testing was conducted on a Ubuntu 18.04 computer with a 6GB GeForce GTX 1060 graphics card, using Python 3.6.9. Experiments were conducted on a Ubuntu 16.04 computer with two graphics cards: a 12Gb GeForce GTX 1080 Ti and a 12Gb Titan Xp, using Python 3.5.2.

The library stack was kept consistent between the two setups and can be seen in the `requirements.txt`. Most notably, the HNS framework requires TensorFlow 2.0 and above (it was developed using the pre-alpha release of tf2).

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
    
4. Evaluate the HNS model on the MNIST dataset.

5. Evaluate the HNS model on the CIFAR10 dataset.

6. Evaluate the HNS model on the CIFAR100 dataset.

7. Evaluate the HNS model on the Fashion-MNIST dataset.

## Detailed Guide :

The detailed guide will consist of four parts. The first will focus on a description of the top-level modules and what each does. The remaining three will focus on 3 ways for running each module: preset experiments, custom experiments through the CLI and using the modules independently.

### Description of Top-Level Modules

There are five top-level modules which can be used for training and evaluating the HnS model on any given dataset.

- `pretrain_hider.py`: This module, as its name implies, is meant to train a *Hider* from scratch.
- `pretrain_seeker.py`: This module, likewise, is meant to train a *Seeker* from scratch. **Note**: This step is necessary for generating a baseline, which is required to measure the Fidelity, FIR and FII of an HnS model.
- `baseline.py`: This module generates the baseline from a pretrained Seeker. **Note**: This requires a pre-trained seeker and is required to measure the Fidelity, FIR and FII of an HnS model.
- `collaborative_training.py`: This is the main module. It is used to train an HnS model, either from scratch or from a pretrained Hider/Seeker. **Note**: In order to measure the Fidelity, FIR and FII the baseline must first be generated.
- `evaluation.py`: This module evaluates a trained HnS model.

Additionally we'll provide a description of the top-level directories: 

- `analysis/`: Contains Jupyter Notebooks analyzing the experiments
- `utils/`: Contains several modules necessary for building, training and evaluaing HnS models.
- `networks/`: Contain modules with the network architecures we used.
- `logs/`: Store log files after running the experiments.
- `results/`: Stores the results from `baseline.py` and `evaluation.py`.
- `weights/`: Stores the weights of the models during and after training.

The latter three follow the following directory structure:

```
logs
├── config
│   ├── model_type
│   │   ├── identifier
│   │   │   ├── batch
│   │   │   │   └── events.out.file
│   │   │   └── epoch
│   │   │       └── events.out.file
...
```

For example:

```
logs
├── cifar10
│   ├── hider
│   │   ├── default
│   │   │   ├── batch
│   │   │   │   └── events.out.tfevents.1565254564.pinkfloyd.deep.islab.ntua.gr.19339.317.v2
│   │   │   └── epoch
│   │   │       └── events.out.tfevents.1565254564.pinkfloyd.deep.islab.ntua.gr.19339.325.v2
|   |   ...
│   ├── hns
│   │   ├── deterministic
│   │   │   ├── full_training_10
│   │   │   │   ├── 1
│   │   │   │   │   ├── batch
│   │   │   │   │   │   └── events.out.tfevents.1581436099.pinkfloyd.deep.islab.ntua.gr.25563.613.v2
│   │   │   │   │   └── epoch
│   │   │   │   │       └── events.out.tfevents.1581436099.pinkfloyd.deep.islab.ntua.gr.25563.621.v2
│   |   |   ...
|   |   ├── stochastic
|   |   |   ├── st1
|   |   |   |   ├── full_training_10
|   |   |   |   |   ├── 1
|   |   ...
│   ├── seeker
        └── final
            ├── batch
            │   └── events.out.tfevents.1568023158.pinkfloyd.deep.islab.ntua.gr.3237.172.v2
            └── epoch
                └── events.out.tfevents.1568023158.pinkfloyd.deep.islab.ntua.gr.3237.180.v2
```

### Preset experiments

The easiest way to run experiments is to preset their parameters in the `config` file. Some of those parameters include the location of the data, the size and number of the images, the batch size, the number of classes etc. Not all parameters are mandatory. An example entry is the following:

```
[example]
    data_dir = /path/to/data/dir                      # directory where data is located
    hider_weights = /path/to/hider/weights.h5         # path to a valid "hider" model weights file
    seeker_weights = /path/to/seeker/weights.h5       # path to a valid "seeker" model weights file
    image_size = 256                                  # desired image dimensions: images will be resized to (256, 256)
    channels = 3                                      # number of channels (3 for RGB, 1 for grayscale)
    train_images = 10000                              # number of images in the training set (optional but recommended)
    test_images = 5000                                # number of images in the test set (optional but recommended)
    num_classes = 13                                  # number of classes (optional but recommended)
    max_epochs = 13                                   # maximum number of epochs to train the model
    batch_size = 64                                   # what batch size to use
    gpu = 1                                           # which gpu to use to train the model (for multi-gpu environments)
    model = hns_large                                 # select size of model to use, 'small' and 'large' available
```

By adding the *example* configuration in the `config` file, you can call this through the CLI:

```
python collaborative_training.py --config example
```

instead of 

```
python collaborative_training.py --data_dir /path/to/data/dir \
                                 --hider_weights /path/to/hider/weights.h5 \
                                 --seeker_weights /path/to/seeker/weights.h5 \
                                 --image_size 256 \
                                 ...
```

A more detailed description of the parameters is given in the next section. 

### Running Custom Experiments

This way uses the full power of the CLI, however all relevant parameters need to be specified during this step. These can be viewed by adding the argument `-h` or `--help` at the end of any script. For example:

```
python collaborative_training.py --help

  -h, --help            show this help message and exit
  --identifier IDENTIFIER
                        Name of the current experiment, will be used to name
                        the folders containing the logs and the weights
  --config CONFIG       Name of a valid configuration from "config.txt"
  --num_trainings NUM_TRAININGS
                        How many times to train the model
  --stochastic          Select if you want to use Binary Stochastic Neurons,
                        instead of Deterministic ones.
  --estimator ESTIMATOR
                        Name of the gradient estimator. only relevant for
                        stochastic neurons
  --rate RATE           Slope increase rate of Slope-Annealing estimator (only
                        relevant for this estimator). How much the slope
                        increases per epoch. E.g. "0.5" means that at the end
                        of the first epoch the slope will be 50{'container':
                        <argparse._ArgumentGroup object at 0x7fe610f1cd30>,
                        'metavar': None, 'dest': 'rate', 'required': False,
                        'type': 'float', 'default': None, 'choices': None,
                        'nargs': None, 'const': None, 'help': 'Slope increase
                        rate of Slope-Annealing estimator (only relevant for
                        this estimator). How much the slope increases per
                        epoch. E.g. "0.5" means that at the end of the first
                        epoch the slope will be 50% larger than what it
                        started.', 'option_strings': ['--rate'], 'prog':
                        'collaborative_training.py'}rger than what it started.
  --monitor MONITOR     What loss to monitor: "classification" or "total".Only
                        relevant for adaptive loss weighting.
  --patience PATIENCE   How many iterations to check for a significant
                        changein classification loss before reducing a. Only
                        relevant for adaptive loss weighting.
  --alpha ALPHA         Value for alpha. Should be between 0 and 1. Higher
                        values cause the "classification loss" to contribute
                        more to the total loss, while lower values cause the
                        "mask loss" to contribute more. If not specified, an
                        adaptive loss weighting will occur.
  --data_dir DATA_DIR   Directory where training data is located
  --image_size IMAGE_SIZE
                        Image dimensions
  --channels CHANNELS   Number of channels (3 for rgb, 1 for grayscale)
  --num_classes NUM_CLASSES
                        Number of classes in the dataset
  --train_images TRAIN_IMAGES
                        How many images you want to train on. Usually set to
                        the training set size. (optional)
  --test_images TEST_IMAGES
                        How many images you want to test on; usually set to
                        the test set size. (optional)
  --baseline BASELINE   Location of a file containing the baseline for the
                        specific dataset. Required for computation of
                        Fidelityand FIR.
  --model MODEL         Type of model to use. Available: "hns_small",
                        "hns_large" and "hns_resnet"
  --hider_weights HIDER_WEIGHTS
                        Location of a valid pretrained "hider" weights file
                        (optional but recommended)
  --seeker_weights SEEKER_WEIGHTS
                        Location of a valid pretrained "seeker" weights file
                        (optional)
  --batch_size BATCH_SIZE
                        Batch size
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs. Adaptive loss weighting may
                        cause the network to converge faster.
  --gpu GPU             Which gpu to use. Only relevant for multi-gpu
                        enviromnemts.
  --memory MEMORY       How much memory to allocate on a gpu
  --debug               If set to True, no weights or logs will be stored for
                        the models. It is intended for seeing if a script runs
                        properly, without generating empty logs or useless
                        weights.
  --evaluate            Choose whether or not to evaluate the modelafter the
                        training is completed.
```

The default values are:

```
'batch_size': 64, 'max_epochs': 10, 'gpu': 0, 'model': 'hns_small', 'config': None, 'data_dir': None,
'image_size': None, 'channels': None, 'num_classes': None, 'hider_weights': None, 'seeker_weights': None,
'train_images': None, 'test_images': None, 'stochastic': False, 'estimator': 'st1', 'patience': 100,
'alpha': None, 'monitor': 'classification', 'rate': 0.5, 'debug': False, 'num_trainings': 1, 'memory': None,
'evaluate': False, 'baseline': None
```

Note that out of these `data_dir`, `image_size` and `channels` are **mandatory** (i.e. they need to be specified), while 
`train_images`, `test_images` and `num_classes` can be infered upon if using a custom dataset.
