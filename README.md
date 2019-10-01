#  Foundations Atlas Tutorial
<img src='images/atlas_logo.png'>

# What is Atlas?

Atlas was built by our machine learning engineers to dramatically reduce the model development time, from the experimentation workflow to production.

Here are some of the core features:

**1. Experiment Management & Tracking:**

Tag experiments and easily track hyperparameters, metrics, and artifacts such as images, GIFs, and audio clips in a web-based GUI to track the performance of your models
<img src='images/management_tracking.png'>

**2. Job queuing & scheduling:**

Launch and queue thousands of experiment variations to fully utilize your system resources
<img src='images/job_queue.png'>

**3. Collaboration & Bookkeeping:**

Keep a journal of thoughts, ideas, and comments on projects

**4. Reproducibility:**

Maintain an audit trail of every single experiment you run, complete with code and any saved items


# Start Guide

**Prerequisites**

1. Docker version >18.09 (Docker installation: [Mac](https://docs.docker.com/docker-for-mac/install/) | [Windows](https://docs.docker.com/docker-for-windows/install/))
2. Python >3.6 ([Anaconda installation](https://www.anaconda.com/distribution/))
3. \>5GB of free machine storage
4. The atlas_ce_installer.py file (Download after signup [here](https://www.atlas.dessa.com/))


**Steps**

See [Atlas documentation](https://dessa-atlas-community-docs.readthedocs-hosted.com/en/latest/ce-quickstart-guide/). 

<details>
  <summary>FAQ: How to upgrade an older version of Atlas?</summary>
<br>

1. Stop atlas server using `atlas-server stop`
2. Remove docker images related to Atlas in your terminal

`docker images | grep atlas-ce | awk '{print $3}' | xargs docker rmi -f`

-------------------------------------------------------------------------------------------------------------------------
</details>

# Image Segmentation

This tutorial demonstrates how to make use of the features of Foundations Atlas. Note that **any machine learning job can be run in Atlas without modification.** However, with minimal changes to the code we can take advantage of Atlas features that will enable us to:

* view artifacts such as plots and tensorboard logs, alongside model performance metrics
* launch many training jobs at once
* organize model experiments more systematically


## Start Atlas

Activate the conda environment in which Foundations Atlas is installed. Then run `atlas-server start` in a new tab terminal. Validate that the GUI has been started by accessing it at [http://localhost:5555/projects]().



## Data



In this repo, we show how can one perfom image segmentation on some pet imgaes using tensorflow.


# How to run it
In order to run a single job with foundations, just type the following in the terminal:
```python
foundations submit scheduelr . main.py
```
Make sure your current directory is `pet_segmentation`.

In order to run multiple experiments, run `python hyperparameter_search.py` in the terminal.

# Track experiments
