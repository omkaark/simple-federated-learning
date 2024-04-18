<p align="center">
  <img src="https://github.com/omkaark/simple-federated-learning/assets/20964404/2a098f38-c7d0-4cc3-9f51-a1af8ecd835a" height="300" alt="Simple Federated Learning" />
</p>
<p align="center">
  <em>Federated Learning for local networks (photo credit: Parth Sareen's expert prompt engineering talent)</em>
</p>

# Federated Learning Setup Guide

This guide provides step-by-step instructions on how to set up and run a federated learning system with a central leader and multiple learners.

## Prerequisites

Before you begin, ensure you have the following requirements installed:

- Python 3.6+
- pip
- Access to a terminal or command line interface

## Installation

### Install Required Packages

Navigate to the project directory and install the required Python packages (preferrable in an [environment](https://docs.python.org/3/library/venv.html):

```bash
pip install -r requirements.txt
```

### Prepare Model Artifacts

Add your `model.py` file to the `model_artifacts` directory. This file should define the necessary variables including `device`, `model`, `criterion`, and `optimizer_function`. Check out my resnet-18 example model at model_artifacts/model.py If you want to change the model from Resnet to anything else, make sure you change it here: `model = <MODEL-DEF-GOES-HERE>`

### Configure Data Loader

Modify the data loader in `utils.py` to suit your setup or continue using the default setup configured for CIFAR-10.

## Network Configuration

Ensure all computers (leader and learners) are connected to the same network to facilitate communication.

## Usage

### Start the Leader

Run the leader script specifying the number of learners to wait for:

```bash
python leader.py --learner-count X
```

Replace `X` with the number of learners you want (# of computers you want to run training on).

### Start the Learners

Once the leader is running, it will display its address in the format `ADDRESS:PORT`. Use this address to start each learner:

```bash
python learner.py --leader-address ADDRESS:PORT
```

## Training Process

- After the specified number of learners have joined, training will start automatically.
- The system will train the model distributed across all learners and periodically synchronize their updates with the leader.
- Upon completion of the training, the validation accuracy will be printed, and a `model.pth` file will be generated. This file contains the trained model binary.

## Final Notes

- The training session's progress and results will be logged in the terminal.
- For troubleshooting and detailed logs, refer to the log files generated in the `logs` directory.
