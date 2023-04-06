# Rubik's Cube Reinforcement Learning Model

This repository contains the code for my reinforcement learning model that solves Rubik's Cube. The model was developed as part of the Umojahack Africa 2023 competition, where it ranked first.

## The Rubik's Cube Problem

Rubik's Cube is a 3D combination puzzle that was invented by Ernő Rubik in 1974. The goal of the puzzle is to manipulate the cube until each face is a single color. The Rubik's Cube problem is a challenging task for reinforcement learning models due to its high complexity and the difficulty of providing a useful reward signal.

## Model Description

The model used in this repository is a variant of an off-policy Deep Q-Network (DQN) that learns through pseudo reward. The trained model is used as an estimation function in a beam search. I used different widths [100, 500, 1000, 5000, 10000] during training to optimize performance.
#
# Winning Solution

My reinforcement learning model was successful in solving all of the evaluation samples provided by Instadeep (easy, medium, hard). This is a significant achievement, considering the difficulty of the Rubik's Cube problem and the sparsity of the reward signal.

## How to Use

- Clone the repository to your local machine.
- Train the reinforcement learning model by running python `training.py`. This will train the model using the training data and save the resulting model to the `models` directory.
- Evaluate the trained model on the evaluation samples (easy, medium, hard) using python `inference.py`. This will load the trained model and evaluate its performance on each of the evaluation samples. The results will be printed to the console.
- (Optional) Modify the hyperparameters in training.py to optimize the model's performance.

Thank you for your interest in my Rubik's Cube Reinforcement Learning Model!
