# ML4VRP Project

**Machine Learning Guided Optimization for the Vehicle Routing Problem**

## Overview

This project focuses on solving the **Capacitated Vehicle Routing Problem (CVRP)** using a hybrid approach that combines optimization algorithms and machine learning techniques. The goal is to develop a system capable of generating efficient delivery routes for multiple customers while minimizing total travel distance and satisfying vehicle capacity constraints.

The project is inspired by the **Machine Learning for Vehicle Routing Problems (ML4VRP)** research initiative, which investigates how machine learning can improve traditional optimization algorithms for routing tasks.

## Problem Description

The **Vehicle Routing Problem (VRP)** is a well-known combinatorial optimization problem in logistics and transportation. In this project we focus on the **Capacitated Vehicle Routing Problem (CVRP)**, where:

* A fleet of vehicles starts from a central **depot**
* Each **customer** has a specific demand
* Each vehicle has a limited **capacity**
* Vehicles must deliver goods to customers while minimizing total travel distance

The objective is to determine optimal routes that serve all customers without exceeding vehicle capacity.

## Project Objectives

The main objectives of this project are:

* Understand the structure of VRP benchmark datasets
* Implement a baseline routing solution
* Develop a **Genetic Algorithm (GA)** based solver for CVRP
* Integrate **Machine Learning models** to guide the optimization process
* Evaluate performance using standard routing metrics
* Visualize routing solutions and experimental results

## Project Structure

```
ML4VRP_Project/
│
├── data/
│   VRP dataset instances
│
├── src/
│   Core source code
│   ├── load_vrp.py
│   ├── distance_matrix.py
│   ├── baseline algorithms
│   ├── genetic algorithm solver
│   └── machine learning components
│
├── docs/
│   Documentation and dataset description
│
├── experiments/
│   Scripts used to run experiments and evaluations
│
├── visualization/
│   Tools for plotting routes and performance graphs
│
└── README.md
```

## Milestones

The project is divided into several development stages:

### ST1: Dataset and Problem Setup

* Download and analyze VRP datasets
* Extract depot, customer coordinates, demand, and vehicle capacity
* Implement dataset loading utilities
* Create helper functions for route evaluation

### ST2: Baseline Solution

* Implement simple heuristics such as:

  * Nearest Neighbor
  * Greedy Route Construction
* Use baseline solutions for performance comparison

### ST3: Genetic Algorithm Solver

* Encode VRP routes as chromosomes
* Implement genetic algorithm components:

  * population initialization
  * fitness evaluation
  * selection
  * crossover
  * mutatio
