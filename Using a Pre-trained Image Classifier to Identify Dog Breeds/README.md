# 🐾 AI Image Classifier & CNN Architecture Evaluator

A Python-based computer vision project that builds an image classification pipeline to evaluate and compare three different Convolutional Neural Network (CNN) architectures—**ResNet**, **AlexNet**, and **VGG**—using PyTorch and torchvision.

This project was developed as part of the Udacity AI programming curriculum to understand image feature extraction, label parsing, dataset statistics calculation, and model evaluation metrics.

---

## 🚀 Project Overview

The program processes a directory of pet and object images, performs automated classification using pre-trained deep learning models, and compares the predicted labels against true pet image labels to evaluate performance across two primary objectives:
1. **Objective 1:** Distinguishing between dogs and non-dog images.
2. **Objective 2:** Accurately classifying specific dog breeds.

---

## 🛠️ Project Structure

* **`check_images.py`**: The main orchestration script that controls the overall workflow from command-line arguments to final output.
* **`get_pet_labels.py`**: Extracts true pet labels from image filenames and populates a results dictionary.
* **`classify_images.py`**: Runs images through the selected CNN model (`resnet`, `alexnet`, or `vgg`) via PyTorch and compares predicted labels to truth labels.
* **`adjust_results4_isadog.py`**: Cross-references labels against a known dog name dictionary (`dognames.txt`) to determine whether both real and classifier labels represent dogs.
* **`calculates_results_stats.py`**: Computes core performance metrics, matching percentages, dog/non-dog accuracies, and breed classification accuracy.
* **`print_results.py`**: Formats and prints out structured performance summaries and misclassification logs.
* **`run_models_batch.sh` & `run_models_batch_uploaded.sh`**: Batch execution scripts to automate running all three models and piping outputs into text logs.

---

## 📊 Summary of Results

Evaluation metrics across the standard dataset highlight model strengths:
* **VGG** achieved the highest overall breed classification accuracy (**93.3%**) while maintaining 100% accuracy on non-dog separation.
* **ResNet** and **AlexNet** demonstrated strong baseline performance across feature extraction benchmarks.

---

## 💻 Running the Code

To run an individual model pipeline from the terminal:

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
