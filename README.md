# CS 599 — Foundations of Deep Learning  
## Lab 3: Custom Normalization Techniques  
Author: **Janhvi Narayan Mishra**  
Northern Arizona University  
jm5747@nau.edu

---

## 📌 Overview

This repository contains the full implementation for **Lab 3** of *CS 599: Foundations of Deep Learning*.  
The goal of this assignment is to implement three normalization techniques **from scratch**, using **only basic TensorFlow operations**, and compare them against TensorFlow's built-in normalization layers.

Normalization methods implemented:

- **Batch Normalization**  
- **Layer Normalization**  
- **Weight Normalization**

All training was done using **tf.GradientTape()** for the backward pass, as required.

---

## 🧠 Project Objectives

- Implement BatchNorm, LayerNorm, and WeightNorm **without using Keras layers**  
- Build forward passes manually using TensorFlow ops  
- Train models with and without normalization  
- Compare custom normalization outputs with TensorFlow's native implementations  
- Compute gradient consistency and floating-point error  
- Report findings and analyze why results differ (or don’t)

Reference for assignment PDF:  
:contentReference[oaicite:0]{index=0}

---

## 📊 Dataset Summary

The project uses the **Fashion-MNIST** dataset.

| Split | Shape |
|-------|--------|
| Training | `(60000, 28, 28, 1)` |
| Testing | `(10000, 28, 28, 1)` |

Each sample is a **28×28 grayscale image**.

---

## 🧪 Model Training & Results

### **Baseline Model (No Normalization)**  
A simple CNN was trained without any normalization layers.

- **Test Accuracy:** `0.0718`  
This is equivalent to random guessing for a 10-class problem.

### **With Custom Normalization Layers**

Each of the three custom normalization methods (BatchNorm, LayerNorm, WeightNorm) was added individually to the same model architecture.

- **Test Accuracy:** `0.0718` (for all methods)

### **Comparison With TensorFlow’s Normalization**

We compared:

- Custom BatchNorm → `tf.keras.layers.BatchNormalization`
- Custom LayerNorm → `tf.keras.layers.LayerNormalization`
- Custom WeightNorm → `tfa.layers.WeightNormalization` (if available)

**Mean Absolute Difference (MAD) between outputs:**

