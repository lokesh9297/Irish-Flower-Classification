# Iris Flower Classification Project 🌸

## Overview
This project focuses on classifying Iris flower species—Setosa, Versicolor, and Virginica—using machine learning (ML). By analyzing petal and sepal dimensions, the model predicts the species of an Iris flower based on its unique morphological characteristics. This repository includes code, analysis, and results for building, evaluating, and visualizing the classifier.

## Project Goals
- Explore and visualize the Iris dataset to understand feature distribution and relationships.
- Preprocess and scale data to improve model performance.
- Train and evaluate multiple classification models: **K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest**.
- Compare model performance using accuracy, precision, recall, and AUC.
- Use a **custom confusion matrix visualization** to gain deeper insights into model predictions.

## Dataset 📊
The dataset used in this project consists of:
- **150 samples** of Iris flowers.
- **4 features**: Petal length, Petal width, Sepal length, and Sepal width.
- **3 target classes**: Setosa, Versicolor, and Virginica.

## Project Structure 📚
1. **Data Loading & Preprocessing**:
   - Load dataset, clean data, check for class balance, and scale features where needed.
2. **Data Visualization**:
   - Explore feature distributions and relationships using scatter plots and pair plots.
3. **Model Training & Evaluation**:
   - Train three models (**KNN, SVM, Random Forest**) and compare performance using accuracy, precision, recall, and AUC scores.
4. **Confusion Matrix Analysis**:
   - Utilize a custom function to display TP, FP, FN, and TN values for detailed model assessment.
5. **Feature Importance Analysis (Random Forest)**:
   - Identify the most influential features in classification.

## Getting Started 🚀
### Prerequisites
Ensure you have **Python 3.x** installed along with the following libraries:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

### Installation
Clone this repository:
```bash
git clone https://github.com/iskburcin/1.IrishFlowerClassification.git
cd iris-flower-classification
```
Open `iris_classification.ipynb` in Jupyter Notebook.

## Detailed Breakdown 📝
### 1. Data Loading & Preprocessing
- **Import Data**: Load the Iris dataset and perform initial data exploration.
- **Train-Test Split**: Split data into **80% training** and **20% testing**.
- **Feature Scaling**: Standardize data for KNN and SVM models to ensure proper distance-based calculations.

### 2. Data Visualization
- **Pair Plots & Scatter Plots**: Visualize feature separability and identify critical feature combinations.

### 3. Model Training & Evaluation
- **Algorithms Used**: Train KNN, SVM, and Random Forest models.
- **Cross-Validation**: Use **k-fold cross-validation** to enhance model stability and prevent overfitting.
- **Hyperparameter Tuning**: Optimize models using **Grid Search**.

### 4. Confusion Matrix Analysis
- Implement a **custom confusion matrix function** to display **true positives, false positives, false negatives, and true negatives** for each species.

### 5. Feature Importance Analysis
- The **Random Forest model** identifies key features, revealing that **petal length and petal width** play the most significant roles in species differentiation.

## Results & Insights 🏆
- **Model Performance**: Random Forest achieved the best accuracy due to its ensemble learning approach.
- **Key Features**: Petal dimensions were the most decisive factors in classification.
- **Confusion Matrix Insights**: The model performed well in distinguishing Setosa, with minor misclassifications between Versicolor and Virginica.

## Future Enhancements 🔍
- **Explore More Models**: Implement additional classifiers like Decision Trees or Neural Networks.
- **Advanced Hyperparameter Tuning**: Use **Randomized Search** or **Bayesian Optimization** for efficient tuning.
- **Model Deployment**: Develop a web app or API for real-time Iris flower classification.

## Conclusion 🏁
This project successfully demonstrates **Iris flower classification** using machine learning. Through **data visualization, model evaluation, and feature importance analysis**, we achieve accurate and interpretable predictions, laying the groundwork for further enhancements and real-world applications.

---
📖 **References**: Fisher, R.A. (1936). The Use of Multiple Measurements in Taxonomic Problems. Annals of Eugenics.

