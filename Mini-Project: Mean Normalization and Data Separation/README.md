# Mini-Project: Mean Normalization and Data Separation

This repository contains a Jupyter Notebook project completed as part of the **Udacity Future AWS AI Programmer Nanodegree Program**. The project demonstrates fundamental data preprocessing and partitioning techniques using Python and NumPy, simulating key workflows required before feeding data into machine learning models.

---

## Project Overview

In machine learning, preparing raw data properly is critical to ensure algorithms converge efficiently and generalize well to unseen data. This project covers two core phases:

1. **Feature Scaling (Mean Normalization):**
   - Simulates a dataset of 1,000 rows and 20 columns containing random integers between 0 and 5,000.
   - Applies mean normalization across each column using the formula:
     $$\mbox{Norm\_Col}_i = \frac{\mbox{Col}_i - \mu_i}{\sigma_i}$$
   - Centers the data around zero with a standard deviation scaling, ensuring an average value close to zero.

2. **Data Separation (Dataset Splitting):**
   - Generates a random permutation of row indices using `np.random.permutation()` to prevent ordering bias.
   - Splits the normalized dataset into three standard machine learning subsets:
     - **Training Set (60%):** 600 rows $\times$ 20 columns
     - **Cross-Validation Set (20%):** 200 rows $\times$ 20 columns
     - **Test Set (20%):** 200 rows $\times$ 20 columns

---

## Repository Structure

```text
Future-AWS-AI-Programmer-nanodegree-program-/
└── Mini-Project: Mean Normalization and Data Separation/
    ├── Mean Normalization and Data Separation.ipynb
    └── README.md
```

---

## Technologies & Libraries Used

- **Python 3.10+**
- **NumPy** (for array creation, broadcasting, statistical computations, and random permutations)
- **Jupyter Notebook** (interactive development environment)

---

## How to Run the Notebook

1. Clone the repository or download the `.ipynb` file:
   ```bash
   git clone https://github.com/imtiaznehal08-cmd/Future-AWS-AI-Programmer-nanodegree-program-.git
   ```
2. Open Jupyter Lab or Jupyter Notebook on your local machine:
   ```bash
   jupyter notebook
   ```
3. Navigate to the `Mini-Project: Mean Normalization and Data Separation` folder and open `Mean Normalization and Data Separation.ipynb`.
4. Run the cells sequentially to observe data generation, normalization outputs, and dataset shape verification.

---

## Author

**Nehal Imtiaz**  
Udacity Future AWS AI Programmer Nanodegree Scholar
