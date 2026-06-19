# Statistics Projects — Instacart Market Basket Analysis (S8)

Course project for Statistiques (S8) analyzing the Kaggle **Instacart Market Basket Analysis** dataset: exploratory analysis, customer segmentation, and prediction of which products a user will reorder.

## Structure

### Raw data
- `instacart-market-basket-analysis/` — original Kaggle dataset (`orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`, `sample_submission.csv`) plus an early R-based EDA (`Project_1.Rmd` / `.pdf`).
- `data.csv` — large (~3.3 GB) merged/derived dataset built from the above for modeling.
- `Panier achat.csv` — small synthetic boolean transaction dataset (Apple, Bread, Butter, Cheese…) used as a toy example for association-rule mining (Apriori), separate from the Instacart data.

### Reference material (not original work)
- `Instacart-Market-Basket-Analysis-main/` — a cloned/downloaded GitHub project (already has its own README, EDA, Apriori model, predictive model, Power BI dashboard, and a Flask deployment demo) kept for inspiration/reference.
- `eda-instacart-market-basket-analysis.ipynb`, `instacart-eda-rapids-and-similarity-recommenders.ipynb`, `pca-customer-segmentation.ipynb` — public Kaggle kernels downloaded for reference (they carry the standard Kaggle kernel boilerplate header), not authored by the student.

### Student's own work
- **`Trial_1.ipynb`** — first pass: EDA, reorder-rate analysis, a basic market basket analysis, and a simple logistic regression baseline.
- **`First_test.ipynb`** (report: `Rapports/First_test.pdf`) — data preparation/description of Instacart columns, descriptive statistics (orders by department/aisle, time-of-day and day-of-week distributions), then **customer segmentation** via PCA + KMeans (elbow method → 3 clusters), with statistical comparison of clusters (t-tests, Cohen's d) and a profile of each segment (occasional / less-loyal / highly-engaged buyers).
- **`Market Basket.ipynb`** / **`Market_Basket.ipynb`** — EDA of the basket-composition question from four angles: aisles, users, products, departments (notes that a 10k-user sample isn't representative of the full ~206k users, but larger samples are computationally heavier).
- **`prediction.ipynb`** / **`prediction copy.ipynb`** — main modeling pipeline:
  - Memory-usage reduction (downcasting dtypes) for the large prior/train order tables.
  - Feature engineering: user-level features (order count, avg. days between orders, reorder ratio, avg. basket size), product-level features (purchase frequency, avg. cart position, reorder ratio), and user×product features (order count, order rate, orders since first/last purchase).
  - Models: **Gaussian Naive Bayes** and **Logistic Regression** (scikit-learn, plus a `statsmodels` Logit for inference), evaluated with accuracy/precision/recall/F1/ROC-AUC/log-loss.
  - Multicollinearity check via **VIF** (variance inflation factor), followed by dropping redundant features.

### Reports & presentation
- `Rapports/First_test.pdf`, `Rapports/Pre_Rapport.pdf`, `Rapports/Prédiction du panier_ Rapport Final.pdf` — written reports for each project stage.
- `Rapports/Panier achat.pdf` — report on the toy Apriori/`Panier achat.csv` exercise.
- `Statistique_Presetation.pptx` — final presentation slides.
- `1.png` — a figure used in one of the notebooks/reports.

## Goal
Predict, for each user and previously purchased product, the probability that the product will be **reordered** in the user's next basket — combining EDA, unsupervised customer segmentation, and supervised classification.

## Stack
Python: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `seaborn`/`matplotlib`. R (`Project_1.Rmd`) for an early EDA pass.
