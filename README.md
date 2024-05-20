# Tabular-Text Transformer (TTT)
Resources for my ACL 2024 paper (Findings): **Revisiting Multimodal Transformers for Tabular Data with Text Fields.**


### Datasets
The datasets are loaded from a folder called "datasets".
For some of the use cases, we use the original training dataset as the test dataset does not include the true labels (competition data). In that case, we consider the training dataset as the modeling data which is then randomly split into training-validation-test subsets.
- airbnb: "cleansed_listings_dec18.csv" from https://www.kaggle.com/datasets/tylerx/melbourne-airbnb-open-data
- cloth: "Womens Clothing E-Commerce Reviews.csv" from https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
- jigsaw: "jigsaw_train_100k.csv" from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
- kick: "kickstarter_train.csv" ("train.csv" is the name of the original dataset) from https://www.kaggle.com/datasets/codename007/funding-successful-projects
- petfinder: "petfinder_train.csv" ("train.csv" is the name of the original dataset) from https://www.kaggle.com/competitions/petfinder-adoption-prediction/data
- salary: "Data_Scientist_Salary_Train.csv" ("Final_Train_Dataset.csv" is the name of the original dataset) from https://machinehack.com/hackathons/predict_the_data_scientists_salary_in_india_hackathon/overview
- wine10, wine100: "winemag-data-130k-v2.csv" from https://www.kaggle.com/datasets/zynicide/wine-reviews


### Code architecture
Py files:
- settings.py: this file contains all the settings by dataset (e.g. file name, list of features, number of classes) and for the pre-trained models.
- dataset.py: this file contains all the functions for pre-processing the data.
- model.py: this file contains all the classes defining the architectures of the models (including the ablation studies).
- optimization.py: this file contains all the functions for tuning, training, and evaluating the models.
- uncertainty.py: this file contains all the functions for quantifying and explaining the uncertainty.

Notebooks:
- performance_&_uncertainty_quant.ipynb: this main notebook allows to run the experiments, including uncertainty quantification.
- uncertainty_xai.ipynb: this notebook allows to run the experiments related to the explanation of uncertainty for TTT.




