# Predictive Maintenance for Truck Air Systems with RANDOM FOREST MODEL
* to run chosen model: to_bix/models/random_forest/run_train-RF.ipynb
* The layout of this README could be improved by accessing the file.

### Project Overview
This project aims to develop a machine learning model to predict potential air system failures in trucks, enabling proactive maintenance and reducing downtime and costs for a transport company facing increased air system maintenance expenses. 

### Dataset
The project uses two datasets provided:
1. **`air_system_previous_years.csv`:**  Historical data from previous years, containing 171 ENCODED columns, used for training and evaluating the models.
2. **`air_system_present_year.csv`:** Current year's data, used to assess the model's real-world performance by comparing its predictions to the actual maintenance records, effectively verifying the model's ability to correctly identify trucks requiring maintenance.

### Project Structure
- **`air_system_present_year.csv`:** Dataset for the current year.
- **`air_system_previous_years.csv`:** Dataset for previous years.
- **`run-train_RandomF.ipynb`:**  Random Forst Model
- **`run-train_NeuralN.py`:**  Neural Network Model
- **`best_params.txt`:** File to store the best hyperparameters found during tuning. 
- **`README.md`:**  Project documentation.


**Structure Dir:**
|├── dataset
|│ ├── air_system_present_year.csv
|│ └── air_system_previous_years.csv
|├── models
|│ ├── neural_network
|│ │ └── run-train_NeuralN.py
|│ └── random_forest
|│ └── run-train_RandomF.ipynb (chosen model)
|├── results
|│ └── best_params.txt
|└── README.md

### Dependencies
**The chosen model is Random Forest in Jupyter Notebook for better visualization of the process, with the following Python packages listed in requirements.txt:**
* Create the venv: ```$ pytohn3 -m venv .venv```

## Activate the venv:
* On Linux/macOS: ```$ source .venv/bin/activate``` 
* On Windows: ```$ .venv/scripts/activate```

### RANDOM FOREST MODEL - CHOSEN:
**OBS: on the first cell of """run_train-RF.ipynb.ipynb""" contain this pip**
* After venv activated: ```$ pip install -r requirements.txt```

### NEURAL NETWORK:
**For """run_train_NN-pytorch.py""", Neural Network necessary create a venv and needs a GPU, for this run this pip on your terminal**
```$ pip install pandas torch scikit-learn imbalanced-learn tqdm```
```$ python3 run_train_NN-pytorch.py```

### Methodology
1. **Data Preprocessing:**
    - **Handling Missing Values:** Missing values ('na') were replaced with the mean of their respective columns. This approach helps maintain the overall distribution of the data.
    - **Converting Categorical Variables:** The 'class' column, originally containing "pos" and "neg" labels, was converted to a binary format (1 for "pos" and 0 for "neg") for compatibility with machine learning algorithms. 

2. **Feature Engineering:**
    - **Not Performed:** Due to the encoded nature of the features and the lack of information about the meaning behind the encoding, feature engineering was not performed in this project. Gaining a deeper understanding of the features (e.g., through decoding or feature descriptions) would be essential for exploring potential feature engineering opportunities in future work. 

3. **Feature Selection:**
    -  **Feature Importance Analysis:** Random Forest's feature importance scores were utilized to rank features based on their contribution to the model's predictive power.  
    - **Selection:** The top 16 features with the highest importance scores were selected: 
        * `['ag_005', 'az_005', 'ba_000', 'bb_000', 'bx_000', 'bu_000', 'bv_000', 'cc_000', 'ci_000', 'cn_004', 'cn_005', 'cq_000', 'cs_005', 'ee_002', 'ee_003', 'ee_004', 'ad_000']`

4. **Addressing Class Imbalance:**
    -  Recognizing potential class imbalance (more "neg" than "pos" instances), the SMOTE (Synthetic Minority Over-sampling Technique) was employed to generate synthetic samples for the minority class ("pos"). This ensures the model learns effectively from both classes. 

5. **Data Standardization:**
    - Features were standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1. This is a crucial step for many machine learning algorithms.

6. **Model Selection and Training:**
   - **Models Considered:** In addition to Random Forest, other models were considered, including Logistic Regression (using PyTorch), Support Vector Machines (SVM), and Decision Tree. However, Random Forest proved to be more suitable due to its ability to handle the complexity of the encoded data. 
   - **Random Forest Classifier:** This model was selected as the primary model for several reasons. Firstly, it effectively handles potentially noisy data, which aligns well with the encoded columns of the dataset, where relationships between variables might not be linear. Secondly, its robustness to outliers is crucial, given the possibility of unusual sensor readings. 

7. **Hyperparameter Tuning and Evaluation:**
    - **Randomized Search:** `RandomizedSearchCV` was employed to optimize the Random Forest hyperparameters, with the goal of achieving the highest `Recall` score.  Experiments were conducted with 20, 30, and 50 random combinations, progressively increasing the range of values explored for each hyperparameter:
        - `n_estimators`:  Uniform distribution between 50 and 600.
        - `max_depth`: Uniform distribution between 1 and 250.
        - `min_samples_split`: Uniform distribution between 2 and 30.
        - `min_samples_leaf`: Uniform distribution between 1 and 8.
    - **Evaluation Metrics:** The model was evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  Recall was prioritized as the primary metric due to the high cost associated with false negatives.

8. **Model Interpretation:**
    - **Feature Importance:** The top contributing features were analyzed to understand the sensor readings most indicative of potential air system failures. The ability of Random Forest to provide these insights is particularly useful, even with encoded data, for gaining a better understanding of the factors influencing predictions.

    - **Feature Importance Plot:**
    ![First Feature Importance verification Plot](/results/boxplot_columns.png)
    ![Second Feature Importance verification Plot](/results/correlation_columns.png) 

## Results and Business Impact

### Model Performance and Cost Savings
After performing a randomized search for hyperparameter optimization, exploring 50 different combinations with 5-fold cross-validation, the following optimal hyperparameters were found for the Random Forest model:

* `max_depth`: 216
* `min_samples_leaf`: 1
* `min_samples_split`: 3
* `n_estimators`: 324

The model's performance on the "Present Year" dataset, using these optimized hyperparameters with 50iterations, is summarized below:
|--------------------|-----------------|
| Metric             | Score           |
|--------------------|-----------------|
| Accuracy           | 0.978           |
| Precision          | 0.525           |
| Recall             | 0.845           |
| F1-Score           | 0.648           |
| ROC-AUC            | 0.913           |
|--------------------|-----------------|

**Confusion Matrix:**
[[15339 287]
[ 58 317]]

**Cost Analysis RANDOM FOREST:**
* **Total Cost without Model:** $187,500 
* **Total Cost with Model:** $39,795
* **Potential Savings:** $147,705

* **Present_year_real_values:** 15626 neg and 375 pos.
* **Present_year_pred_values:** 15570 neg and 431 pos.
* **-------------- Diference:** -56 neg and 56 pos. 

**Cost Analysis NEURAL NETWORK:**
* **Total Cost without Model:** $187,500 
* **Total Cost with Model:** $46,200
* **Potential Savings:** $141,300

* **Present_year_real_values:** 15626 neg and 375 pos.
* **Present_year_pred_values:** 15484 neg and 517 pos.
* **-------------- Diference:** -142 neg and 142 pos. 

As observed, the model significantly reduces maintenance costs. This reduction is largely attributed to the high Recall score (0.845), which means the model correctly identifies a large proportion of trucks that require maintenance, minimizing the occurrence of costly corrective maintenance due to missed failures.

**Direct Comparison:**
* **Actual Maintenance:** 15,626 trucks were labeled as "neg" (no maintenance) and 375 were labeled as "pos" (received maintenance).
* **Model Predictions:** 15,397 trucks were predicted as "neg" and 604 were predicted as "pos."
* **Difference:** The model correctly adjusted the classification, identifying 229 more trucks that likely needed maintenance.

**Trucks Recommended for Maintenance (Example):**
|----------|---------------|----------------|--------------------|------------------|---------------|-------------------|
| id_truck | predict_class | original_class | maintenance_action | maintenance_cost | original_cost | threshold_predict |
|----------|---------------|----------------|--------------------|------------------|---------------|-------------------|
| 418      | 1             | 0              | maintenance_now    | 25               | 500           | 1.0               |
| 428      | 1             | 1              | maintenance_now    | 25               | 25            | 1.0               |
| 436      | 1             | 1              | maintenance_now    | 25               | 25            | 1.0               |
| 445      | 1             | 1              | maintenance_now    | 25               | 25            | 1.0               |
| 461      | 1             | 0              | maintenance_now    | 25               | 500           | 1.0               |
| ...      | ...           | ...            | ...                | ...              | ...           | ...               |
|----------|---------------|----------------|--------------------|------------------|---------------|-------------------|

This table showcases a sample of trucks identified by the model as potentially needing maintenance. This information can be used by the company to proactively schedule inspections and repairs, preventing costly breakdowns and maximizing fleet efficiency.

## Addressing the Challenge Activities
**1. Steps to Solve the Problem:** The project followed a structured methodology involving data preprocessing, feature selection, handling of class imbalance, model selection and training, hyperparameter tuning, evaluation, and interpretation.  Key steps included addressing missing values, converting categorical variables, applying the SMOTE technique, selecting Random Forest as the primary model, and optimizing for Recall using Randomized Search. 

**2. Technical Data Science Metric:** Recall (Sensitivity) was chosen as the primary metric for model optimization.

**3. Business Metric:**  The primary business metric used to evaluate the model's impact is cost savings resulting from reduced maintenance expenses.

**4. Relationship Between Metrics:** A higher recall leads to fewer missed failures (false negatives), which directly translates to lower maintenance costs, as false negatives represent the most expensive scenario ($500 per missed failure). While maximizing Recall is crucial, Precision also plays a role in cost optimization. Higher Precision minimizes unnecessary maintenance on trucks that wouldn't actually require it (reducing the $10 cost per false positive).

**5. Database Analysis:** 
    - **Exploratory Data Analysis (EDA):** Analysis was performed to gain a better understanding of the data, using techniques such as histograms, scatter plots, and correlation analysis to examine data distribution, patterns, and relationships between features values and the target binary variable.
    - **Feature Importance Analysis:**  Feature importance scores were analyzed to identify the most relevant predictors for the model.

**6. Dimensionality Reduction:**  Feature selection based on feature importance was employed to reduce dimensionality. This method was chosen because it directly selects the most influential features for predicting air system failures, simplifying the model and potentially improving its performance. Other techniques like a component analysis could be explored if the encoded features were interpretable, allowing for the creation of new features based on patterns.

**7. Variable Selection:**  Random Forest's built-in feature importance scores were used to rank and select the most relevant features. 

**8. Predictive Models Tested:**
    - **Random Forest Classifier:** This model, known for its performance and interpretability, was implemented and evaluated. To optimize its performance, a Randomized Search approach was employed for hyperparameter tuning, focusing on maximizing the Recall score. This choice was made due to the high cost associated with false negatives (missed maintenance).
    - **Neural Network:** A neural network model was also developed and tested using PyTorch. The model's architecture and training process are detailed in the code. However, due to hardware limitations and the computational cost of hyperparameter tuning for neural networks, an extensive hyperparameter search was not performed for this model. 
    - **Support Vector Machines (Considered, Not Implemented):**  SVMs are powerful models capable of capturing complex decision boundaries and potentially offering high accuracy. However, due to time constraints and the computational demands of SVMs, this model was not implemented in this project iteration.
    - **Decision Tree (Considered, Not Implemented):**  Decision Trees are simple and interpretable models that could provide insights into the decision-making process. However, their tendency to overfit, especially with complex datasets, led to the prioritization of other models in this iteration.

**9. Model Selection:** Both the Random Forest and the Neural Network models were evaluated based on their Recall scores, as well as other relevant metrics and their potential business impact. The final model selection will be determined based on a comprehensive analysis of these factors.

**10. Model Explanation and Feature Importance:**
    - Feature importance scores provide insights into the most influential sensor readings. This analysis is further detailed in section 8, where the chosen model and its interpretability are discussed.

**11. Financial Impact Assessment:** The "Cost Impact" section quantifies the potential savings based on the model's predictions and the associated maintenance costs.

**12. Hyperparameter Optimization:** Randomized Search was employed to fine-tune the Random Forest model's hyperparameters. 

**13. Risks and Precautions:**
    - **Data Dependency:** The model's performance is inherently dependent on the quality and representativeness of the historical data used for training. Changes in data patterns or the introduction of new, unforeseen factors could impact future predictions.
    - **Model Monitoring:** Continuous monitoring of the model's performance is crucial to detect any degradation in predictive accuracy over time. This allows for timely retraining or adjustments to ensure ongoing effectiveness. 
    - **Data Interpretability:** Due to the encoded nature of the features, interpreting the model's decisions and understanding the specific factors driving predictions can be challenging. 
    - **Model Updates:**  As the trucking fleet and operational conditions evolve, the model might require periodic retraining with new data to maintain its accuracy and relevance.
    
**14. Model Deployment:**
    - To operationalize the model, a pipeline will be developed to automate data preprocessing, model prediction, and generation of maintenance recommendations. 
    - The model can be deployed as a standalone application, integrated into the company's existing maintenance software via an API, or accessed through a user-friendly dashboard that displays predictions and insights.

**15. Model Monitoring:**
    - Key performance metrics, including accuracy, recall, precision, and F1-score, will be continuously tracked to monitor the model's effectiveness over time.
    - Additionally, business-specific metrics, such as average maintenance cost and truck downtime, will be monitored to assess the model's real-world impact. 
    - Automated alerts will be set up to notify stakeholders of significant performance drops, indicating a potential need for model retraining.

**16. Model Retraining:**
    - The model will be retrained periodically, for instance, on a quarterly or annual basis, using new data collected from the fleet. This ensures that the model remains up-to-date with potential changes in data patterns and maintains its predictive accuracy.
    - Retraining will also be triggered if significant performance degradation is detected during monitoring or if there are substantial changes in the trucking fleet or operational conditions.

## Conclusion

This project demonstrates the effectiveness of machine learning in predicting truck air system failures. Both the Random Forest and Neural Network models showcased the potential for significant cost savings through proactive maintenance. By leveraging these models, the company can optimize maintenance schedules, minimize truck downtime, and improve overall operational efficiency. 
