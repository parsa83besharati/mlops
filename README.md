# Customer Churn Prediction MLOps Project

## Introduction

This project represents a comprehensive implementation of Machine Learning Operations (MLOps) principles applied to the problem of customer churn prediction. Using the Telco Customer Churn dataset as our foundation, we have developed an end-to-end machine learning system that handles everything from data ingestion to production deployment. The primary objective was to create a fully automated pipeline that can ingest the latest version of customer data, perform necessary preprocessing and feature engineering, train and compare multiple machine learning models, track all experiments systematically using MLflow, select the optimal model based on performance metrics, and finally deploy that model as a containerized web service capable of handling real-time prediction requests.

The system has been designed with reproducibility as a core principle, ensuring that any training run can be precisely replicated using the same codebase and configuration parameters. This approach not only facilitates experimentation and model improvement but also provides the necessary infrastructure for maintaining and updating models in production environments.

## Project Architecture

The repository is organized into a logical directory structure that separates concerns and promotes maintainability. The data directory contains raw, processed, and versioned datasets, ensuring that we maintain a clear lineage of our data transformations. The models directory stores trained model artifacts, while the deployment_model directory holds the final exported model ready for serving. All MLflow tracking data is stored in the mlruns directory, which maintains the complete history of our experimentation. The source code resides in the src directory, with modular components for preprocessing, feature engineering, training, evaluation, and MLflow utilities. The main pipeline orchestration is handled by the run_pipeline.py script, while Dockerfile and requirements.txt facilitate containerization and dependency management.

## Dataset Characteristics and Preprocessing

The Telco Customer Churn dataset provides rich customer information including demographic details, subscription plans, billing history, and the critical target variable indicating whether a customer has churned. This dataset presents several challenges that our preprocessing pipeline addresses comprehensively. Missing values are handled through appropriate imputation strategies, categorical variables are transformed using one-hot encoding to make them suitable for machine learning algorithms, and numerical features undergo scaling to ensure they contribute equally to model training.

Beyond these standard preprocessing steps, we have implemented feature engineering to create additional meaningful variables that capture customer behavior patterns more effectively. The average monthly charge is calculated as the ratio of total charges to tenure, providing insight into the customer's typical spending level. A binary indicator for high monthly charges helps identify customers who may be particularly sensitive to pricing changes. The new customer flag distinguishes recent acquisitions from long-standing customers, as these groups often exhibit different churn patterns. These engineered features have proven valuable in improving model performance and providing additional context for prediction.

## Machine Learning Models and Optimization

Our approach employs an ensemble of four distinct machine learning algorithms to maximize the chances of identifying the best performing model for this specific prediction task. Logistic Regression serves as our baseline linear model, providing interpretability and serving as a reference point for more complex approaches. Random Forest offers robust ensemble learning through bagging and feature randomness, handling non-linear relationships effectively. XGBoost brings gradient boosting with advanced regularization and optimization techniques, often delivering state-of-the-art performance on structured data. CatBoost completes our model suite with its specialized handling of categorical features and innovative ordered boosting algorithm.

Each model undergoes comprehensive hyperparameter optimization using GridSearchCV, which systematically explores combinations of hyperparameter values to identify the configuration that yields the best performance on validation data. The optimization process is integrated into our pipeline, ensuring that every model is properly tuned before evaluation. This approach prevents the common pitfall of comparing untuned models, which can lead to misleading conclusions about algorithm superiority.

## Experiment Tracking with MLflow

One of the most critical components of our MLOps implementation is the systematic tracking of all experiments using MLflow. Every training run automatically logs a comprehensive set of information including all hyperparameters used, training metrics, validation metrics, test metrics, and the trained model artifacts themselves. This extensive logging enables us to maintain a complete history of our experimentation, allowing us to compare different runs, analyze the impact of various hyperparameters and preprocessing choices, and identify the best performing models with confidence.

The MLflow tracking server provides a user-friendly interface accessible through a web browser, making it easy to visualize and compare experiment results. This transparency is invaluable for collaboration among team members and for maintaining a clear record of model development history. The ability to revisit previous experiments and reproduce them exactly as they were run ensures that our work remains scientifically rigorous and auditable.

## Model Selection and Performance

Based on the validation set performance, our pipeline automatically selects the model with the highest F1 score as the final deployment candidate. The F1 score was chosen as our primary selection metric because it provides a balanced measure of precision and recall, which is particularly important in churn prediction where both false positives and false negatives carry significant business implications. In our most recent run, the Logistic Regression model emerged as the best performer, achieving a validation F1 score of 0.6543 and a test F1 score of 0.5867.

This selection demonstrates that sometimes simpler models can outperform more complex alternatives, particularly when the dataset size is moderate and feature engineering has been done effectively. The complete performance metrics for all models, including accuracy, precision, recall, ROC-AUC, and confusion matrices, are automatically logged and available for review in the MLflow interface. This comprehensive evaluation allows stakeholders to understand the trade-offs between different models and make informed decisions about deployment.

## Deployment Pipeline

The deployment process begins immediately after model selection, with the best performing model being exported to the deployment_model directory in MLflow's standardized format. This exported model contains all necessary information for serving, including the preprocessing steps, feature engineering transformations, and the trained model itself, ensuring consistency between training and inference environments.

For local testing and development, the model can be served using MLflow's built-in serving functionality, which creates a REST API endpoint on port 5001. The health of the service can be verified through the ping endpoint, confirming that the model is loaded and ready to accept predictions. This local serving capability is invaluable for testing the model's integration with other systems and validating its behavior before production deployment.

## Containerization with Docker

To ensure portability and consistency across different environments, we have containerized the prediction service using Docker. The Dockerfile defines a lightweight container image that includes all necessary dependencies and configures the service to run automatically when the container starts. Building the image is straightforward with a single docker build command, and running the container maps the internal port 8080 to the host port 5001, making the service accessible from the host machine.

This containerized approach provides several significant advantages for production deployment. The container encapsulates the complete runtime environment, eliminating the common problem of "it works on my machine" dependency issues. Scaling the service horizontally becomes trivial by launching additional container instances behind a load balancer. The container can be deployed on any platform that supports Docker, including Kubernetes clusters, cloud container services, and on-premise infrastructure, providing flexibility in deployment choices.

## Making Predictions

The deployed service accepts prediction requests in JSON format, with the request body containing a dataframe_records array where each element represents a customer's feature vector. The features must exactly match those used during training, including all one-hot encoded categorical variables and the engineered features. We provide a sample.json file that demonstrates the correct format, making it easy for clients to construct valid requests.

Prediction responses are returned as JSON containing an array of predictions, where 0 indicates that the customer is predicted to stay and 1 indicates predicted churn. The service is designed to handle batch predictions efficiently, processing multiple customers in a single request. For integration with other systems, the REST API follows standard conventions and can be accessed using any HTTP client, including command-line tools like Invoke-RestMethod on Windows or curl on Unix-like systems.

## Reproducibility and Continuous Improvement

The entire pipeline is designed for maximum reproducibility. Running the run_pipeline.py script again will automatically load the latest available dataset version, execute all preprocessing and feature engineering steps, train all models with their respective hyperparameter optimization, evaluate performance on validation and test sets, log everything to MLflow, and update the deployment model if a better performing model is found. This automation means that as new data becomes available, the system can be retrained with minimal manual intervention.

This reproducibility extends to the MLflow tracking as well, where each run is identified by a unique run ID and all associated parameters and artifacts are stored permanently. This allows us to revisit any previous experiment, understand exactly what was done, and potentially reproduce the results. The combination of automated pipelines and comprehensive tracking creates a foundation for continuous improvement of the churn prediction system over time.

## Technology Stack Summary

The project leverages a modern Python-based data science and machine learning stack. Pandas and NumPy handle data manipulation and numerical operations efficiently. Scikit-learn provides the preprocessing tools, model evaluation metrics, and the Logistic Regression and Random Forest implementations. XGBoost and CatBoost contribute their specialized gradient boosting algorithms. MLflow serves as the experiment tracking and model management platform. Docker enables consistent deployment across environments. Matplotlib and Seaborn support visualization and exploratory data analysis when needed.

## Conclusion

This MLOps implementation for customer churn prediction demonstrates the practical application of machine learning engineering principles to a real-world business problem. The system successfully integrates data preprocessing, feature engineering, model training and selection, experiment tracking, and containerized deployment into a cohesive, automated pipeline. The resulting service provides reliable predictions through a REST API, supporting integration with various downstream systems and business processes.

The architecture is designed to be maintainable and extensible, allowing for easy incorporation of new models, additional preprocessing steps, or alternative dataset sources as requirements evolve. By emphasizing reproducibility and systematic tracking, we have created a foundation that supports both operational stability and ongoing model improvement. This project serves as a template for implementing MLOps practices in other predictive modeling initiatives, demonstrating how to bridge the gap between data science experimentation and production machine learning systems.