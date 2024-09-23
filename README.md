# **E-Commerce Customer Spending Prediction with PySpark**

## **Project Overview**
This project uses PySpark to perform linear regression on an e-commerce dataset. The goal is to predict the **Yearly Amount Spent** by customers based on features such as **Avg Session Length**, **Time on App**, and **Length of Membership**.

## **Dataset**
The dataset used for this project is `Ecommerce_Customers.csv`. It contains the following key features:

- **Avg Session Length**: Average time customers spend on the website per session.
- **Time on App**: The amount of time customers spend on the mobile app.
- **Length of Membership**: How long a customer has been a member of the service.
- **Yearly Amount Spent**: Target variable representing the annual amount spent by a customer.

## **Tools and Libraries**
- **PySpark**: For distributed computing and linear regression modeling.
- **Pandas**: Data manipulation and cleaning.
- **Matplotlib**: Data visualization.
- **Jupyter Notebook**: For interactive analysis.

## **Project Workflow**
1. **Installing PySpark**:
    Install PySpark in your environment using the following command:
    ```bash
    !pip install pyspark
    ```

2. **Create a SparkSession**:
    Start by creating a `SparkSession`, which is the entry point to use Spark:
    ```python
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName('Customers').getOrCreate()
    ```

3. **Loading the Dataset**:
    Load the CSV dataset using `Spark`:
    ```python
    dataset = spark.read.csv("C:\\Users\\HP\\Downloads\\Ecommerce_Customers.csv", inferSchema=True, header=True)
    dataset.show()
    dataset.printSchema()
    ```

4. **Preparing Features**:
    Use `VectorAssembler` to combine input features into a single vector:
    ```python
    from pyspark.ml.feature import VectorAssembler

    featuresassembler = VectorAssembler(inputCols=["Avg Session Length", "Time on App", "Length of Membership"], outputCol="Independent Features")
    output = featuresassembler.transform(dataset)
    output.show()
    ```

5. **Selecting the Relevant Data**:
    Prepare the final dataset by selecting the features and target variable:
    ```python
    finalized_data = output.select("Independent Features", "Yearly Amount Spent")
    finalized_data.show()
    ```

6. **Train-Test Split**:
    Split the data into 75% training and 25% testing sets:
    ```python
    train_data, test_data = finalized_data.randomSplit([0.75, 0.25])
    ```

7. **Building and Training the Model**:
    Define a linear regression model using the selected features and target:
    ```python
    from pyspark.ml.regression import LinearRegression

    regressor = LinearRegression(featuresCol='Independent Features', labelCol='Yearly Amount Spent')
    regressor = regressor.fit(train_data)
    ```

8. **Inspecting the Model Coefficients**:
    View the coefficients and intercept of the trained model:
    ```python
    print("Coefficients: ", regressor.coefficients)
    print("Intercept: ", regressor.intercept)
    ```

9. **Evaluating the Model**:
    Use the testing data to evaluate the model's predictions:
    ```python
    pred_results = regressor.evaluate(test_data)
    pred_results.predictions.show()
    ```

## **Results**
- The model predicts **Yearly Amount Spent** based on customer metrics with a linear regression approach.
- Key outputs include the model's coefficients, intercept, and a set of predictions on unseen test data.

## **Conclusion**
This project demonstrates how to use PySpark for linear regression modeling. The analysis helps in understanding the factors influencing customer spending in an e-commerce environment. The model provides insights into the relationship between customer behavior metrics and their yearly spending, allowing businesses to make informed decisions.

## **Requirements**
Ensure the following libraries are installed:
- PySpark
- Pandas
- Matplotlib



