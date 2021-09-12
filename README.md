# Periculum_Group_B_IE
# Risk Based Segmentation Package

Periculum is for risk assessment. It helps in data cleaning and modeling.

This package tries, through a very simple approach, to collect all the common tasks that are normally done through pandas DataFrames - classes and functions are created to facilitate this work:

# Install
To install this package, simply pip from this git repo:
**!pip install Periculum_Group_B_IE**

# 1-	Class
class **RiskDataframe:**

The class is used to extend the properties of Dataframes to a particular type of Dataframes in the Risk Industry. It provides the end user with both general and specific cleaning functions, though they never reference a specific VARIABLE NAME. It facilitates the End User to perform some Date Feature Engineering, Scaling, Encoding, etc. to avoid code repetition.

# 2-	Data Manipulation

Method **date_types**:

This method is to create a list for each column types, in order to use these lists in further methods that will be called.

Parameters: All variables

Returns: list of each column type 

Method **date_to_int**:

This is an optional method in case the data in the dataframe has date columns; it will convert all the dates to year fraction from the reporting date to calculate the time difference. Which is necessary for the application of the segmentation method, because Sklearn logistic regression does not accept date type values. 

Parameters: date type variables

Returns: integer value of the time delta

# 3-	 Risk Based Approach

Method **missing_not_at_random**:

This method checks for the correlation between the missing values in all the columns, pair by pair in order to see if the correlation is higher than threshold to be considered missing not at random.

corr_threshold: This variable is the threshold that will be used as a cut off to decide if the correlation between the missing values between a pair of columns is high enough to be considered missing not at random.

Parameters: All variables

Returns: Variables name that are missing not at random

Function **redundant_pairs**:

Used to find pairs of columns that are repeated in the correlation matrix used in the method missing_not_at_random

Parameters: Variables that are correlated in missing not at random, correlation threshold defined as corr_threshold

Returns: Variables name that are missing not at random

# 4-	Splitting and Modeling

Method **find_segment_split**:

The method aims to segment customers depending on the variables if they are good for segmentation or not to have a better model fit.
The model used is logistic regression for a binary classification, which does not accept alphanumeric values; therefore, label encoder is automatically called if the method detects these data type columns. 

The required argument for this method is target, since the logistic regression model needs this. Robust_scaler is an optional argument in order to enhance model performance. Once the baseline model with the full file without segmentation is calculated, the method continues to find where is the optimal place for splitting each column by applying a decision tree classifier, and extracting the root node splitting point. Finally it fits a model on the segmented dataset and compares the results of both models.

Parameters: Categorical variables transformed to numerical values (there should be no missing values).

Returns: 

ORIGINAL_BOOKED_AMOUNT: Not good for segmentation. After analysis, we did not find a good split using this variable.

Model Developed on ORIGINAL_BOOKED_AMOUNT Seg 1 (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 1 (test sample): 0.269 %

Model Developed on Full Population (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 1 (test sample): 0.269 %

Model Developed on ORIGINAL_BOOKED_AMOUNT Seg 2 (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 2 (test sample): 0.263 %

Model Developed on Full Population (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 2 (test sample): 0.263 %

Function **splits**:

This function is used to variable by variable in order to find the optimal point for segmentation. This is done by running a DecisionTreeClassifier from Sklearn to each variable, using the same target that was passed on to find_segmentation_split. In order to extract from the tree the root node spliting variable. 

Parameters: Variables that will be be passed through to the decision tree classifier and the target variable.

Returns: Splits will return a dictionary with each variable name and the root node splitting variable.

Function **segmentation**:

This function goes variable by variable and calculates the GINI score of the fitted full model vs segmented model, to decide if the segmented model has better results. It is required to have run splits before, since this function needs the dictionary that was return by splits, in order to decide where to split each variable.

Parameters: Variables that will be be passed through to the decision tree classifier, the dictionary returned by splits with the optimal splitting point, and the target variable.

Returns:

ORIGINAL_BOOKED_AMOUNT: Not good for segmentation. After analysis, we did not find a good split using this variable.

Model Developed on ORIGINAL_BOOKED_AMOUNT Seg 1 (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 1 (test sample): 0.269 %

Model Developed on Full Population (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 1 (test sample): 0.269 %

Model Developed on ORIGINAL_BOOKED_AMOUNT Seg 2 (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 2 (test sample): 0.263 %

Model Developed on Full Population (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 2 (test sample): 0.263 %
