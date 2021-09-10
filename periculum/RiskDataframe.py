import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, RobustScaler
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


class RiskDataframe(pd.DataFrame):
    """
    The class is used to extend the properties of Dataframes to a particular
    type of Dataframes in the Risk Industry.
    It provides the end user with both general and specific cleaning functions, 
    though they never reference a specific VARIABLE NAME.
    
    It facilitates the End User to perform some Date Feature Engineering,
    Scaling, Encoding, etc. to avoid code repetition.
    """

    # Initializing the inherited pd.DataFrame
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cat_columns = []
        self.num_columns = []
        self.date_columns = []
        self.mnra_columns = 'Must run the missing_not_at_random method to update this attribute'
        self.full_file_segment_variable = 'Must run the missing_not_at_random method to update this attribute'
        self.thin_file_segment_variable = 'Must run the missing_not_at_random method to update this attribute'
        self.GINI_fitted_full_model = 'Must run the find_segmentation_split method to update this attribute'
        self.accuracy_fitted_full_model = 'Must run the find_segmentation_split method to update this attribute'
        self.variable_split = 'Must run the find_segment_split method to update this attribute'
        self.date_types()

    @property
    def _constructor(self):
        def func_(*args, **kwargs):
            df = RiskDataframe(*args, **kwargs)
            return df

        return func_

    # -----------------------------------------------------------------------------
    # DATA HANDLING
    # -----------------------------------------------------------------------------

    def SetAttributes(self, kwargs):
        """
        The function will update the type of the variable submitted for change.
        It will veify first that the key is present in the desired dataframe.
        If present, it will try to change the type to the desired format.
        If not possible, it will continue to the next element.
        Parameters
        ----------
        **kwargs : The key-argument pair of field-type relationship that
        wants to be updated.
        Returns
        -------
        None.
        """
        if self.shape[0] > 0:
            for key, vartype in kwargs.items():
                if key in self.columns:
                    try:
                        self[key] = self[key].astype(vartype)
                    except:
                        print("Undefined type {}".format(str(vartype)))
                else:
                    print("The dataframe does not contain variable {}.".format(str(key)))
        else:
            print("The dataframe has not yet been initialized")

    def date_types(self):
        """
        This method is used as a way to create a list for each column types,
        in order to use these lists in further methods that will be called.
        Since this method is necessary for the other methods in the class,
        it is called when the class is instantiated.
        """
        for column in self.columns:
            if self[column].dtype == 'O':
                self.cat_columns.append(column)
            elif self[column].dtype == 'float64':
                self.num_columns.append(column)
            elif self[column].dtype == 'int64':
                self.num_columns.append(column)
            elif self[column].dtype == '<M8[ns]':
                self.date_columns.append(column)
            else:
                None

    def date_to_int(self, reporting_date):
        """
        This is an optional method in case the data in the dataframe has date columns,
        this method will convert all the dates to year fraction from the reporting date
        to calculate the time difference. Which is necessary for the application of the
        segmentation method, because Sklearn logistic regression does not accept date type
        values.
        ----------
        reporting_date : This variable is a datetime object that will be the point in time
        where all of the timedelta's will be calculated from.
        -------
        """
        for column in self.columns:
            if self[column].dtype == '<M8[ns]':
                self[column] = abs(self[column] - reporting_date).astype('timedelta64[D]')
                self[column] = round(self[column] / 365, 2)
            else:
                pass
        self.num_columns = self.num_columns + self.date_columns
        self.date_columns = "None due to date_to_int method being called before."

    # -----------------------------------------------------------------------------
    # RISK BASED APPROACH
    # -----------------------------------------------------------------------------
    def missing_not_at_random(self, corr_threshold=0.9):
        """
        This method is checks for the correlation between the missing values in all the columns, pair by
        pair in order, in order to see if the correlation is higher than threshold to be considered missing
        not at random.
        -------
        corr_threshold: This variable is the threshold that will be used as a cut off to decide if the
        correlation between the missing values between a pair of columns is high enough to be considered missing
        not at random.
        """

        def redundant_pairs(self):
            """
            This function inside the method is used to find pairs of columns that are
            are repeated in the correlation matrix used in the method missing_not_at_random.
            """
            pairs_to_drop = set()
            cols = self.columns
            for i in range(0, self.shape[1]):
                for j in range(0, i + 1):
                    pairs_to_drop.add((cols[i], cols[j]))
            return pairs_to_drop

        NaS = self.iloc[:, [i for i, n in enumerate(np.var(self.isna(), axis='rows')) if n > 0]]
        labels_to_drop = redundant_pairs(NaS)
        NaS_df = NaS.isnull().corr().unstack()
        NaS_corr = NaS_df.drop(labels=labels_to_drop).sort_values(ascending=False)
        mnra_list = []

        for i in range(len(NaS_corr)):
            if (NaS_corr[i] >= corr_threshold):
                mnra_list.append(NaS_corr.index[i])
            else:
                pass
        mnra_columns = list(set([item for sublist in mnra_list for item in sublist]))
        full_file_segment_variable = self.num_columns + self.cat_columns
        thin_file_segment_variable = [x for x in full_file_segment_variable if x not in mnra_columns]

        self.mnra_columns = mnra_columns
        self.full_file_segment_variable = full_file_segment_variable
        self.thin_file_segment_variable = thin_file_segment_variable

        if len(mnra_columns) > 0:
            print(
                f'Missing Not At Random Repport - {mnra_columns} variables seem Missing Not at Random,there for we recommend:')
            print(f'Thin File Segment Variables: {thin_file_segment_variable}')
            print(f'Full File Segment Variables: {full_file_segment_variable}')
        else:
            print(
                'There are no missing not at random variables in the data. You can try another threshold if you would like to')

    def find_segment_split(self, target='', robust_scaler=''):
        """
        This method finds if the data in each column performs better if it is split in order to segment the data
        and have a better model fit. The model used is logistic regression for a binary classification, which does
        not accept alphanumeric values, therefore labelencoder is automatically called if the method detects these
        data type columns. The required argument for this method is target, since the logistic regression model needs
        this. Robust_scaler is an optional argument in order to enhance model performance. Once the baseline model with
        the full file without segmentation is calculated, this method continues to find where is the optimal place
        for spltting each column by applying a decision tree classifier, and extracting the root node splitting point.
        Finally it fits a model on the segmented dataset and compares the results of both models.

        Returns
        -------
        Example 1:
        ORIGINAL_BOOKED_AMOUNT: Not good for segmentation. Afer analysis, we did not find a good split using this variable.
        Model Developed on ORIGINAL_BOOKED_AMOUNT Seg 1 (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 1 (test sample): 0.269 %
        Model Developed on Full Population (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 1 (test sample): 0.269 %
        Model Developed on ORIGINAL_BOOKED_AMOUNT Seg 2 (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 2 (test sample): 0.263 %
        Model Developed on Full Population (train sample) applied on ORIGINAL_BOOKED_AMOUNT Seg 2 (test sample): 0.263 %
                
        """

        if len(self.cat_columns) > 0:
            df_cat = self[self.cat_columns]
            for column in range(len(self.cat_columns)):
                df_cat[self.cat_columns[column]] = LabelEncoder().fit_transform(df_cat[self.cat_columns[column]])
            self.drop(self.cat_columns, inplace=True, axis=1)
            for col in df_cat.columns:
                self[col] = df_cat[col]
        else:
            pass

        if robust_scaler.upper() == 'YES':
            non_target_df = self.drop(target, axis=1)
            scaled_features = RobustScaler().fit_transform(non_target_df.values)
            scaled_df = pd.DataFrame(scaled_features, index=non_target_df.index, columns=non_target_df.columns)
            self.drop(scaled_df.columns, inplace=True, axis=1)
            for col in scaled_df.columns:
                self[col] = scaled_df[col]
        else:
            pass

        # Baseline model
        df_train, df_test = train_test_split(self, test_size=0.2, random_state=42)
        try:
            self.num_columns.remove(target)
        except:
            self.cat_columns.remove(target)
        X_train = df_train.drop(target, axis=1)
        y_train = df_train[target]
        X_test = df_test.drop(target, axis=1)
        y_test = df_test[target]
        method = LogisticRegression(random_state=0, solver='lbfgs', max_iter=100)
        fitted_full_model = method.fit(X_train, y_train)
        y_pred_proba = fitted_full_model.predict_proba(X_test)[:, 0]
        y_pred = fitted_full_model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        self.GINI_fitted_full_model = abs((2 * roc_auc) - 1)
        self.accuracy_fitted_full_model = accuracy_score(y_test, y_pred)

        # Function to decide where to split
        all_columns = self.num_columns + self.cat_columns
        split_list = []
        def splits(column):
            x = self.drop(target, axis=1)
            y = self[target]
            single_x = np.array(x[column]).reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(single_x, y, test_size=0.2, random_state=42)
            method = DecisionTreeClassifier(random_state=0, max_depth=3)
            individual_feature_model = method.fit(X_train, y_train)
            y_pred = individual_feature_model.predict(X_test)
            split = str(tree.export_text(individual_feature_model))
            split = float(split[17:23])
            split_list.append(split)
            return split_list
        np.vectorize(splits, otypes=[list])(all_columns)
        self.variable_split = dict(zip(all_columns, split_list))

        # Function to decide if good segmentation loop
        def segmentation(column, split):
            df_train_seg1 = df_train[self[column] > split]
            df_train_seg2 = df_train[self[column] <= split]
            df_test_seg1 = df_test[self[column] > split]
            df_test_seg2 = df_test[self[column] <= split]

            X_train_seg1 = df_train_seg1[all_columns]
            y_train_seg1 = df_train_seg1[target]
            X_test_seg1 = df_test_seg1[all_columns]
            y_test_seg1 = df_test_seg1[target]

            fitted_model_seg1 = method.fit(X_train_seg1, y_train_seg1)
            y_pred_seg1 = fitted_model_seg1.predict_proba(X_test_seg1)[:, 1]
            y_pred_seg1_fullmodel = fitted_full_model.predict_proba(X_test_seg1)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test_seg1, y_pred_seg1)
            roc_auc = auc(fpr, tpr)
            GINI_seg1 = round(abs((2 * roc_auc) - 1),3)

            fpr, tpr, thresholds = roc_curve(y_test_seg1, y_pred_seg1_fullmodel)
            roc_auc = auc(fpr, tpr)
            GINI_seg1_full = round(abs((2 * roc_auc) - 1),3)

            X_train_seg2 = df_train_seg2[all_columns]
            y_train_seg2 = df_train_seg2[target]
            X_test_seg2 = df_test_seg2[all_columns]
            y_test_seg2 = df_test_seg2[target]

            fitted_model_seg2 = method.fit(X_train_seg2, y_train_seg2)
            y_pred_seg2 = fitted_model_seg2.predict_proba(X_test_seg2)[:, 1]
            y_pred_seg2_fullmodel = fitted_full_model.predict_proba(X_test_seg2)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test_seg2, y_pred_seg2)
            roc_auc = auc(fpr, tpr)
            GINI_seg2 = round(abs((2 * roc_auc) - 1),3)

            fpr, tpr, thresholds = roc_curve(y_test_seg2, y_pred_seg2_fullmodel)
            roc_auc = auc(fpr, tpr)
            GINI_seg2_full = round(abs((2 * roc_auc) - 1),3)

            if GINI_seg1 > GINI_seg1_full and GINI_seg2 > GINI_seg2_full:
                print(f'{column}: Good for segmentation.')
                print(f'Segment1: {column} > {split} [GINI Full Model: {GINI_seg1_full}% / GINI Segmented Model: {GINI_seg1}')
                print(f'Segment2: {column} > {split} [GINI Full Model: {GINI_seg2_full}% / GINI Segmented Model: {GINI_seg2}')
            else:
                print(f'{column}: Not good for segmentation. Afer analysis, we did not find a good split using this variable.')

            print(f"Model Developed on {column} Seg 1 (train sample) applied on {column} Seg 1 (test sample):",
                  GINI_seg1,'%')
            print(f"Model Developed on Full Population (train sample) applied on {column} Seg 1 (test sample):",
                  GINI_seg1_full,'%')
            print(f"Model Developed on {column} Seg 2 (train sample) applied on {column} Seg 2 (test sample):",
                  GINI_seg2,'%')
            print(f"Model Developed on Full Population (train sample) applied on {column} Seg 2 (test sample):",
                  GINI_seg2_full,'%')
        np.vectorize(segmentation, otypes=[list])(all_columns, split_list)