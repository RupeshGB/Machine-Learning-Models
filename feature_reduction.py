import math
import sys
from typing import List
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class FeatureReduction(object):

    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data: pd.DataFrame, target: pd.Series,
        significance_levels: List[float]=[0.01, 0.1, 0.2]) ->dict:
        """		
		Args:
		    data: (pandas data frame) contains the feature matrix
		    target: (pandas series) represents target feature to search to generate significant features
		    significance_levels: (list) thresholds to reject the null hypothesis
		Return:
		    significance_level_feature_map: (python map) contains significant features for each significance_level.
		    The key will be the significance level, for example, 0.01, 0.1, or 0.2. The values associated with the keys would be
		    equal features that has p-values less than the significance level.
		"""
        significant_features_map = {level: [] for level in significance_levels}
        remaining_features = list(data.columns)
        selected_features = []

        while remaining_features:
            pvals = []
            for feature in remaining_features:
                # Try adding each remaining feature to the selected features and fit a model
                test_features = selected_features + [feature]
                X_test = sm.add_constant(data[test_features])
                model = sm.OLS(target, X_test).fit()
                # Store the p-value of the added feature
                pvals.append((feature, model.pvalues[feature]))

            # Choose the feature with the smallest p-value
            pvals.sort(key=lambda x: x[1])
            best_feature, best_pval = pvals[0]

            # Check if the best feature's p-value is below the highest significance level
            if best_pval < max(significance_levels):
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                # Update the map based on the new p-values
                for level in significance_levels:
                    if best_pval < level:
                        significant_features_map[level].append(best_feature)
            else:
                # If no remaining features are significant at the highest level, stop the process
                break

        return significant_features_map

    @staticmethod
    def backward_elimination(data: pd.DataFrame, target: pd.Series,
        significance_levels: List[float]=[0.01, 0.1, 0.2]) ->dict:
        """		
		Args:
		    data: (pandas data frame) contains the feature matrix
		    target: (pandas series) represents target feature to search to generate significant features
		    significance_levels: (list) thresholds to reject the null hypothesis
		Return:
		    significance_level_feature_map: (python map) contains significant features for each significance_level.
		    The key will be the significance level, for example, 0.01, 0.1, or 0.2. The values associated with the keys would be
		    equal features that has p-values less than the significance level.
		"""
        significant_features_map = {level: [] for level in significance_levels}
        features = list(data.columns)
        
        # Initially, consider all features significant at all levels
        for level in significance_levels:
            significant_features_map[level] = features.copy()

        while features:
            # Fit model with current features
            X = sm.add_constant(data[features])
            model = sm.OLS(target, X).fit()
            pvalues = model.pvalues[1:]  # Exclude intercept

            # Check if the max p-value is greater than the highest significance level
            if pvalues.max() > max(significance_levels):
                # Find the feature with the highest p-value
                worst_feature = pvalues.idxmax()
                features.remove(worst_feature)

                # Update the map for each significance level
                for level in significance_levels:
                    # Remove feature if its p-value is above the significance level
                    if pvalues[worst_feature] > level:
                        significant_features_map[level].remove(worst_feature)
            else:
                # If all features are significant at the highest level, stop the process
                break

        return significant_features_map


    def evaluate_features(data: pd.DataFrame, y: pd.Series,
        significance_level_feature_map: dict) ->None:
        """
        PROVIDED TO STUDENTS

        Performs linear regression on the dataset only using the features discovered by feature reduction for each significance level.

        Args:
            data: (pandas data frame) contains the feature matrix
            y: (pandas series) output labels
            significance_level_feature_map: (python map) contains significant features for each significance_level. Each feature name is a string
        """
        min_rmse = sys.maxsize
        min_significance_level = 0
        for significance_level, features in significance_level_feature_map.items(
            ):
            removed_features = set(data.columns.tolist()) - set(features)
            print(
                f'significance level: {significance_level}, Removed features: {removed_features}'
                )
            data_curr_features = data[features]
            x_train, x_test, y_train, y_test = train_test_split(
                data_curr_features, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            print(f'significance level: {significance_level}, RMSE: {rmse}')
            if min_rmse > rmse:
                min_rmse = rmse
                min_significance_level = significance_level
        print(
            f'Best significance level: {min_significance_level}, RMSE: {min_rmse}'
            )
        print('')
