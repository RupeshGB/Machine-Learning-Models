import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default='notebook'


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) ->None:
        """		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
		      Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
		    X: (N,D) numpy array corresponding to a dataset
		
		Return:
		    None
		
		Set:
		    self.U: (N, min(N,D)) numpy array
		    self.S: (min(N,D), ) numpy array
		    self.V: (min(N,D), D) numpy array
		"""
        X_centered = X - np.mean(X, axis=0)
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.U = U
        self.S = S
        self.V = Vt  

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    K: int value for number of columns to be kept
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        if self.V is None:
            raise ValueError("PCA model has not been fitted yet.")
        
        data_centered = data - np.mean(data, axis=0)
        V_k = self.V[:K, :]  
        X_new = np.dot(data_centered, V_k.T)  
        
        return X_new

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    retained_variance: float value for amount of variance to be retained
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
		           to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        data_centered = data - np.mean(data, axis=0)
        
        total_variance = np.sum(self.S**2)
        cumulative_variance = np.cumsum(self.S**2) / total_variance
        K = np.where(cumulative_variance >= retained_variance)[0][0] + 1
        
        V_k = self.V[:K, :]
        X_new = np.dot(data_centered, V_k.T)
        
        return X_new


    def get_V(self) ->np.ndarray:
        """		
		Getter function for value of V
		"""
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
        """		
		You have to plot three different scatterplots (2d and 3d for strongest 2 features and 2d for weakest 2 features) for this function. For plotting the 2d scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later weakest) features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
		Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
		Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.
		Hint: To extract weakest features, consider the order of components returned in PCA.
		
		Args:
		    xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
		    ytrain: (N,) numpy array, the true labels
		
		Return: None
		"""
    # Fit PCA on the dataset if not already done
        self.fit(X)
    
    # Transform the data to the 2 strongest principal components
        X_transformed_strong = self.transform(X, K=2)
    
    # Transform the data to the 2 weakest principal components
    # We calculate this by using the last two components of V
        K = len(self.S)  # Total number of components
        V_weakest = self.V[-2:, :]  # Select the last two components
        X_centered = X - np.mean(X, axis=0)
        X_transformed_weak = np.dot(X_centered, V_weakest.T)
    
    # Creating DataFrame for strong features plot
        df_strong = pd.DataFrame(X_transformed_strong, columns=['Strongest Feature 1', 'Strongest Feature 2'])
        df_strong['Label'] = y.astype(str)  # Convert labels to string for coloring
    
    # Creating DataFrame for weak features plot
        df_weak = pd.DataFrame(X_transformed_weak, columns=['Weakest Feature 1', 'Weakest Feature 2'])
        df_weak['Label'] = y.astype(str)  # Convert labels to string for coloring
    
    # Strongest Features 2D Plot
        fig_strong_2d = px.scatter(df_strong, x='Strongest Feature 1', y='Strongest Feature 2', color='Label', title=f'{fig_title} - 2D Strongest Features')
        fig_strong_2d.show()

    # Weakest Features 2D Plot
        fig_weak_2d = px.scatter(df_weak, x='Weakest Feature 1', y='Weakest Feature 2', color='Label', title=f'{fig_title} - 2D Weakest Features')
        fig_weak_2d.show()

    # For 3D plot of the strongest features, transform the data to 3 principal components first
        X_transformed_3D_strong = self.transform(X, K=3)
        df_strong_3D = pd.DataFrame(X_transformed_3D_strong, columns=['Strongest Feature 1', 'Strongest Feature 2', 'Strongest Feature 3'])
        df_strong_3D['Label'] = y.astype(str)  # Convert labels to string for coloring

    # Strongest Features 3D Plot
        fig_strong_3d = px.scatter_3d(df_strong_3D, x='Strongest Feature 1', y='Strongest Feature 2', z='Strongest Feature 3', color='Label', title=f'{fig_title} - 3D Strongest Features')
        fig_strong_3d.show()

        
    
		 
