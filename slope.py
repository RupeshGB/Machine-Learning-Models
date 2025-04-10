import numpy as np
from pca import PCA
from regression import Regression


class Slope(object):

    def __init__(self):
        pass

    @staticmethod
    def pca_slope(X, y):
        """		
		Calculates the slope of the first principal component given by PCA
		
		Args:
		    x: N x 1 array of feature x
		    y: N x 1 array of feature y
		Return:
		    slope: (float) scalar slope of the first principal component
		"""
        data = np.hstack((X.reshape(-1, 1), y.reshape(-1, 1)))
        
        pca = PCA()
        pca.fit(data)
        first_component = pca.V[0, :]
        
        slope = first_component[1] / first_component[0]
        
        return slope


    @staticmethod
    def lr_slope(X, y):
        """		
		Calculates the slope of the best fit returned by linear_fit_closed()
		
		For this function don't use any regularization
		
		Args:
		    X: N x 1 array corresponding to a dataset
		    y: N x 1 array of labels y
		Return:
		    slope: (float) slope of the best fit
		"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Adding a bias term to X
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Initialize the Regression class and fit the model
        reg = Regression()
        theta = reg.linear_fit_closed(X_bias, y)
        
        # The slope is the second element in the theta vector, as the first element is the intercept
        slope = theta[1, 0]
        
        return slope

    @classmethod
    def addNoise(cls, c, x_noise=False, seed=1):
        """		
		Creates a dataset with noise and calculates the slope of the dataset
		using the pca_slope and lr_slope functions implemented in this class.
		
		Args:
		    c: (float) scalar, a given noise level to be used on Y and/or X
		    x_noise: (Boolean) When set to False, X should not have noise added
		            When set to True, X should have noise.
		            Note that the noise added to X should be different from the
		            noise added to Y. You should NOT use the same noise you add
		            to Y here.
		    seed: (int) Random seed
		Return:
		    pca_slope_value: (float) slope value of dataset created using pca_slope
		    lr_slope_value: (float) slope value of dataset created using lr_slope
		"""
        np.random.seed(seed)
        
        # Generate synthetic dataset
        X = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 data points between -5 and 5
        true_slope = 2  # Example slope
        b = 1  # Example intercept
        Y = true_slope * X + b
        
        # Add noise to Y
        Y_noise = Y + np.random.normal(0, c, Y.shape)
        
        # Optionally add noise to X
        if x_noise:
            X_noise = X + np.random.normal(0, c, X.shape)
        else:
            X_noise = X
        
        # Flatten X_noise for compatibility with pca_slope and lr_slope, which expect 1D arrays
        X_noise = X_noise.flatten()
        Y_noise = Y_noise.flatten()
        
        # Calculate slopes using pca_slope and lr_slope
        pca_slope_value = cls.pca_slope(X_noise, Y_noise)
        lr_slope_value = cls.lr_slope(X_noise.reshape(-1, 1), Y_noise.reshape(-1, 1))  # lr_slope might expect 2D array for X
        
        return pca_slope_value, lr_slope_value
