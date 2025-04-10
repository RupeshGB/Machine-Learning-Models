from typing import Tuple
import numpy as np


class ImgCompression(object):

    def __init__(self):
        pass

    def svd(self, X: np.ndarray) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """		
		Do SVD. You could use numpy SVD.
		Your function should be able to handle black and white
		images ((N,D) arrays) as well as color images ((3,N,D) arrays)
		In the image compression, we assume that each column of the image is a feature. Perform SVD on the channels of
		each image (1 channel for black and white and 3 channels for RGB)
		Image is the matrix X.
		
		Hint: np.linalg.svd by default returns the transpose of V. We want you to return the transpose of V, not V.
		
		Args:
		    X: (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images
		
		Return:
		    U: (N,N) numpy array for black and white images / (3,N,N) numpy array for color images
		    S: (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
		    V^T: (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
		"""
        if X.ndim == 2:  # Grayscale image case
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            return U, S, VT
        elif X.ndim == 3:  # Color image case
            channels, N, D = X.shape
            U = np.empty((channels, N, N))
            S = np.empty((channels, min(N, D)))
            VT = np.empty((channels, D, D))
            for i in range(channels):
                Ui, Si, Vti = np.linalg.svd(X[i], full_matrices=False)
                U[i, :, :] = Ui
                S[i, :len(Si)] = Si
                VT[i, :, :] = Vti
            return U, S, VT
       
        
        
    def compress(self, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
        ) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """		
		Compress the SVD factorization by keeping only the first k components
		
		Args:
		    U (np.ndarray): (N,N) numpy array for black and white simages / (3,N,N) numpy array for color images
		    S (np.ndarray): (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
		    V (np.ndarray): (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
		    k (int): int corresponding to number of components to keep
		
		Returns:
		    Tuple[np.ndarray, np.ndarray, np.ndarray]:
		        U_compressed: (N, k) numpy array for black and white images / (3, N, k) numpy array for color images
		        S_compressed: (k, ) numpy array for black and white images / (3, k) numpy array for color images
		        V_compressed: (k, D) numpy array for black and white images / (3, k, D) numpy array for color images
		"""
        if U.ndim == 2:  # B&W image case
            U_compressed = U[:, :k]
            S_compressed = S[:k]
            V_compressed = V[:k, :]
        elif U.ndim == 3:  # Color image case
            channels = U.shape[0]
            U_compressed = np.empty((channels, U.shape[1], k))
            S_compressed = np.empty((channels, k))
            V_compressed = np.empty((channels, k, V.shape[2]))
            for i in range(channels):
                U_compressed[i] = U[i, :, :k]
                S_compressed[i] = S[i, :k]
                V_compressed[i] = V[i, :k, :]
        else:
            raise ValueError("U must be a 2D or 3D numpy array")

        return U_compressed, S_compressed, V_compressed

    def rebuild_svd(self, U_compressed: np.ndarray, S_compressed: np.
        ndarray, V_compressed: np.ndarray) ->np.ndarray:
        """		
		Rebuild original matrix X from U, S, and V which have been compressed to k componments.
		
		Args:
		    U_compressed: (N,k) numpy array for black and white images / (3,N,k) numpy array for color images
		    S_compressed: (k, ) numpy array for black and white images / (3,k) numpy array for color images
		    V_compressed: (k,D) numpy array for black and white images / (3,k,D) numpy array for color images
		
		Return:
		    Xrebuild: (N,D) numpy array of reconstructed image / (3,N,D) numpy array for color images
		
		Hint: numpy.matmul may be helpful for reconstructing color images
		"""
        if U_compressed.ndim == 2:  
            S_mat = np.diag(S_compressed)
            X_rebuild = np.dot(U_compressed, np.dot(S_mat, V_compressed))
        elif U_compressed.ndim == 3:  
            channels = U_compressed.shape[0]
            N, k = U_compressed.shape[1], U_compressed.shape[2]
            D = V_compressed.shape[2]
            X_rebuild = np.empty((channels, N, D))
            for i in range(channels):
                S_mat = np.diag(S_compressed[i])
                X_rebuild[i] = np.dot(U_compressed[i], np.dot(S_mat, V_compressed[i]))
        else:
            raise ValueError("U_compressed must be a 2D or 3D numpy array")

        return X_rebuild

    def compression_ratio(self, X: np.ndarray, k: int) ->float:
        """		
		Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)
		Refer to https://timbaumann.info/svd-image-compression-demo/
		Args:
		    X: (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images
		    k: int corresponding to number of components
		
		Return:
		    compression_ratio: float of proportion of storage used by compressed image
		"""
        if X.ndim == 2:  # Black and white image case
            N, D = X.shape
            original_values = N * D
            compressed_values = k * (N + D + 1)
        elif X.ndim == 3:  # Color image case
            _, N, D = X.shape
            original_values = 3 * N * D
            compressed_values = 3 * k * (N + D + 1)
        else:
            raise ValueError("X must be a 2D or 3D numpy array")
        
        return compressed_values / original_values


    def recovered_variance_proportion(self, S: np.ndarray, k: int) ->float:
        """		
		Compute the proportion of the variance in the original matrix recovered by a rank-k approximation
		
		Args:
		   S: (min(N,D), ) numpy array black and white images / (3,min(N,D)) numpy array for color images
		   k: int, rank of approximation
		
		Return:
		   recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
		"""
        if S.ndim == 1: 
            total_variance = np.sum(S**2)
            recovered_variance = np.sum(S[:k]**2)
            recovered_var_proportion = recovered_variance / total_variance
        elif S.ndim == 2:  
            recovered_var_proportion = np.zeros(S.shape[0])
            for i in range(S.shape[0]):  
                total_variance = np.sum(S[i]**2)
                recovered_variance = np.sum(S[i][:k]**2)
                recovered_var_proportion[i] = recovered_variance / total_variance
        else:
            raise ValueError("S must be a 1D or 2D numpy array")

        return recovered_var_proportion


    def memory_savings(self, X: np.ndarray, U: np.ndarray, S: np.ndarray, V:
        np.ndarray, k: int) ->Tuple[int, int, int]:
        """
        PROVIDED TO STUDENTS

        Returns the memory required to store the original image X and
        the memory required to store the compressed SVD factorization of X

        Args:
            X (np.ndarray): (N,D) numpy array corresponding to black and white images / (3,N,D) numpy array for color images
            U (np.ndarray): (N,N) numpy array for black and white simages / (3,N,N) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (3,min(N,D)) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (3,D,D) numpy array for color images
            k (int): integer number of components

        Returns:
            Tuple[int, int, int]:
                original_nbytes: number of bytes that numpy uses to represent X
                compressed_nbytes: number of bytes that numpy uses to represent U_compressed, S_compressed, and V_compressed
                savings: difference in number of bytes required to represent X
        """
        original_nbytes = X.nbytes
        U_compressed, S_compressed, V_compressed = self.compress(U, S, V, k)
        compressed_nbytes = (U_compressed.nbytes + S_compressed.nbytes +
            V_compressed.nbytes)
        savings = original_nbytes - compressed_nbytes
        return original_nbytes, compressed_nbytes, savings

    def nbytes_to_string(self, nbytes: int, ndigits: int=3) ->str:
        """
        PROVIDED TO STUDENTS

        Helper function to convert number of bytes to a readable string

        Args:
            nbytes (int): number of bytes
            ndigits (int): number of digits to round to

        Returns:
            str: string representing the number of bytes
        """
        if nbytes == 0:
            return '0B'
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        scale = 1024
        units_idx = 0
        n = nbytes
        while n > scale:
            n = n / scale
            units_idx += 1
        return f'{round(n, ndigits)} {units[units_idx]}'
