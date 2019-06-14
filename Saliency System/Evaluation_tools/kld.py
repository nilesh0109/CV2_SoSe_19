import numpy as np
from scipy.special import softmax

# Input: two 2D floating point numpy arrays of the same shape.
# The inputs will be interpreted as probability mass functions and the KL divergence is returned.
def KLD(P, G):
	if P.ndim != 2:
		raise ValueError("Expected P to be 2 dimensional array")
	if G.ndim != 2:
		raise ValueError("Expected G to be 2 dimensional array")
	if P.shape != G.shape:
		raise ValueError('The shape of P: {} must match the shape of G: {}'.format(P.shape, G.shape))
	if np.any(P<0):
		raise ValueError('P has some negative values')
	if np.any(G<0):
		raise ValueError('G has some negative values')

	# Normalize P and G using softmax
	p_n = softmax(P)
	g_n = softmax(G)

	EPS = 1e-16 # small regularization constant for numerical stability
	kl = np.sum(g_n * np.log2( EPS + (g_n / (EPS + p_n) ) ))

	return kl



# Example usage
P = np.random.rand(120, 320)
G = np.random.rand(120, 320)
k = KLD(P,G)
print('KL divergence of random images: {:f}'.format(k))

# Final evaluation criterion will be KLD averaged over all images in the test set.