import numpy as np
from kld import KLD
import imageio

#x = np.random.random_sample((5,5)) * 10
NUM_IMAGES = 1600
path_to_fixations = 'images/fixations'
path_to_saliency = 'images/saliency/predicted_saliency'
log = open('images/KL_log','a+')

def get_pmf(arr):
    pmf = np.zeros_like(arr, dtype=np.float32)
    m, n = arr.shape
    val, bin = np.histogram(arr, bins=np.arange(257))

    for i in range(m):
        for j in range(n):
            pmf[i,j] = val[int(arr[i,j])] / (m * n)
    return pmf

kld_score = np.zeros(NUM_IMAGES)

for ind in range(1,NUM_IMAGES+1):
    P = imageio.imread('{:s}/{:04d}.jpg'.format(path_to_fixations, ind))
    G = imageio.imread('{:s}/{:04d}.jpg'.format(path_to_saliency, ind))
    kld_score[ind-1] = KLD(get_pmf(G), get_pmf(P))
    if ind % 100 == 0:
        print('Evaluating index {:d}...'.format(ind))

print('KL divergence of the %d images is %f '% (NUM_IMAGES, np.mean(kld_score)))

log.write('\nKL divergence %d images is %f '% (NUM_IMAGES, np.mean(kld_score)))
log.write('For %s' % (path_to_saliency))
log.close()


