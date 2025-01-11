
# %%
from numpy import *
from matplotlib.pyplot import *
import util
Si = util.sqrtm(array([[3,2],[2,4]]))
x = dot(random.randn(1000,2), Si)
plot(x[:,0], x[:,1], 'b.')
show(False)
dot(x.T,x) / real(x.shape[0]-1) # The sample covariance matrix. Random generated data cause result to vary
array([[ 3.01879339,  2.07256783],
       [ 2.07256783,  4.15089407]])

# %%