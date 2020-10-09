# split_kde
Split data based on their distance using scikit-learn KernelDensity as alternative for one dimensional clustering

## example

example:

```python
import numpy as np
from sklearn.neighbors import KernelDensity

a = np.array([10,11,9,23,21,11,45,20,11,12]).reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=3)
kde = split_kde(a, start_end=(0,100))

print(a)
print(kde.splitted_)
print(kde.labels_)
```

result:

```
[[10]
 [11]
 [ 9]
 [23]
 [21]
 [11]
 [45]
 [20]
 [11]
 [12]]
[array([10, 11,  9, 11, 11, 12]), array([23, 21, 20]), array([45])]
[0 0 0 1 1 0 2 1 0 0]
```

plot:

```python
from matplotlib.pyplot import plot
plot(kde.s_, kde.e_)
```
