# split_kde
Split data based on their distance using scikit-learn KernelDensity as alternative for one dimensional clustering

## example

example:

```python
import numpy as np
from sklearn.neighbors import KernelDensity

x = np.array([10,11,9,23,21,11,45,20,11,12]).reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=3)
kde = split_kde(x, start_end=(0,100))

print(x)
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

to force predifined number of groups (auto search bandwidth size), use `n_groups`:

```
kde = KernelDensity(kernel='gaussian', bandwidth=3)
kde = split_kde(x, model = kde, n_groups = 4, max_iter=1000)
```
