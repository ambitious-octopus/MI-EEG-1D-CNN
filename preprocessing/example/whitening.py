
import numpy as np

run1 = runs[0][0]
run2 = runs[0][1]
run3 = runs[0][2]

run1_data = run1._data
run2_data = run2._data
run3_data = run3._data

def get_variance(data):
    var = []
    for a in data:
        b = np.var(a)
        var.append(b)
    return var

var1 = np.mean(get_variance(run1_data))
var2 = np.mean(get_variance(run2_data))
var3 = np.mean(get_variance(run3_data))

