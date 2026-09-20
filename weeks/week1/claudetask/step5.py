import numpy as np
#the only variable in half life is literally ke
def half_life(ke):
    return np.log(2)/ke

ke = .1
print(half_life(ke), "hours")


