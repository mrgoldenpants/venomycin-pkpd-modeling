import numpy as np

def dCdt(t, C, ke):
    return -ke*C

#C=C0 is just the start of the concentration
#C=0, this means that the concentration is at the end and there is none left
#since there is none left the concentration is changing at 0, because theres none left