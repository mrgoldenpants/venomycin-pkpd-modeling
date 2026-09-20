def validate_initial_guess(theta):
    CL, Q, V1, V2 = theta
    if not (.05 <= CL <= 15):
        return False
    if not (3 <= V1 <= 20):
        return False

    return True

initial_guess = [5, 2, 22, 20]

assert not validate_initial_guess(initial_guess)
