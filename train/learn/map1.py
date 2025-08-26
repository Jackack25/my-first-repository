def f(x):
    return x * x
r = map(f, [1, 2, 3, 4, 5])
print(list(r))


from functools import reduce
def fn(a, b):
    return 10 * a + b
print(reduce(fn, [1, 2, 3, 4, 5]))

def char2num(s):
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return digits[s]
print(reduce(fn, map(char2num, '13579')))