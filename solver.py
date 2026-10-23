def thomas_algorithm(d, a, b, c):   #in order to calculate the inverse of matrix with time complexity O(n)
    n = len(b)
    b = b.copy()  #in order to avoid modification to variable we need to use later
    a = a.copy()
    d = d.copy()
    for i in range(1, n):
        b[i] -= a[i - 1]/b[i - 1] * c[i - 1]
        d[i] -= a[i - 1]/b[i - 1] * d[i - 1]
    x = [0] * (n - 1) + [d[-1]/b[-1]]
    for i in range(2, n + 1):
        x[n - i] = (d[n - i] - c[n - i] * x[n - (i - 1)])/b[n - i]

    return x