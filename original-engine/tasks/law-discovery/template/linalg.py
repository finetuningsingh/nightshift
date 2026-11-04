"""Small pure-Python linear algebra helpers. Read-only; standard library only.

    solve(a, b)                      Gaussian elimination with partial pivoting
    lstsq(design, ys, ridge=0.0)     least squares by the normal equations
    polyfit(ts, ys, degree, ridge)   polynomial in the scaled time u = (t - 2) / 2, returns predict(t)
"""

CENTER = 2.0
SCALE = 2.0


def solve(a, b):
    """Solve ``a x = b`` for a square matrix given as a list of rows. Raises ValueError if it is singular."""
    n = len(b)
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            if factor:
                for k in range(col, n + 1):
                    m[r][k] -= factor * m[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][k] * x[k] for k in range(i + 1, n))) / m[i][i]
    return x


def lstsq(design, ys, ridge=0.0):
    """Coefficients ``c`` minimising ``|design c - ys|^2 + ridge |c|^2`` (``design`` is a list of feature rows)."""
    p = len(design[0])
    gram = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for row, y in zip(design, ys):
        for i in range(p):
            rhs[i] += row[i] * y
            gi = gram[i]
            for j in range(i, p):
                gi[j] += row[i] * row[j]
    for i in range(p):
        for j in range(i):
            gram[i][j] = gram[j][i]
        gram[i][i] += ridge
    return solve(gram, rhs)


def polyfit(ts, ys, degree, ridge=0.0):
    """Fit a degree-``degree`` polynomial in ``u = (t - 2) / 2`` and return a function mapping t to its prediction."""
    def powers(t):
        u = (t - CENTER) / SCALE
        return [u ** k for k in range(degree + 1)]

    coef = lstsq([powers(t) for t in ts], ys, ridge)
    return lambda t: sum(c * f for c, f in zip(coef, powers(t)))
