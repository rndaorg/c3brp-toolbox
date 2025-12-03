
def df(f, x, h=1e-8):
    """Numerical derivative of f."""
    return (f(x + h) - f(x - h)) / (2 * h)


def newton_raphson(f, x0, x_min=None, x_max=None, max_iter=100, tol=1e-12):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        fpx = df(f, x)
        if fpx == 0:
            break
        x_new = x - fx / fpx
        # Enforce bounds if provided
        if x_min is not None and x_new < x_min:
            x_new = (x + x_min) / 2
        if x_max is not None and x_new > x_max:
            x_new = (x + x_max) / 2
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    # If Newton fails, fall back to bisection if bracket provided
    if x_min is not None and x_max is not None:
        return bisection(f, x_min, x_max)
    else:
        raise RuntimeError("Newton-Raphson failed to converge")



def bisection(f, a, b, max_iter=200, tol=1e-12):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise RuntimeError(f"Bisection failed: f(a)={fa}, f(b)={fb} have same sign")
    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2.0
    
