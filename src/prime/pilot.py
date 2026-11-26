import math
from statistics import NormalDist


def n_per_group(sigma, mde, alpha=0.05, power=0.8, comparisons=1):
    """Сколько клиентов нужно в каждой из двух равных групп."""
    z_alpha = NormalDist().inv_cdf(1 - alpha / (2 * comparisons))
    z_power = NormalDist().inv_cdf(power)
    return math.ceil(2 * ((z_alpha + z_power) * sigma / mde) ** 2)


def mde_for_n(sigma, n, alpha=0.05, power=0.8, comparisons=1):
    """Минимальный обнаруживаемый эффект при n клиентах на группу."""
    z_alpha = NormalDist().inv_cdf(1 - alpha / (2 * comparisons))
    z_power = NormalDist().inv_cdf(power)
    return (z_alpha + z_power) * sigma * math.sqrt(2 / n)


def diff_ci(t, c, alpha=0.05):
    """Разность средних, интервал и двустороннее p по нормальному приближению."""
    diff = t.mean() - c.mean()
    se = math.sqrt(t.var() / len(t) + c.var() / len(c))
    margin = NormalDist().inv_cdf(1 - alpha / 2) * se
    if se == 0:
        p = float(diff == 0)
    else:
        p = math.erfc(abs(diff) / (se * math.sqrt(2)))
    return {"diff": diff, "lo": diff - margin, "hi": diff + margin, "p": p}


def srm_pvalue(sizes):
    """Проверка равных долей для двух или трёх групп."""
    expected = sum(sizes) / len(sizes)
    chi2 = sum((size - expected) ** 2 / expected for size in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    if len(sizes) == 3:
        return math.exp(-chi2 / 2)
    raise ValueError("Нужно передать размеры двух или трёх групп")


def cuped(y, x):
    """Скорректированная метрика и коэффициент предпериода."""
    variance = x.var()
    theta = y.cov(x) / variance if variance > 0 else 0.0
    return y - theta * (x - x.mean()), theta
