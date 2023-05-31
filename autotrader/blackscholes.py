from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np

N = norm.cdf
binary_flag = {"c": 1, "p": -1}


def pdf(x):
    """the probability density function"""
    one_over_sqrt_two_pi = 0.3989422804014326779399460599343818684758586311649
    return one_over_sqrt_two_pi * np.exp(-0.5 * x * x)


def d1(S, K, t, r, sigma):
    sigma_squared = sigma * sigma
    numerator = np.log(S / float(K)) + (r + sigma_squared / 2.0) * t
    denominator = sigma * np.sqrt(t)

    if not denominator:
        print("")
    return numerator / denominator


def d2(S, K, t, r, sigma):
    return d1(S, K, t, r, sigma) - sigma * np.sqrt(t)


def forward_price(S, t, r):
    return S / np.exp(-r * t)


def call(S, K, t, r, sigma):
    e_to_the_minus_rt = np.exp(-r * t)
    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)

    return S * N(D1) - K * e_to_the_minus_rt * N(D2)


def put(S, K, t, r, sigma):
    e_to_the_minus_rt = np.exp(-r * t)
    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)

    return -S * N(-D1) + K * e_to_the_minus_rt * N(-D2)


def delta(S, K, t, r, sigma, flag):
    d_1 = d1(S, K, t, r, sigma)

    if flag.upper().startswith("P"):
        return N(d_1) - 1.0
    else:
        return N(d_1)


def gamma(S, K, t, r, sigma):
    d_1 = d1(S, K, t, r, sigma)
    return pdf(d_1) / (S * sigma * np.sqrt(t))


def theta(S, K, t, r, sigma, flag):
    two_sqrt_t = 2 * np.sqrt(t)

    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)

    first_term = (-S * pdf(D1) * sigma) / two_sqrt_t

    if flag.upper().startswith("C"):
        second_term = r * K * np.exp(-r * t) * N(D2)
        return (first_term - second_term) / 365.0

    else:
        second_term = r * K * np.exp(-r * t) * N(-D2)
        return (first_term + second_term) / 365.0


def vega(S, K, t, r, sigma):
    d_1 = d1(S, K, t, r, sigma)
    return S * pdf(d_1) * np.sqrt(t) * 0.01


def rho(S, K, t, r, sigma, flag):
    d_2 = d2(S, K, t, r, sigma)
    e_to_the_minus_rt = np.exp(-r * t)
    if flag.upper().startswith("C"):
        return t * K * e_to_the_minus_rt * N(d_2) * 0.01
    else:
        return -t * K * e_to_the_minus_rt * N(-d_2) * 0.01


def implied_volatility(price, S, K, t, r, flag):
    if flag.upper().startswith("P"):
        f = lambda sigma: price - put(S, K, t, r, sigma)
    else:
        f = lambda sigma: price - call(S, K, t, r, sigma)

    return brentq(
        f, a=1e-12, b=100, xtol=1e-15, rtol=1e-15, maxiter=1000, full_output=False
    )


def main():
    # Comparing time to calculate implied volatility using two different methods
    import timeit

    # Generate random data
    np.random.seed(42)
    Ss = np.random.uniform(40000, 45000, 100)
    Ks = np.random.uniform(40000, 45000, 100)
    ts = np.random.uniform(0.0027, 0.0191, 100)
    rs = np.array([0.05] * 100)
    flags = np.random.choice(["c", "p"], 100)
    sigmas = np.random.uniform(0.1, 0.5, 100)
    prices = np.array(
        [
            call(s, k, t, r, sigma) if f == "c" else put(s, k, t, r, sigma)
            for s, k, t, r, sigma, f in zip(Ss, Ks, ts, rs, sigmas, flags)
        ]
    )
    deltas = np.array(
        [
            delta(s, k, t, r, sigma, f)
            for s, k, t, r, sigma, f in zip(Ss, Ks, ts, rs, sigmas, flags)
        ]
    )
    gammas = np.array(
        [gamma(s, k, t, r, sigma) for s, k, t, r, sigma in zip(Ss, Ks, ts, rs, sigmas)]
    )
    thetas = np.array(
        [
            theta(s, k, t, r, sigma, f)
            for s, k, t, r, sigma, f in zip(Ss, Ks, ts, rs, sigmas, flags)
        ]
    )
    vegas = np.array(
        [vega(s, k, t, r, sigma) for s, k, t, r, sigma in zip(Ss, Ks, ts, rs, sigmas)]
    )

    # Calculate implied volatility using two different methods
    start = timeit.default_timer()
    ivs = []
    for price, s, k, t, r, f in zip(prices, Ss, Ks, ts, rs, flags):
        iv = implied_volatility(price, s, k, t, r, f)
        ivs.append(iv)

    stop = timeit.default_timer()
    print("Time to calculate implied volatility using brentq: ", stop - start)

    import pandas as pd

    return pd.DataFrame(
        {
            "spot": Ss,
            "strike": Ks,
            "time": ts * 365,
            "rate": rs,
            "flag": flags,
            "sigma": sigmas,
            "price": prices,
            "delta": deltas,
            "gamma": gammas,
            "theta": thetas,
            "vega": vegas,
            "implied_volatility": ivs,
        }
    )


def iv_curve_adjustor(movement, iv=1, spot=100, strike=100):

    """
    This function returns a function that adjusts the implied volatility to account for the curve effect. The function
    is currently calibrated for 2 days to expiry and it simulates a movement from atm. It tries to arrive at an IV
    for the atm after the movement.
    :param movement: movement of the underlying in percentage with sign
    :param iv: implied volatility of the strike
    :param spot: spot price
    :param strike: strike price
    :return: adjusted implied volatility for the strike after the movement
    """

    coefs = [1065.29, 11.3532, 0.973536]
    current_diff = (spot/strike - 1)
    current_iv_multiple = coefs[0] * current_diff ** 2 + coefs[1] * current_diff + coefs[2]
    atm_iv = iv / current_iv_multiple
    print(f'atm_iv: {atm_iv}, iv: {iv}, current_iv_multiple: {current_iv_multiple}')
    displacement_from_atm = movement + current_diff
    print(f'displacement from atm: {displacement_from_atm}, movement: {movement}')
    premium_to_atm_iv = coefs[0] * displacement_from_atm ** 2 + coefs[1] * displacement_from_atm + coefs[2]
    return atm_iv * premium_to_atm_iv


def target_movement(flag, current_price, target_price, current_spot, strike, timeleft, time_delta=None):
    """
    :param flag: 'c' or 'p'
    :param current_price: current price of the option
    :param target_price: target price of the option
    :param current_spot: current spot price
    :param strike: strike price
    :param timeleft: time left to expiry in years
    :param time_delta: in minutes
    :return:
    """
    flag = flag.lower()[0]
    price_func = call if flag == 'c' else put
    vol = implied_volatility(current_price, current_spot, strike, timeleft, 0.06, flag)
    print(vol)
    timeleft = timeleft - (time_delta / 525600) if time_delta else timeleft
    f = lambda s1: price_func(s1, strike, timeleft, 0.06, vol) - target_price

    if target_price > current_price:
        if flag == 'c':
            a = current_spot
            b = 2 * current_spot
        else:
            a = 0.05
            b = current_spot
    else:
        if flag == 'c':
            a = 0.05
            b = current_spot
        else:
            a = current_spot
            b = 2 * current_spot

    target_spot = brentq(f, a=a, b=b, xtol=1e-15, rtol=1e-15, maxiter=1000, full_output=False)

    assert isinstance(target_spot, float)

    movement = (target_spot/current_spot) - 1

    return movement
