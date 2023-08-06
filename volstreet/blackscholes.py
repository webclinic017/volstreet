from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np
import logging
from datetime import datetime
from volstreet.exceptions import OptionModelInputError

bs_logger = logging.getLogger("blackscholes")
today = datetime.now().strftime("%Y-%m-%d")
file_handler = logging.FileHandler(f"bs-{today}.log")
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
bs_logger.setLevel(logging.INFO)
bs_logger.addHandler(file_handler)

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

    try:
        return brentq(
            f, a=1e-12, b=100, xtol=1e-15, rtol=1e-15, maxiter=1000, full_output=False
        )
    except Exception as e:
        bs_logger.error(
            f"Error in implied_volatility: {e}, price={price}, S={S}, K={K}, t={t}, r={r}, flag={flag}"
        )
        raise e


def test_func():
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


def iv_transformer_coeffs(tte):
    adjuster = 3 if tte < (0.8 / 365) else 1
    dfs2 = 1 / ((tte**1.2) * adjuster)
    dfs2 = min(dfs2, 20000)

    dfs = 1 / ((tte**0.45) * 5)
    dfs = min(dfs, 5)
    dfs = -6 + dfs
    return dfs2, dfs, 0.97


def iv_transformer_coeffs_wip(tte):
    # distance squared coefficient
    dfs2 = 3270.27 * np.exp(-384.38 * tte) + 100
    dfs2 = min(dfs2, 20000)

    # distance coefficient
    if tte < 0.26 / 365:
        dfs = 1
    else:
        dfs = 1 / ((tte**0.45) * 5)
        dfs = min(dfs, 5)
        dfs = -6 + dfs

    # intercept
    if tte < 3 / (24 * 365):
        intercept = 1.07
    elif tte < 0.27 / 365:
        intercept = 1
    else:
        intercept = 0.98
    return dfs2, dfs, intercept


def iv_curve_adjustor(
    movement,
    time_to_expiry,
    iv: int | tuple = 1,
    spot=100,
    strike=100,
    _print_details=False,
):
    """
    This function returns the adjusted implied volatility accounting for the curve effect.
    :param movement: movement of the underlying in percentage with sign
    :param time_to_expiry: time to expiry in years
    :param iv: implied volatility of the strike
    :param spot: spot price
    :param strike: strike price
    :param _print_details: print details of the adjustment
    :return: adjusted implied volatility for the strike after the movement
    """

    coefs = iv_transformer_coeffs_wip(time_to_expiry)
    current_diff = strike / spot - 1
    current_iv_multiple = (
        coefs[0] * current_diff**2 + coefs[1] * current_diff + coefs[2]
    )
    atm_iv = iv / current_iv_multiple

    new_spot = spot * (1 + movement)
    total_displacement = strike / new_spot - 1
    premium_to_atm_iv = (
        coefs[0] * total_displacement**2 + coefs[1] * total_displacement + coefs[2]
    )
    new_iv = atm_iv * premium_to_atm_iv

    if _print_details:
        print(
            f"New iv: {new_iv} for strike {strike} spot {new_spot} "
            f"iv {iv} atm_iv {atm_iv} movement {movement} "
            f"time_to_expiry {time_to_expiry}"
        )

    return new_iv


def target_movement(
    flag,
    current_price,
    target_price,
    current_spot,
    strike,
    timeleft,
    time_delta=None,
    _print_details=False,
):
    """
    :param flag: 'c' or 'p'
    :param current_price: current price of the option
    :param target_price: target price of the option
    :param current_spot: current spot price
    :param strike: strike price
    :param timeleft: time left to expiry in years
    :param time_delta: in minutes
    :param _print_details: print details of the adjustment
    :return:
    """
    flag = flag.lower()[0]
    strike_diff = current_spot - strike if flag == "c" else strike - current_spot
    if strike_diff > current_price:
        raise OptionModelInputError(
            f"Current price {current_price} of {'call' if flag == 'c' else 'put'} is less than the strike difference"
        )
    price_func = call if flag == "c" else put
    vol = implied_volatility(current_price, current_spot, strike, timeleft, 0.06, flag)
    delta_ = delta(current_spot, strike, timeleft, 0.06, vol, flag)
    estimated_movement_points = (target_price - current_price) / delta_
    estimated_movement = estimated_movement_points / current_spot
    timeleft = timeleft - (time_delta / 525600) if time_delta else timeleft

    if (
        timeleft < 0.0008
    ):  # On expiry day we need to adjust the vol as iv increases steadily as we approach expiry
        vol_multiple = 2 - (1401.74 * timeleft)
        vol = vol * vol_multiple

    modified_vol = iv_curve_adjustor(
        estimated_movement,
        timeleft,
        iv=vol,
        spot=current_spot,
        strike=strike,
        _print_details=_print_details,
    )

    if _print_details:
        print(
            f"estimated movement: {estimated_movement}, vol: {vol}, modified vol: {modified_vol}"
        )

    f = lambda s1: price_func(s1, strike, timeleft, 0.06, modified_vol) - target_price

    if target_price > current_price:
        if flag == "c":
            a = current_spot
            b = 2 * current_spot
        else:
            a = 0.05
            b = current_spot
    else:
        if flag == "c":
            a = 0.05
            b = current_spot
        else:
            a = current_spot
            b = 2 * current_spot

    target_spot = brentq(
        f, a=a, b=b, xtol=1e-15, rtol=1e-15, maxiter=1000, full_output=False
    )

    assert isinstance(target_spot, float)

    movement = (target_spot / current_spot) - 1

    return movement
