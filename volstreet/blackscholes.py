from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np
import logging
from datetime import datetime
from volstreet.exceptions import OptionModelInputError
from volstreet.constants import iv_models

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
    except ValueError as e:
        if "f(a) and f(b) must have different signs" in str(e):
            raise e
        else:
            bs_logger.error(
                f"Error in implied_volatility: {e}, price={price}, S={S}, K={K}, t={t}, r={r}, flag={flag}"
            )
            raise e
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


def get_iv_model_for_time_to_expiry(time_to_expiry):
    # Filtering the models based on the time to expiry
    filtered_model = [*filter(lambda x: x[0] <= time_to_expiry < x[1], iv_models)][0]
    # Returning the model for the segment
    return iv_models[filtered_model]


def iv_multiple_to_atm(time_to_expiry, spot, strike, symbol="NIFTY"):
    iv_model = get_iv_model_for_time_to_expiry(time_to_expiry)
    distance = (strike / spot) - 1
    distance_squared = distance**2
    moneyness = spot / strike
    distance_time_interaction = distance_squared * time_to_expiry
    finnifty = True if symbol.upper() == "FINNIFTY" else False
    nifty = True if symbol.upper() == "NIFTY" else False

    return iv_model.predict(
        [
            [
                distance,
                distance_squared,
                moneyness,
                distance_time_interaction,
                finnifty,
                nifty,
            ]
        ]
    )[0]


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
    symbol="NIFTY",
    time_delta_minutes=None,
    print_details=False,
):
    """
    This function returns the adjusted implied volatility accounting for the curve effect.
    :param movement: movement of the underlying in percentage with sign
    :param time_to_expiry: time to expiry in years
    :param iv: implied volatility of the strike
    :param spot: spot price
    :param strike: strike price
    :param symbol: symbol for rhe random forest model
    :param time_delta_minutes: time delta in minutes
    :param print_details: print details of the adjustment
    :return: adjusted implied volatility for the strike after the movement
    """

    def get_iv_multiple_to_atm(tte, s, k, sym, distance):
        try:
            # Model the IV curve using the random forest models
            return iv_multiple_to_atm(tte, s, k, sym)
        except Exception as e:
            bs_logger.error(
                f"Error in iv_multiple_to_atm: {e}, time_to_expiry={tte}, spot={s}, strike={k}, symbol={sym}"
            )

            # Get the regression coefficients for the IV curve
            coeffs = iv_transformer_coeffs_wip(tte)

            # Apply the IV curve model to the current displacement
            return coeffs[0] * distance**2 + coeffs[1] * distance + coeffs[2]

    # Calculate the current displacement from ATM
    current_displacement = strike / spot - 1

    # Calculate the new spot price after the movement
    new_spot = spot * (1 + movement)

    # Calculate the new displacement from ATM
    total_displacement = strike / new_spot - 1

    # Get the IV multiple for the current displacement
    current_iv_multiple = get_iv_multiple_to_atm(
        time_to_expiry, spot, strike, symbol, current_displacement
    )

    # Normalize the given IV to the ATM level by dividing by the current IV multiple
    atm_iv = iv / current_iv_multiple

    # New time to expiry after the movement
    new_time_to_expiry = (
        time_to_expiry - (time_delta_minutes / 525600)
        if time_delta_minutes
        else time_to_expiry
    )

    if new_time_to_expiry < 0.000001:
        new_time_to_expiry = 0.000001

    # Apply the IV curve model to the new displacement
    premium_to_atm_iv = get_iv_multiple_to_atm(
        new_time_to_expiry, new_spot, strike, symbol, total_displacement
    )

    if (
        new_time_to_expiry < 0.0008
    ):  # On expiry day we need to adjust the vol as iv increases steadily as we approach expiry
        vol_multiple = 1 + (time_delta_minutes / 375)
        new_atm_iv = atm_iv * vol_multiple
    else:
        new_atm_iv = atm_iv

    # Scale the normalized ATM IV by the premium to get the new IV
    new_iv = new_atm_iv * premium_to_atm_iv

    if print_details:
        print(
            f"New IV: {new_iv} for Strike: {strike}\n"
            f"Starting IV: {iv}, ATM IV: {atm_iv}\nMovement {movement}\n"
            f"Spot after move: {new_spot}\n"
            f"Time to expiry: {new_time_to_expiry} from {time_to_expiry}"
        )

    return new_iv


def target_movement(
    flag,
    starting_price,
    target_price,
    starting_spot,
    strike,
    time_left,
    time_delta_minutes=None,
    symbol="NIFTY",
    print_details=False,
):
    """
    :param flag: 'c' or 'p'
    :param starting_price: current price of the option
    :param target_price: target price of the option
    :param starting_spot: current spot price
    :param strike: strike price
    :param time_left: time left to expiry in years
    :param time_delta_minutes: time delta in minutes
    :param print_details: print details of the adjustment
    :param symbol: symbol for the random forest model
    :return:
    """
    flag = flag.lower()[0]
    strike_diff = starting_spot - strike if flag == "c" else strike - starting_spot
    if strike_diff > starting_price:
        raise OptionModelInputError(
            f"Current price {starting_price} of {'call' if flag == 'c' else 'put'} is less than the strike difference"
        )
    price_func = call if flag == "c" else put
    vol = implied_volatility(
        starting_price, starting_spot, strike, time_left, 0.06, flag
    )
    new_time_left = (
        time_left - (time_delta_minutes / 525600) if time_delta_minutes else time_left
    )
    delta_ = delta(starting_spot, strike, new_time_left, 0.06, vol, flag)
    estimated_movement_points = (target_price - starting_price) / delta_
    estimated_movement = estimated_movement_points / starting_spot

    modified_vol = iv_curve_adjustor(
        estimated_movement,
        time_left,
        iv=vol,
        spot=starting_spot,
        strike=strike,
        symbol=symbol,
        time_delta_minutes=time_delta_minutes,
        print_details=print_details,
    )

    f = (
        lambda s1: price_func(s1, strike, new_time_left, 0.06, modified_vol)
        - target_price
    )

    if target_price > starting_price:
        if flag == "c":
            a = starting_spot
            b = 2 * starting_spot
        else:
            a = 0.05
            b = starting_spot
    else:
        if flag == "c":
            a = 0.05
            b = starting_spot
        else:
            a = starting_spot
            b = 2 * starting_spot

    target_spot = brentq(
        f, a=a, b=b, xtol=1e-15, rtol=1e-15, maxiter=1000, full_output=False
    )

    assert isinstance(target_spot, float)

    movement = (target_spot / starting_spot) - 1

    return movement
