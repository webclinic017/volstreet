import volstreet as vs
import volstreet.datamodule as dm
import threading
from datetime import time
from time import sleep


def get_user_data(client, user, pin, apikey, authkey, webhook_url):
    # Checking if either client or user, pin, apikey and authkey are provided
    if client is None and (
        user is None or pin is None or apikey is None or authkey is None
    ):
        raise ValueError(
            "Either client or user, pin, apikey and authkey must be provided"
        )

    # If client is provided, user, pin, apikey and authkey will be fetched from the environment variables
    if client:
        user = __import__("os").environ[f"{client}_USER"]
        pin = __import__("os").environ[f"{client}_PIN"]
        apikey = __import__("os").environ[f"{client}_API_KEY"]
        authkey = __import__("os").environ[f"{client}_AUTHKEY"]

        if webhook_url is None:
            try:
                webhook_url = __import__("os").environ[f"{client}_WEBHOOK_URL"]
            except KeyError:
                webhook_url = None

    return user, pin, apikey, authkey, webhook_url


def intraday_options_on_indices(
    parameters,
    strategy,  # "straddle" or "strangle"
    client=None,
    user=None,
    pin=None,
    apikey=None,
    authkey=None,
    webhook_url=None,
    shared_data=True,
    start_time=(9, 16),
    safe_indices=None,
    special_parameters=None,
):
    """
    :param parameters: parameters for the strategy (refer to the strategy's docstring)
    :param strategy: 'straddle' or 'strangle' will invoke intraday_straddle or intraday_strangle of the index
    :param client:
    :param user:
    :param pin:
    :param apikey:
    :param authkey:
    :param webhook_url:
    :param shared_data:
    :param start_time:
    :param safe_indices: list of indices to be traded when no clear close expiry is available
    :param special_parameters: special parameters for a particular index
    :return:
    """

    if special_parameters is None:
        special_parameters = {}

    user, pin, apikey, authkey, discord_webhook_url = get_user_data(
        client, user, pin, apikey, authkey, webhook_url
    )

    # If today is a holiday, the script will exit
    if vs.currenttime().date() in vs.holidays:
        vs.notifier("Today is a holiday. Exiting.", discord_webhook_url)
        exit()

    vs.login(
        user=user,
        pin=pin,
        apikey=apikey,
        authkey=authkey,
        webhook_url=discord_webhook_url,
    )
    nifty = vs.Index("NIFTY", webhook_url=discord_webhook_url)
    bnf = vs.Index("BANKNIFTY", webhook_url=discord_webhook_url)
    fin = vs.Index("FINNIFTY", webhook_url=discord_webhook_url)
    midcap = vs.Index("MIDCPNIFTY", webhook_url=discord_webhook_url)

    indices = vs.get_strangle_indices_to_trade(nifty, bnf, fin, midcap, safe_indices=safe_indices)

    parameters["quantity_in_lots"] = parameters["quantity_in_lots"] // len(indices)

    # Setting the shared data
    if shared_data:
        shared_data = vs.SharedData()
        update_data_thread = threading.Thread(target=shared_data.update_data)
        parameters["shared_data"] = shared_data
    else:
        shared_data = None
        update_data_thread = None

    options_threads = []
    for index in indices:
        index_parameters = parameters.copy()
        index_parameters.update(special_parameters.get(index.name, {}))
        vs.logger.info(
            f"Trading {index.name} {strategy} with parameters {index_parameters}"
        )
        vs.notifier(f"Trading {index.name} {strategy}.", discord_webhook_url)
        thread = threading.Thread(
            target=getattr(index, f"intraday_{strategy}"), kwargs=index_parameters
        )
        options_threads.append(thread)

    # Wait for the market to open
    while vs.currenttime().time() < time(*start_time):
        sleep(1)

    # Start the data updater thread
    if shared_data and update_data_thread is not None:
        update_data_thread.start()

    # Start the options threads
    for thread in options_threads:
        thread.start()

    for thread in options_threads:
        thread.join()

    # Stop the data updater thread
    if shared_data and update_data_thread is not None:
        shared_data.force_stop = True
        update_data_thread.join()

    # Call the data appender function on the traded indices
    for index in indices:
        vs.append_data_to_json(
            index.order_log, f"{user}_{index.name}_{strategy}_log.json"
        )


def overnight_straddle_nifty(
    quantity_in_lots,
    client=None,
    user=None,
    pin=None,
    apikey=None,
    authkey=None,
    webhook_url=None,
):
    user, pin, apikey, authkey, discord_webhook_url = get_user_data(
        client, user, pin, apikey, authkey, webhook_url
    )

    # If today is a holiday, the script will exit
    if vs.currenttime().date() in vs.holidays:
        vs.notifier(
            "Today is either a holiday or Friday. Exiting.", discord_webhook_url
        )
        exit()

    vs.login(
        user=user,
        pin=pin,
        apikey=apikey,
        authkey=authkey,
        webhook_url=discord_webhook_url,
    )
    nifty = vs.Index("NIFTY", webhook_url=discord_webhook_url)

    # Rolling over the short straddle
    nifty.rollover_overnight_short_straddle(
        quantity_in_lots, strike_offset=1.003, take_avg_price=True
    )

    # Buying next week's hedge if it is expiry day
    if vs.timetoexpiry(nifty.current_expiry, in_days=True) < 1:
        nifty.buy_weekly_hedge(
            quantity_in_lots, "strangle", call_offset=0.997, put_offset=0.98
        )

    try:
        vs.append_data_to_json(nifty.order_log, f"{user}_NIFTY_ON_straddle_log.json")
    except Exception as e:
        vs.notifier(f"Appending data failed: {e}", discord_webhook_url)


def intraday_trend_on_indices(
    parameters,
    indices,
    client=None,
    user=None,
    pin=None,
    apikey=None,
    authkey=None,
    webhook_url=None,
):
    """

    :param parameters: parameters for the strategy (refer to the strategy's docstring)
                       summary of parameters:
                       quantity_in_lots,
                       start_time=(9, 15, 58),
                       exit_time=(15, 27),
                       sleep_time=5,
                       threshold_movement=None,
                       minutes_to_avg=45,
                       beta=0.8,
                       max_entries=3
    :param indices: list of indices to trade
    :param client: client's name
    :param user: username
    :param pin: user's pin
    :param apikey: user apikey
    :param authkey: user authkey
    :param webhook_url: discord webhook url

    """

    user, pin, apikey, authkey, discord_webhook_url = get_user_data(
        client, user, pin, apikey, authkey, webhook_url
    )

    if vs.currenttime().date() in vs.holidays:
        vs.notifier("Today is a holiday hence exiting.", discord_webhook_url)
        exit()

    vs.login(
        user=user,
        pin=pin,
        apikey=apikey,
        authkey=authkey,
        webhook_url=discord_webhook_url,
    )

    threads = []
    for index_symbol in indices:
        index = vs.Index(index_symbol, webhook_url=discord_webhook_url)
        thread = threading.Thread(target=index.intraday_trend, kwargs=parameters)
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def index_vs_constituents(
    index_symbol,
    strike_offset=0,
    index_strike_offset=None,
    cutoff_pct=90,
    exposure_per_stock=10000000,
    expirys=None,
    ignore_last=0,
):
    index_strike_offset = (
        strike_offset if index_strike_offset is None else index_strike_offset
    )
    expirys = ("future", "current") if expirys is None else expirys

    # Fetch constituents
    constituent_tickers, constituent_weights = vs.get_index_constituents(
        index_symbol, cutoff_pct
    )
    total_weight, number_of_stocks = sum(constituent_weights), len(constituent_tickers)
    percent_weights = [weight / total_weight for weight in constituent_weights]
    total_exposure = exposure_per_stock * number_of_stocks

    # Fetch index info
    index = vs.Index(index_symbol)
    index_info = index.fetch_otm_info(index_strike_offset, expiry=expirys[0])
    index_iv, index_shares = (
        index_info["avg_iv"],
        int(total_exposure / (index.fetch_ltp() * index.lot_size)) * index.lot_size,
    )
    index_premium_value = index_info["total_price"] * index_shares
    index_break_even_points = (
        index_info["underlying_price"],
        index_info["call_strike"],
        index_info["put_strike"],
    )
    index_break_even_points += (
        index_info["call_strike"] + index_info["total_price"],
        index_info["put_strike"] - index_info["total_price"],
    )

    # Calculate movements to break even
    def _return_abs_movement(current_price, threshold_price):
        return abs((threshold_price / current_price - 1)) * 100

    index_break_even_points += tuple(
        _return_abs_movement(index_info["underlying_price"], bep)
        for bep in index_break_even_points[1:3]
    )
    index_call_break_even, index_put_break_even = index_break_even_points[-2:]

    # Fetch constituent info
    constituents = list(map(vs.Stock, constituent_tickers))
    constituent_infos = [
        stock.fetch_otm_info(strike_offset, expiry=expirys[1]) for stock in constituents
    ]
    constituent_ivs = [info["avg_iv"] for info in constituent_infos]
    constituent_ivs_weighted_avg = sum(
        iv * pw for iv, pw in zip(constituent_ivs, percent_weights)
    )
    weighted_exposures = [total_exposure * pw for pw in percent_weights]
    shares_per_stock = [
        int(exposure / (stock.fetch_ltp() * stock.lot_size)) * stock.lot_size
        for exposure, stock in zip(weighted_exposures, constituents)
    ]
    premium_per_stock = [info["total_price"] for info in constituent_infos]
    premium_values_per_stock = [
        premium * shares for premium, shares in zip(premium_per_stock, shares_per_stock)
    ]
    premium_difference = sum(premium_values_per_stock) - index_premium_value
    break_even_points_per_stock = [
        (
            info["underlying_price"],
            info["call_strike"],
            info["put_strike"],
            info["call_strike"] + premium,
            info["put_strike"] - premium,
        )
        for info, premium in zip(constituent_infos, premium_per_stock)
    ]
    break_even_points_per_stock = [
        (
            bep[0],
            bep[1],
            bep[2],
            bep[3],
            bep[4],
            _return_abs_movement(info["underlying_price"], bep[1]),
            _return_abs_movement(info["underlying_price"], bep[2]),
        )
        for bep, info in zip(break_even_points_per_stock, constituent_infos)
    ]

    # Average break evens
    break_evens_weighted_avg = [
        sum(
            bep[i] * pw for bep, pw in zip(break_even_points_per_stock, percent_weights)
        )
        for i in [3, 4]
    ]

    # Analyzing recent realized volatility
    recent_vols = dm.get_multiple_recent_vol(
        [index_symbol] + constituent_tickers,
        frequency="M-THU",
        periods=[2, 5, 7, 10, 15, 20],
        ignore_last=ignore_last,
    )
    period_vol_dict = {
        f"Last {period} period avg": {
            "index": recent_vols[index_symbol][period][0],
            "constituents_vols_weighted_avg": sum(
                ticker[period][0] * pw
                for ticker, pw in zip(list(recent_vols.values())[1:], percent_weights)
            ),
        }
        for period in recent_vols[index_symbol]
    }

    # Return the data
    return {
        "index_iv": index_iv,
        "constituents_iv_weighted": constituent_ivs_weighted_avg,
        "constituents_iv_unweighted": sum(constituent_ivs) / number_of_stocks,
        "index_shares": index_shares,
        "index_premium_value": index_premium_value,
        "constituent_tickers": constituent_tickers,
        "constituent_weights": constituent_weights,
        "shares_per_stock": shares_per_stock,
        "premium_values_per_stock": premium_values_per_stock,
        "total_constituents_premium_value": sum(premium_values_per_stock),
        "premium_value_difference": premium_difference,
        "total_exposure": total_exposure,
        "profit_percentage": premium_difference / total_exposure * 100,
        "index_trade_info": index_break_even_points,
        "constituent_trade_infos": break_even_points_per_stock,
        "index_call_break_even": index_call_break_even,
        "index_put_break_even": index_put_break_even,
        "call_side_break_evens_wtd_avg": break_evens_weighted_avg[0],
        "put_side_break_evens_wtd_avg": break_evens_weighted_avg[1],
        "recent_vols": period_vol_dict,
    }
