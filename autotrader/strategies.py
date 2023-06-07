from autotrader import autotradingfunctions as atf
import autotrader.datamodule as dm
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


def intraday_straddles_on_indices(
    parameters,
    client=None,
    user=None,
    pin=None,
    apikey=None,
    authkey=None,
    webhook_url=None,
    shared_data=True,
    start_time=(9, 16),
    multi_before_weekend=True,
):
    user, pin, apikey, authkey, discord_webhook_url = get_user_data(
        client, user, pin, apikey, authkey, webhook_url
    )

    # Setting the shared data
    if shared_data:
        shared_data = atf.SharedData()
        update_data_thread = threading.Thread(target=shared_data.update_data)
        parameters["shared_data"] = shared_data
    else:
        shared_data = None
        update_data_thread = None

    # If today is a holiday, the script will exit
    if atf.currenttime().date() in atf.holidays:
        atf.notifier("Today is a holiday. Exiting.", discord_webhook_url)
        exit()

    atf.login(
        user=user,
        pin=pin,
        apikey=apikey,
        authkey=authkey,
        webhook_url=discord_webhook_url,
    )
    nifty = atf.Index("NIFTY", webhook_url=discord_webhook_url)
    bnf = atf.Index("BANKNIFTY", webhook_url=discord_webhook_url)
    fin = atf.Index("FINNIFTY", webhook_url=discord_webhook_url, spot_future_rate=0.05)

    indices = atf.indices_to_trade(
        nifty, bnf, fin, multi_before_weekend=multi_before_weekend
    )
    quantity_multiplier = 2 if len(indices) == 1 else 1
    parameters["quantity_in_lots"] = (
        parameters["quantity_in_lots"] * quantity_multiplier
    )

    straddle_threads = []
    for index in indices:
        atf.notifier(f"Trading {index.name} straddle.", discord_webhook_url)
        thread = threading.Thread(target=index.intraday_straddle, kwargs=parameters)
        straddle_threads.append(thread)

    # Wait for the market to open
    while atf.currenttime().time() < time(*start_time):
        pass

    # Start the data updater thread
    if shared_data and update_data_thread is not None:
        update_data_thread.start()

    # Start the straddle threads
    for thread in straddle_threads:
        thread.start()

    for thread in straddle_threads:
        thread.join()

    # Stop the data updater thread
    if shared_data and update_data_thread is not None:
        shared_data.force_stop = True
        update_data_thread.join()

    # Call the data appender function on the traded indices
    for index in indices:
        atf.append_data_to_json(
            index.order_log, f"{user}_{index.name}_straddle_log.json"
        )


def intraday_strangles_on_indices(
    parameters,
    client=None,
    user=None,
    pin=None,
    apikey=None,
    authkey=None,
    webhook_url=None,
    start_time=(9, 16),
    multi_before_weekend=True,
):
    user, pin, apikey, authkey, discord_webhook_url = get_user_data(
        client, user, pin, apikey, authkey, webhook_url
    )

    # If today is a holiday, the script will exit
    if atf.currenttime().date() in atf.holidays:
        atf.notifier("Today is a holiday. Exiting.", discord_webhook_url)
        exit()

    atf.login(
        user=user,
        pin=pin,
        apikey=apikey,
        authkey=authkey,
        webhook_url=discord_webhook_url,
    )
    nifty = atf.Index("NIFTY", webhook_url=discord_webhook_url)
    bnf = atf.Index("BANKNIFTY", webhook_url=discord_webhook_url)
    fin = atf.Index("FINNIFTY", webhook_url=discord_webhook_url, spot_future_rate=0.05)

    indices = atf.indices_to_trade(
        nifty, bnf, fin, multi_before_weekend=multi_before_weekend
    )
    quantity_multiplier = 2 if len(indices) == 1 else 1
    parameters["quantity_in_lots"] = (
        parameters["quantity_in_lots"] * quantity_multiplier
    )

    strangle_threads = []
    for index in indices:
        atf.notifier(f"Trading {index.name} strangle.", discord_webhook_url)
        thread = threading.Thread(target=index.intraday_strangle, kwargs=parameters)
        strangle_threads.append(thread)

    # Wait for the market to open
    while atf.currenttime().time() < time(*start_time):
        pass

    # Start the straddle threads
    for thread in strangle_threads:
        thread.start()

    for thread in strangle_threads:
        thread.join()

    # Call the data appender function on the traded indices
    for index in indices:
        atf.append_data_to_json(
            index.order_log, f"{user}_{index.name}_strangle_log.json"
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
    if atf.currenttime().date() in atf.holidays:
        atf.notifier(
            "Today is either a holiday or Friday. Exiting.", discord_webhook_url
        )
        exit()

    atf.login(
        user=user,
        pin=pin,
        apikey=apikey,
        authkey=authkey,
        webhook_url=discord_webhook_url,
    )
    nifty = atf.Index("NIFTY", webhook_url=discord_webhook_url)

    # Rolling over the short straddle
    nifty.rollover_overnight_short_straddle(
        quantity_in_lots, strike_offset=1.003, take_avg_price=True
    )

    # Buying next week's hedge if it is expiry day
    if atf.timetoexpiry(nifty.current_expiry, in_days=True) < 1:
        nifty.buy_weekly_hedge(
            quantity_in_lots, "strangle", call_offset=0.997, put_offset=0.98
        )

    try:
        atf.append_data_to_json(nifty.order_log, f"{user}_NIFTY_ON_straddle_log.json")
    except Exception as e:
        atf.notifier(f"Appending data failed: {e}", discord_webhook_url)


@atf.log_errors
def intraday_trend_on_nifty(
    quantity_in_lots,
    client=None,
    user=None,
    pin=None,
    apikey=None,
    authkey=None,
    webhook_url=None,
    start_time=(9, 15, 55),
    exit_time=(15, 27),
):
    user, pin, apikey, authkey, discord_webhook_url = get_user_data(
        client, user, pin, apikey, authkey, webhook_url
    )

    if atf.currenttime().date() in atf.holidays:
        atf.notifier("Today is a holiday. Exiting.")
        exit()

    atf.login(
        user=user,
        pin=pin,
        apikey=apikey,
        authkey=authkey,
        webhook_url=discord_webhook_url,
    )

    nifty = atf.Index("NIFTY", webhook_url=discord_webhook_url)

    while atf.currenttime().time() < time(*start_time):
        pass

    nifty_open_price = nifty.fetch_ltp()
    movement = 0
    vix = atf.get_current_vix()
    threshold_movement = vix / 48
    exit_time = time(*exit_time)
    scan_end_time = atf.datetime.combine(atf.currenttime().date(), exit_time)
    scan_end_time = scan_end_time - atf.timedelta(minutes=10)
    scan_end_time = scan_end_time.time()
    upper_limit = nifty_open_price * (1 + threshold_movement / 100)
    lower_limit = nifty_open_price * (1 - threshold_movement / 100)

    atf.notifier(
        f"Nifty trender starting with {threshold_movement:0.2f} threshold movement\n"
        f"Current Price: {nifty_open_price}\nUpper limit: {upper_limit:0.2f}\n"
        f"Lower limit: {lower_limit:0.2f}.",
        discord_webhook_url,
    )
    last_printed_time = atf.currenttime()
    while (
        abs(movement) < threshold_movement and atf.currenttime().time() < scan_end_time
    ):
        movement = ((nifty.fetch_ltp() / nifty_open_price) - 1) * 100
        if atf.currenttime() > last_printed_time + atf.timedelta(minutes=1):
            print(f"Nifty trender: {movement:0.2f} movement.")
            last_printed_time = atf.currenttime()
        sleep(1)

    if atf.currenttime().time() > scan_end_time:
        atf.notifier("Nifty trender exiting due to time.", discord_webhook_url)
        return

    price = nifty.fetch_ltp()
    atm_strike = atf.findstrike(price, nifty.base)
    position = "BUY" if movement > 0 else "SELL"
    nifty.place_synthetic_fut_order(
        atm_strike,
        nifty.current_expiry,
        position,
        quantity_in_lots,
        prices="LIMIT",
        check_status=True,
    )
    stop_loss_multiplier = 1.0032 if position == "SELL" else 0.9968
    stop_loss_price = price * stop_loss_multiplier
    stop_loss_hit = False
    atf.notifier(
        f"Nifty {position} trender triggered with {movement:0.2f} movement. Nifty at {price}. "
        f"Stop loss at {stop_loss_price}.",
        discord_webhook_url,
    )
    while atf.currenttime().time() < exit_time and not stop_loss_hit:
        if position == "BUY":
            stop_loss_hit = nifty.fetch_ltp() < stop_loss_price
        else:
            stop_loss_hit = nifty.fetch_ltp() > stop_loss_price
        sleep(3)
    nifty.place_synthetic_fut_order(
        atm_strike,
        nifty.current_expiry,
        "SELL" if position == "BUY" else "BUY",
        quantity_in_lots,
        prices="LIMIT",
        check_status=True,
    )
    stop_loss_message = "Trender stop loss hit. " if stop_loss_hit else ""
    atf.notifier(
        f"{stop_loss_message}Nifty trender exited. Nifty at {nifty.fetch_ltp()}.",
        discord_webhook_url,
    )


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
    constituent_tickers, constituent_weights = atf.get_index_constituents(
        index_symbol, cutoff_pct
    )
    total_weight, number_of_stocks = sum(constituent_weights), len(constituent_tickers)
    percent_weights = [weight / total_weight for weight in constituent_weights]
    total_exposure = exposure_per_stock * number_of_stocks

    # Fetch index info
    index = atf.Index(index_symbol)
    index_info = index.fetch_otm_info(index_strike_offset, expiry=expirys[0])
    index_iv, index_shares = (
        index_info["avg_iv"],
        int(total_exposure / (index.fetch_ltp() * index.lot_size)) * index.lot_size,
    )
    index_premium_value = index_info["total_price"] * index_shares
    index_break_even_points = (index_info["underlying_price"], index_info["call_strike"], index_info["put_strike"])
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
    constituents = list(map(atf.Stock, constituent_tickers))
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
