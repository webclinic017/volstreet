from autotrader import autotradingfunctions as atf
import threading
from datetime import time
import pandas as pd


def index_intraday_straddles(
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

    discord_webhook_url = webhook_url

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
    fin = atf.Index("FINNIFTY", webhook_url=discord_webhook_url, spot_future_rate=0.01)

    indices = atf.indices_to_trade(nifty, bnf, fin, multi_before_weekend=multi_before_weekend)
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

    shared_data.force_stop = True
    update_data_thread.join()

    # Call the data appender function on the traded indices
    for index in indices:
        atf.append_data_to_json(
            index.order_log, f"{user}_{index.name}_straddle_log.json"
        )


def index_vs_constituents(
        index_symbol, strike_offset=0, index_strike_offset=None, cutoff_pct=90, exposure_per_stock=10000000
):

    if index_strike_offset is None:
        index_strike_offset = strike_offset

    # Fetching the constituents of the index and filtering out the top x% weighted stocks
    constituents = pd.read_csv(f'autotrader/{index_symbol}_constituents.csv')
    constituents = constituents.sort_values('Index weight', ascending=False)
    constituents['cum_weight'] = constituents['Index weight'].cumsum()
    constituents = constituents[constituents.cum_weight < cutoff_pct]
    constituent_tickers = constituents.Ticker.to_list()
    constituent_weights = constituents['Index weight'].to_list()

    # Calculating the exposure per stock
    total_weight = sum(constituent_weights)
    percent_weights = [weight / total_weight for weight in constituent_weights]
    number_of_stocks = len(constituent_tickers)
    total_exposure = exposure_per_stock * number_of_stocks
    weighted_exposures = [(exposure_per_stock*number_of_stocks) * percent_weights[i] for i in range(number_of_stocks)]

    # BNF info
    index = atf.Index(index_symbol)
    index_info = index.fetch_otm_info(index_strike_offset, fut_expiry=True)
    index_iv = index_info['avg_iv']
    index_shares = int(total_exposure / (index.fetch_ltp() * index.lot_size)) * index.lot_size
    index_premium_value = index_info['total_price'] * index_shares
    index_trade_info = (index_info['underlying_price'], index_info['call_strike'], index_info['put_strike'])

    # Constituent info
    constituents = [atf.Stock(stock) for stock in constituent_tickers]
    constituent_infos = [stock.fetch_otm_info(strike_offset) for stock in constituents]
    constituent_ivs = [stock_info['avg_iv'] for stock_info in constituent_infos]
    constituent_ivs_weighted_avg = sum([constituent_ivs[i] * percent_weights[i] for i in
                                        range(number_of_stocks)])
    constituent_ivs_unweighted_avg = sum(constituent_ivs) / number_of_stocks
    shares_per_stock = [int(exposure / (stock.fetch_ltp() * stock.lot_size)) * stock.lot_size for exposure, stock in
                        zip(weighted_exposures, constituents)]
    premium_per_stock = [stock_info['total_price'] for stock_info in constituent_infos]
    premium_values_per_stock = [premium_per_stock[i] * shares_per_stock[i] for i in range(number_of_stocks)]
    total_constituents_premium_value = sum(premium_values_per_stock)
    premium_difference = total_constituents_premium_value - index_premium_value
    constituent_trade_infos = [(stock_info['underlying_price'], stock_info['call_strike'], stock_info['put_strike'])
                               for stock_info in constituent_infos]
    break_even_points_per_stock = [(info[1] + premium_per_stock[i], info[2] - premium_per_stock[i]) for i, info in
                                   enumerate(constituent_trade_infos)]

    # Returning the data
    return {'index_iv': index_iv, 'constituents_iv_weighted': constituent_ivs_weighted_avg,
            'constituents_iv_unweighted': constituent_ivs_unweighted_avg, 'index_shares': index_shares,
            'index_premium_value': index_premium_value, 'constituent_tickers': constituent_tickers,
            'constituent_weights': constituent_weights, 'shares_per_stock': shares_per_stock,
            'premium_values_per_stock': premium_values_per_stock,
            'total_constituents_premium_value': total_constituents_premium_value,
            'premium_value_difference': premium_difference, 'total_exposure': total_exposure,
            'profit_percentage': premium_difference/total_exposure * 100, 'index_trade_info': index_trade_info,
            'constituent_trade_infos': constituent_trade_infos,
            'break_even_points_per_stock': break_even_points_per_stock}
