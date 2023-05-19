from autotrader import autotradingfunctions as atf
import threading
from datetime import time


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

def stocks_vs_index():
    pass