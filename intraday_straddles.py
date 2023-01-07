import autotradingfunctions as atf
import threading
import json

discord_webhook_url = None

if atf.currenttime().weekday() in [0, 1, 2, 3, 4]:

    atf.login(user='user',
              pin='pin',
              apikey='apikey',
              authkey='authkey',
              webhook_url=None)

    # Getting the ticker file
    atf.get_ticker_file()

    nifty = atf.Index('NIFTY', webhook_url=discord_webhook_url)
    bnf = atf.Index('BANKNIFTY', webhook_url=discord_webhook_url)
    finnifty = atf.Index('FINNIFTY', webhook_url=discord_webhook_url)

    if atf.timetoexpiry(finnifty.current_expiry) * 365 < 2:

        nifty_straddle = threading.Thread(target=nifty.intraday_straddle,
                                          kwargs={'quantity_in_lots': 25,
                                                  'multiple_of_orders': 2,
                                                  'exit_time': (15, 28, 30)})
        bnf_straddle = threading.Thread(target=bnf.intraday_straddle,
                                        kwargs={'quantity_in_lots': 25,
                                                'multiple_of_orders': 2,
                                                'exit_time': (15, 28, 30)})
        finnifty_straddle = threading.Thread(target=finnifty.intraday_straddle,
                                             kwargs={'quantity_in_lots': 25,
                                                     'multiple_of_orders': 2,
                                                     'exit_time': (15, 28, 30)})
    else:

        nifty_straddle = threading.Thread(target=nifty.intraday_straddle,
                                          kwargs={'quantity_in_lots': 25,
                                                  'multiple_of_orders': 3,
                                                  'exit_time': (15, 28, 30)})
        bnf_straddle = threading.Thread(target=bnf.intraday_straddle,
                                        kwargs={'quantity_in_lots': 25,
                                                'multiple_of_orders': 3,
                                                'exit_time': (15, 28, 30)})
        finnifty_straddle = False

    while atf.currenttime().time() < time(9, 15):
        pass

    if nifty_straddle:
        nifty_straddle.start()
    if bnf_straddle:
        bnf_straddle.start()
    if finnifty_straddle:
        finnifty_straddle.start()

    if nifty_straddle:
        nifty_straddle.join()
    if bnf_straddle:
        bnf_straddle.join()
    if finnifty_straddle:
        finnifty_straddle.join()

    # Logging the trades
    order_list = []
    if nifty_straddle:
        order_list.append(nifty.order_list[0])
    if bnf_straddle:
        order_list.append(bnf.order_list[0])
    if finnifty_straddle:
        order_list.append(finnifty.order_list[0])

    with open(f"{atf.obj.userId}_order_log.txt", "a") as file:
        for trade in order_list:
            dump_data = {}
            for key, value in trade.items():
                if isinstance(value, np.int32) or isinstance(value, np.int64):
                    dump_data[key] = int(value)
                elif isinstance(value, float):
                    dump_data[key] = float(value)
                else:
                    dump_data[key] = value
            file.write(json.dumps(dump_data) + "\n")
else:
    atf.notifier('Market is closed.', webhook_url=discord_webhook_url)
