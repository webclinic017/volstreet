from autotrader import autotradingfunctions as atf
import threading
import json
from datetime import time
import numpy as np

discord_webhook_url = None

atf.login(user='user',
          pin='pin',
          apikey='apikey',
          authkey='authkey',
          webhook_url=None)

nifty = atf.Index('NIFTY', webhook_url=discord_webhook_url)
bnf = atf.Index('BANKNIFTY', webhook_url=discord_webhook_url)
finnifty = atf.Index('FINNIFTY', webhook_url=discord_webhook_url)
shared_data = atf.SharedData()
update_data_thread = threading.Thread(target=shared_data.update_data)
update_data_thread.start()


if atf.timetoexpiry(finnifty.current_expiry, effective_time=True, in_days=True) < 3:

    nifty_straddle = threading.Thread(target=nifty.intraday_straddle,
                                      kwargs={'quantity_in_lots': 5,
                                              'monitor_sl': True,
                                              'exit_time': (15, 28, 30),
                                              'shared_data': shared_data})
    bnf_straddle = threading.Thread(target=bnf.intraday_straddle,
                                    kwargs={'quantity_in_lots': 5,
                                            'monitor_sl': True,
                                            'exit_time': (15, 28, 30),
                                            'shared_data': shared_data})
    finnifty_straddle = threading.Thread(target=finnifty.intraday_straddle,
                                         kwargs={'quantity_in_lots': 5,
                                                 'monitor_sl': True,
                                                 'exit_time': (15, 28, 30),
                                                 'shared_data': shared_data})
else:

    nifty_straddle = threading.Thread(target=nifty.intraday_straddle,
                                      kwargs={'quantity_in_lots': 10,
                                              'monitor_sl': True,
                                              'exit_time': (15, 28, 30),
                                              'shared_data': shared_data})
    bnf_straddle = threading.Thread(target=bnf.intraday_straddle,
                                    kwargs={'quantity_in_lots': 10,
                                            'monitor_sl': True,
                                            'exit_time': (15, 28, 30),
                                            'shared_data': shared_data})
    finnifty_straddle = False

while atf.currenttime().time() < time(9, 16):
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

