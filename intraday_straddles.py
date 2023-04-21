from autotrader import autotradingfunctions as atf
import threading
from datetime import time
from autotrader.discord_bot import run_bot

# User inputs
discord_webhook_url = None
discord_bot_token = ''
user = ''
pin = ''
apikey = ''
authkey = ''

# Trade inputs
quantity_in_lots = 5
exit_time = (15, 26, 00)
move_sl = False
stoploss = 'dynamic'
wait_for_equality = False
target_disparity = 2
catch_trend = False
take_profit = False
take_profit_points = 3

# If today is a holiday, the script will exit
if atf.currenttime().date() in atf.holidays:
    atf.notifier('Today is a holiday. Exiting.', discord_webhook_url)
    exit()

atf.login(user=user, pin=pin, apikey=apikey, authkey=authkey, webhook_url=discord_webhook_url)

indices = [atf.Index(index_name, webhook_url=discord_webhook_url) for index_name in ['FINNIFTY', 'NIFTY', 'BANKNIFTY']]
shared_data = atf.SharedData()
update_data_thread = threading.Thread(target=shared_data.update_data)

less_than_3_days = atf.timetoexpiry(indices[0].current_expiry, effective_time=True, in_days=True) < 3
main_expiry = atf.timetoexpiry(indices[1].current_expiry, effective_time=True, in_days=True) < 1

straddle_threads = []
for index in indices:
    if index.name == 'FINNIFTY':
        if not main_expiry and less_than_3_days:
            # If FINNIFTY is allowed to trade, update the quantity
            quantity_in_lots = quantity_in_lots * 2
        else:
            atf.notifier(f'Skipping {index.name} straddle.', discord_webhook_url)
            index.traded = False
            continue
    else:
        # If FINNIFTY is allowed to trade, skip other indices
        if not main_expiry and less_than_3_days:
            atf.notifier(f'Skipping {index.name} straddle.', discord_webhook_url)
            index.traded = False
            continue

    thread = threading.Thread(target=index.intraday_straddle,
                              kwargs={'quantity_in_lots': quantity_in_lots,
                                      'wait_for_equality': wait_for_equality,
                                      'move_sl': move_sl,
                                      'exit_time': exit_time,
                                      'shared_data': shared_data,
                                      'catch_trend': catch_trend,
                                      'stoploss': stoploss,
                                      'target_disparity': target_disparity,
                                      'take_profit': take_profit,
                                      'take_profit_points': take_profit_points})
    straddle_threads.append(thread)
    index.traded = True

# Start the discord bot
discord_bot_thread = threading.Thread(target=run_bot, args=(discord_bot_token, indices))
discord_bot_thread.daemon = True
discord_bot_thread.start()

# Wait for the market to open
while atf.currenttime().time() < time(9, 16):
    pass

# Start the data updater thread
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
    if index.traded:
        atf.append_data_to_json(index.order_log, f'{user}_{index.name}_straddle_log.json')
