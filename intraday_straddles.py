from autotrader import autotradingfunctions as atf
import threading
from datetime import time

discord_webhook_url = None
user = ''
pin = ''
apikey = ''
authkey = ''

atf.login(user=user, pin=pin, apikey=apikey, authkey=authkey)

indices = [atf.Index(index_name, webhook_url=discord_webhook_url) for index_name in ['FINNIFTY', 'NIFTY', 'BANKNIFTY']]
shared_data = atf.SharedData()
update_data_thread = threading.Thread(target=shared_data.update_data)

less_than_3_days = atf.timetoexpiry(indices[0].current_expiry, effective_time=True, in_days=True) < 3
main_expiry = atf.timetoexpiry(indices[1].current_expiry, effective_time=True, in_days=True) < 1
quantity_in_lots = 2
exit_time = (15, 29, 15)
monitor_sl = True
move_sl = False
wait_for_equality = False

straddle_threads = []
for index in indices:
    if index.name == 'FINNIFTY' and (not less_than_3_days or main_expiry):
        atf.notifier(f'Skipping {index.name} straddle.', discord_webhook_url)
        index.traded = False
        quantity_in_lots = int(quantity_in_lots * (3/2))
        continue
    thread = threading.Thread(target=index.intraday_straddle,
                              kwargs={'quantity_in_lots': quantity_in_lots,
                                      'wait_for_equality': wait_for_equality,
                                      'monitor_sl': monitor_sl,
                                      'move_sl': move_sl,
                                      'exit_time': exit_time,
                                      'shared_data': shared_data})
    straddle_threads.append(thread)
    index.traded = True

while atf.currenttime().time() < time(9, 16):
    pass

update_data_thread.start()

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
