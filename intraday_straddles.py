from autotrader import autotradingfunctions as atf
import threading
from datetime import time
from autotrader.discord_bot import run_bot

# User inputs
client = ''
try:
    discord_webhook_url = __import__('os').environ[f'{client}_WEBHOOK_URL']
except KeyError:
    discord_webhook_url = None

user = __import__('os').environ[f'{client}_USER']
pin = __import__('os').environ[f'{client}_PIN']
apikey = __import__('os').environ[f'{client}_API_KEY']
authkey = __import__('os').environ[f'{client}_AUTHKEY']

try:
    discord_bot_token = __import__('os').environ[f'{client}_DISCORD_BOT_TOKEN']
except KeyError:
    discord_bot_token = None

# Setting the shared data
shared_data = atf.SharedData()
update_data_thread = threading.Thread(target=shared_data.update_data)

# Trade inputs
parameters = dict(
    quantity_in_lots=1,
    exit_time=(15, 28),
    websocket=None,
    shared_data=shared_data,
    wait_for_equality=False,
    target_disparity=10,
    move_sl=False,
    stoploss='dynamic',
    catch_trend=True,
    trend_qty_ratio=0.5,
    trend_catcher_sl=0.0033,
    smart_exit=True,
    safeguard=True,
    safeguard_movement=0.003,
    safeguard_spike=1.2,
    convert_to_butterfly=True,
)

# If today is a holiday, the script will exit
if atf.currenttime().date() in atf.holidays:
    atf.notifier('Today is a holiday. Exiting.', discord_webhook_url)
    exit()

atf.login(user=user, pin=pin, apikey=apikey, authkey=authkey, webhook_url=discord_webhook_url)
nifty = atf.Index('NIFTY', webhook_url=discord_webhook_url)
bnf = atf.Index('BANKNIFTY', webhook_url=discord_webhook_url)
fin = atf.Index('FINNIFTY', webhook_url=discord_webhook_url, spot_future_rate=0.01)

indices = atf.indices_to_trade(nifty, bnf, fin)
quantity_multiplier = 2 if len(indices) == 1 else 1
parameters['quantity_in_lots'] = parameters['quantity_in_lots'] * quantity_multiplier

straddle_threads = []
for index in indices:
    atf.notifier(f'Trading {index.name} straddle.', discord_webhook_url)
    index.traded = False
    thread = threading.Thread(target=index.intraday_straddle,
                              kwargs=parameters)
    straddle_threads.append(thread)

# Start the discord bot
if discord_bot_token:
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
    atf.append_data_to_json(index.order_log, f'{user}_{index.name}_straddle_log.json')
