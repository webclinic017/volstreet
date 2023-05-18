from autotrader.discord_bot import run_bot
from autotrader.strategies import run_index_straddles
from multiprocessing import Manager, Pool
import threading

try:
    discord_bot_token = __import__('os').environ['DISCORD_BOT_TOKEN']
except KeyError:
    discord_bot_token = None

# Trade inputs
parameters = dict(
    quantity_in_lots=1,
    exit_time=(15, 28),
    websocket=None,
    shared_data=None,
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

user_indices_dict = None

# Start the discord bot
if discord_bot_token:
    user_indices_dict = Manager.dict()
    discord_bot_thread = threading.Thread(target=run_bot, args=(discord_bot_token, user_indices_dict))
    discord_bot_thread.daemon = True
    discord_bot_thread.start()

client_dict = {'abc': {}, 'xyz': {}}
with Pool() as p:
    for client in client_dict:
        parameters.update(client_dict[client])
        results = p.apply_async(run_index_straddles, args=(parameters,),
                                kwargs=dict(client=client,
                                            webhook_url=None,
                                            multi_before_weekend=False,
                                            user_indices_dict=user_indices_dict))
    p.close()
    p.join()
