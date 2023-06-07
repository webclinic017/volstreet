from volstreet.strategies import intraday_options_on_indices
from multiprocessing import Process
from time import sleep

# Trade inputs
parameters = dict(
    quantity_in_lots=1,
    exit_time=(15, 29, 30),
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

client_dict = {
    'abc': {},
    'xyz': {
        'quantity_in_lots': 1,
        'catch_trend': False
    }
}


def start_strategy(user, params):
    print('Starting strategy for client:', client)
    process = Process(target=intraday_options_on_indices,
                      kwargs=dict(parameters=params,
                                  strategy='straddle',
                                  client=user,
                                  multi_before_weekend=False))
    process.start()


if __name__ == '__main__':

    sleep(5)
    for client in client_dict:
        client_parameters = parameters.copy()
        client_parameters.update(client_dict[client])
        start_strategy(client, client_parameters)
