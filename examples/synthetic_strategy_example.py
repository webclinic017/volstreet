import volstreet as vs
from threading import Thread
from time import sleep

discord_webhook_url = None
user = ''
pin = ''
apikey = ''
authkey = ''

vs.login(user, pin, apikey, authkey, discord_webhook_url)
nifty = vs.Index("NIFTY")
bnf = vs.Index("BANKNIFTY")
fin = vs.Index("FINNIFTY")

pf = vs.PriceFeed(vs.obj, vs.login_data, correlation_id="optionwatch")
pf.start_websocket()
pf.add_options(nifty, bnf, fin, range_of_strikes=10, expiries="current", mode=3)
t = Thread(target=pf.update_option_chain, kwargs=dict(calculate_iv=False, process_iv_log=False, sleep_time=0.5))
t.start()

sleep(5)
arb_sys = vs.SyntheticArbSystem(pf.symbol_option_chains)
arb_sys.find_arbitrage_opportunities('BANKNIFTY', '25MAY23', 25, at_market=True, threshold=4)
