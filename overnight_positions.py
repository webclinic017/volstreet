from autotrader import autotradingfunctions as atf
from time import sleep
import json

nifty_discord_url = None

bnf_discord_url = None


atf.login(user='user',
          pin='pin',
          apikey='apikey',
          authkey='authkey',
          webhook_url=nifty_discord_url)


# Initialising the indices
nifty = atf.Index('NIFTY', webhook_url=nifty_discord_url)
bnf = atf.Index('BANKNIFTY', webhook_url=bnf_discord_url)

# Rolling over the daily short straddle
nifty.rollover_overnight_short_straddle(quantity_in_lots=30, multiple_of_orders=8, strike_offset=1.003)
sleep(1)

# Rolling over the daily short butterfly
nifty.rollover_short_butterfly(quantity_in_lots=20, multiple_of_orders=2)
sleep(1)

# Checking whether to buy the hedge and buying it if required
if atf.timetoexpiry(nifty.current_expiry) * 365 < 1:
    nifty.buy_weekly_hedge(quantity_in_lots=30, multiple_of_orders=8, type='strangle',
                           call_offset=0.997, put_offset=0.98)
sleep(1)

# Logging the trades
order_log = nifty.order_list + bnf.order_list
with open(f"{atf.obj.userId}_order_log.txt", "a") as file:
    for trade in order_log:
        file.write(json.dumps(trade) + "\n")
