import autotrader.datamodule as dm

# Using DataClient class
client = dm.DataClient(api_key=__import__('os').environ['EOD_API_KEY'])
# %%
# Using get_data and analyser functions
nifty_data = client.get_data(symbol='NIFTY')
bnf_data = client.get_data(symbol='BANKNIFTY')
nifty_weekly_data = dm.analyser(nifty_data, frequency='W-THU')
bnf_weekly_data = dm.analyser(bnf_data, frequency='W-THU')
nifty_monthly_data = dm.analyser(nifty_data, frequency='M-THU')
bnf_monthly_data = dm.analyser(bnf_data, frequency='M-THU')
# %%
# Using ratio_analysis function
ratio_data = dm.ratio_analysis(bnf_weekly_data, nifty_weekly_data, periods_to_avg=5, return_summary=True)
#%%
# Using generate_streak function
bnf2 = dm.generate_streak(bnf_weekly_data, 'abs_change < 2')
# %%
resp = dm.get_multiple_recent_vol(['BANKNIFTY', 'AXISBANK', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN'],
                                  'M-THU', periods=[1, 5, 10, 24, 36], ignore_last=0, client=client)
# %%
dm.get_recent_vol(bnf_weekly_data, periods=[1, 5, 10, 20])
# %%
# Using gambler function
resp2 = dm.gambler(bnf_data, 'W', 'abs_change < 2')
resp2_dataframe = resp['dataframe'][0]
resp2_dataframe['streak_count'].value_counts()
