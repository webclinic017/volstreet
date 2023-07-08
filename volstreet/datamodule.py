import volstreet as vs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.expected_conditions import url_changes
from kiteconnect import KiteConnect
from time import sleep
from functools import partial
import pyotp
from volstreet.exceptions import ApiKeyNotFound
from eod import EodHistoricalData
import pandas as pd
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR
import numpy as np
from datetime import datetime, timedelta, time


class DataClient:
    def __init__(self, api_key):
        if api_key is None:
            raise ApiKeyNotFound("EOD API Key not found")
        self.api_key = api_key
        self.client = EodHistoricalData(api_key=api_key)

    @staticmethod
    def parse_symbol(symbol):

        symbol_dict = {
            "NIFTY": "NSEI.INDX",
            "NIFTY 50": "NSEI.INDX",
            "NIFTY50" : "NSEI.INDX",
            "BANKNIFTY": "NSEBANK.INDX",
            "NIFTY BANK": "NSEBANK.INDX",
            "NIFTYBANK": "NSEBANK.INDX",
            "FINNIFTY": "CNXFIN.INDX",
            "NIFTY FIN SERVICE": "CNXFIN.INDX",
            "VIX": "NIFVIX.INDX",
        }
        symbol = symbol.upper()
        if "." not in symbol:
            if symbol in symbol_dict:
                symbol = symbol_dict[symbol]
            else:
                symbol = symbol + ".NSE"
        return symbol

    def get_data(self, symbol, from_date="2011-01-01", return_columns=None):

        name = symbol.split(".")[0] if "." in symbol else symbol

        symbol = self.parse_symbol(symbol)

        if return_columns is None:
            return_columns = ["open", "close", "gap", "intra", "abs_gap", "abs_intra"]

        resp = self.client.get_prices_eod(
            symbol, period="d", order="a", from_=from_date
        )
        df = pd.DataFrame(resp)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index(df.date, inplace=True)
        df["p_close"] = df.close.shift(1)
        df["gap"] = (df.open / df.p_close - 1) * 100
        df["intra"] = (df.close / df.open - 1) * 100
        df["abs_gap"] = abs(df.gap)
        df["abs_intra"] = abs(df.intra)
        df = df.loc[:, return_columns]
        df.name = name
        return df

    def get_intraday_data(self, symbol, interval, from_date="2011-01-01", return_columns=None):

        name = symbol.split(".")[0] if "." in symbol else symbol

        symbol = self.parse_symbol(symbol)

        if return_columns is None:
            return_columns = ['open', 'high', 'low', 'close']

        resp_list = []
        from_date = pd.to_datetime(from_date)
        while datetime.now().date() > from_date.date():
            to_date = from_date + timedelta(days=120)
            resp = self.client.get_prices_intraday(
                symbol, interval=interval,
                from_=str(int(from_date.timestamp())),
                to=str(int(to_date.timestamp()))
            )

            resp_list.extend(resp)
            from_date += timedelta(days=120)

        df = pd.DataFrame(resp_list)
        df.index = pd.to_datetime(df['datetime'])
        df = df.drop_duplicates()
        df.index = df.index + timedelta(hours=5, minutes=30)
        df = df[return_columns]
        df.name = name
        return df


def retain_name(func):
    def wrapper(df, *args, **kwargs):
        try:
            name = df.name
        except AttributeError:
            name = None
        df = func(df, *args, **kwargs)
        df.name = name
        return df

    return wrapper


def analyser(df, frequency=None, date_filter=None, _print=False):
    name = df.name
    if date_filter is None:
        pass
    else:
        dates = date_filter.split("to")
        if len(dates) > 1:
            df = df.loc[dates[0]: dates[1]]
        else:
            df = df.loc[dates[0]]

    frequency = frequency.upper() if frequency is not None else None

    if frequency is None or frequency.startswith("D") or frequency == "B":
        custom_frequency = "B"
        multiplier = 24
        df = df.resample("B").ffill()

    elif frequency.startswith("W") or frequency.startswith("M"):
        custom_frequency = frequency
        if frequency.startswith("W"):
            multiplier = 9.09
            df = df.resample(frequency).ffill()
        elif frequency.startswith("M"):
            multiplier = 4.4
            if len(frequency) == 1:
                df = df.resample("M").ffill()
            else:
                weekday_module_dict = {
                    "MON": MO,
                    "TUE": TU,
                    "WED": WE,
                    "THU": TH,
                    "FRI": FR,
                }
                frequency = frequency.lstrip("M-")
                df = df.resample(f"W-{frequency.upper()}").ffill()
                df = df.resample("M").ffill()
                df.index = df.index.date + relativedelta(
                    weekday=weekday_module_dict[frequency.upper()](-1)
                )
                df.index = pd.Series(pd.to_datetime(df.index), name="date")
        else:
            raise ValueError("Frequency not supported")
    else:
        raise ValueError("Frequency not supported")

    df.loc[:, "change"] = df.close.pct_change() * 100
    df.loc[:, "open_change"] = ((df.open / df.close.shift(1)) - 1) * 100
    df.loc[:, "abs_change"] = abs(df.change)
    df.loc[:, "abs_open_change"] = abs(df.open_change)
    df.loc[:, "realized_vol"] = df.abs_change * multiplier

    if frequency in ["D-MON", "D-TUE", "D-WED", "D-THU", "D-FRI"]:
        day_of_week = frequency.split("-")[1]
        df = df[df.index.day_name().str.upper().str.contains(day_of_week)]

    if _print:
        print(
            "Vol for period: {:0.2f}%, IV: {:0.2f}%".format(
                df.abs_change.mean(), df.abs_change.mean() * multiplier
            )
        )
    else:
        pass

    df.custom_frequency = custom_frequency.upper()
    df.name = name
    return df


def get_recent_vol(df, periods=None, ignore_last=1):
    """Returns a dictionary of vol for each period in periods list
    :param df: Dataframe with 'abs_change' column
    :param periods: List of periods to calculate vol for
    :param ignore_last: Number of rows to ignore from the end
    :return: Dictionary of vol for each period in periods list
    """

    if periods is None:
        periods = [5]
    else:
        periods = [periods] if isinstance(periods, int) else periods

    if ignore_last == 0:
        df = df
    else:
        df = df.iloc[:-ignore_last]

    vol_dict = {}
    for period in periods:
        abs_change = df.tail(period).abs_change.mean()
        realized_vol = df.tail(period).realized_vol.mean()
        vol_dict[period] = (abs_change, realized_vol)
    return vol_dict


def get_multiple_recent_vol(
    list_of_symbols, frequency, periods=None, ignore_last=1, client=None
):
    if client is None:
        client = DataClient(api_key=__import__("os").environ.get("EOD_API_KEY"))
    df_dict = {}
    for symbol in list_of_symbols:
        symbol_data = client.get_data(symbol=symbol)
        symbol_monthly_data = analyser(symbol_data, frequency=frequency)
        recent_vol = get_recent_vol(
            symbol_monthly_data, periods=periods, ignore_last=ignore_last
        )
        df_dict[symbol] = recent_vol
    return df_dict


def ratio_analysis(
        x_df: pd.DataFrame,
        y_df: pd.DataFrame,
        periods_to_avg: int = None,
        return_summary=True,
        add_rolling: bool | int = False
):

    if periods_to_avg is None:
        periods_to_avg = len(x_df)

    x_close = x_df.iloc[-periods_to_avg:].close
    x_array = x_df.iloc[-periods_to_avg:].abs_change
    x_avg = x_array.mean()

    y_close = y_df.iloc[-periods_to_avg:].close
    y_array = y_df.iloc[-periods_to_avg:].abs_change
    y_avg = y_array.mean()

    avg_ratio = x_avg / y_avg
    ratio_array = x_df.abs_change / y_df.abs_change
    ratio_array = ratio_array[-periods_to_avg:]

    labels = [x_df.name, y_df.name]

    ratio_summary = pd.DataFrame(
        {
            labels[0]: x_close,
            f"{labels[0]} Change": x_array,
            labels[1]: y_close,
            f"{labels[1]} Change": y_array,
            "Ratio": ratio_array,
        }
    )
    # print(f'\n{periods_to_avg} Period Average = {avg_ratio}\n\n')
    if return_summary:
        ratio_summary.loc["Summary"] = ratio_summary.mean()
        ratio_summary.loc["Summary", "Ratio"] = avg_ratio

    if add_rolling:
        rolling_x_avg = x_array.rolling(add_rolling, min_periods=1).mean()
        rolling_y_avg = y_array.rolling(add_rolling, min_periods=1).mean()
        rolling_ratio = rolling_x_avg / rolling_y_avg
        ratio_summary[f"Rolling {add_rolling} Ratio"] = rolling_ratio

    return ratio_summary


def get_summary_ratio(target_symbol, benchmark_symbol, frequency='D', periods_to_avg=50, client=None):

    try:
        if client is None:
            try:
                dc = DataClient(__import__('os').getenv('EOD_API_KEY'))
            except ApiKeyNotFound:
                return None
        else:
            dc = client

        benchmark = dc.get_data(benchmark_symbol)
        target = dc.get_data(target_symbol)
        benchmark = analyser(benchmark, frequency=frequency)
        target = analyser(target, frequency=frequency)
        ratio = ratio_analysis(target, benchmark, periods_to_avg=periods_to_avg)
        return ratio.loc['Summary', 'Ratio']
    except Exception as e:
        vs.logger.error(f'Error in get_summary_ratio: {e}')
        return None


@retain_name
def generate_streak(df, query):
    df = df.copy(deep=True)

    # Create a boolean series with the query
    _bool = df.query(f"{query}")
    df["result"] = df.index.isin(_bool.index)
    df["start_of_streak"] = (df["result"].ne(df["result"].shift())) & (
        df["result"] == True
    )
    df["streak_id"] = df.start_of_streak.cumsum()
    df.loc[df["result"] == False, "streak_id"] = np.nan
    df["streak_count"] = df.groupby("streak_id").cumcount() + 1

    return df[df.result == True].drop(columns=["start_of_streak"])


@retain_name
def gambler(instrument, freq, query):
    """
    This function takes in instrument dataframe, frequency, and query and returns the streaks for the query.
    The instrument df should be a dataframe with daily closing values.
    The query should be a string with the following format: '{column} {operator} {value}'.
    The column should be a column in the instrument dataframe.
    The operator should be one of the following: '>', '<', '>=', '<=', '==', '!='.
    The value should be a number.
    """

    def generate_frequency(frequency):
        if frequency.startswith("W") or frequency.startswith("M"):
            if len(frequency) == 1:
                days = ["mon", "tue", "wed", "thu", "fri"]
                return [f"{frequency}-{day}" for day in days]
            else:
                return [frequency]
        else:
            return [frequency]

    def _calculate_streak_summary(df, frequency, query):

        # Calculate the streak summary

        if df.index[-1].replace(hour=15, minute=30) > vs.currenttime():
            df = df.iloc[:-1]
        check_date = df.index[-1]
        total_instances = len(df)
        df = generate_streak(df, query)
        total_streaks = len(df)
        number_of_positive_events = total_instances - total_streaks
        event_occurrence_pct = number_of_positive_events / total_instances

        df = (
            df.reset_index()
            .groupby("streak_id")
            .agg({"date": ["min", "max"], "streak_count": "max"})
            .reset_index()
        )
        df.columns = ["streak_id", "start_date", "end_date", "streak_count"]

        # Check if there is an ongoing streak
        current_streak = (
            df.iloc[-1].streak_count if df.iloc[-1].end_date == check_date else None
        )

        # Calculating the percentile of the current streak
        if current_streak:
            current_streak_percentile = (
                    df.streak_count.sort_values().values.searchsorted(current_streak) / len(df)
            )
        else:
            current_streak_percentile = 0

        return {
            "freq": frequency,  # Use the given freq value instead of df.iloc[-1].name
            "total_instances": total_instances,
            "total_streaks": total_streaks,
            "event_occurrence": event_occurrence_pct,
            "longest_streak": df.streak_count.max(),
            "longest_streak_start": df.start_date[df.streak_count.idxmax()],
            "longest_streak_end": df.end_date[df.streak_count.idxmax()],
            "current_streak": current_streak,
            "current_streak_percentile": current_streak_percentile,
            "dataframe": df,
        }

    def print_streak_summary(summary):
        print(
            f"Query: {dataframe.name} {query}\n"
            f"Frequency: {summary['freq']}\n"
            f"Total Instances: {summary['total_instances']}\n"
            f"Total Streaks: {summary['total_streaks']}\n"
            f"Event Occurrence: {summary['event_occurrence']}\n"
            f"Longest Streak: {summary['longest_streak']}\n"
            f"Longest Streak Start: {summary['longest_streak_start']}\n"
            f"Longest Streak End: {summary['longest_streak_end']}\n"
            f"Current Streak: {summary['current_streak']}\n"
            f"Current Streak Percentile: {summary['current_streak_percentile']}\n"
        )

    freqs = generate_frequency(freq)
    streaks = []
    for freq in freqs:
        dataframe = analyser(instrument, frequency=freq)
        if query == "abs_change":
            recommended_threshold = dataframe.abs_change.mean() * 0.70  # 0.70 should cover 50% of the data
            # (mildly adjusted for abnormal distribution)
            recommended_threshold = round(recommended_threshold, 2)
            recommended_sign = ">" if dataframe.iloc[-2].abs_change > recommended_threshold else "<"
            query = f"abs_change {recommended_sign} {recommended_threshold}"
            print(f"Recommended query: {query}\n")
        streak_summary = _calculate_streak_summary(dataframe, freq, query)
        streaks.append(streak_summary)
        print_streak_summary(streak_summary)
    # Convert the list of dictionaries to a list of DataFrames
    streaks_df = [pd.DataFrame([streak]) for streak in streaks]

    # Concatenate the list of DataFrames
    return (
        pd.concat(streaks_df)
        .sort_values("longest_streak", ascending=False)
        .reset_index(drop=True)
    )


def simulate_strike_premium_payoff(
    close: pd.Series,
    iv: pd.Series,
    time_to_expiry: pd.Series,
    strike_offset: float,
    base: float = 100,
    label: str = "",
    action="buy",
):

    if label:
        label = f"{label}_"

    action = action.lower()

    if action not in ["buy", "sell"]:
        raise ValueError("action must be either 'buy' or 'sell'")

    data = pd.DataFrame(
        {
            "close": close,
            "iv": iv,
            "time_to_expiry": time_to_expiry,
        }
    )

    data["call_strike"] = data["close"].apply(
        lambda x: vs.findstrike(x * (1 + strike_offset), base)
    )
    data["put_strike"] = data["close"].apply(
        lambda x: vs.findstrike(x * (1 - strike_offset), base)
    )
    data["outcome_spot"] = data["close"].shift(-1)
    data["initial_premium"] = data.apply(
        lambda row: vs.calc_combined_premium(
            row.close,
            row.iv / 100,
            row.time_to_expiry,
            callstrike=row.call_strike,
            putstrike=row.put_strike,
        ),
        axis=1,
    )
    data["outcome_premium"] = data.apply(
        lambda row: vs.calc_combined_premium(
            row.outcome_spot,
            row.iv / 100,
            0,
            callstrike=row.call_strike,
            putstrike=row.put_strike,
        ),
        axis=1,
    )
    data["payoff"] = (
        data["initial_premium"] - data["outcome_premium"]
        if action == "sell"
        else data["outcome_premium"] - data["initial_premium"]
    )
    data["payoff"] = data["payoff"].shift(1)
    data["payoff_pct"] = data["payoff"] / data["close"]
    data = data[
        [
            "call_strike",
            "put_strike",
            "initial_premium",
            "outcome_premium",
            "payoff",
            "payoff_pct",
        ]
    ]
    data.columns = [f"{label}{col}" for col in data.columns]
    return data


def get_index_vs_constituents_recent_vols(
    index_symbol,
    return_all=False,
    simulate_backtest=False,
    strike_offset=0,
    hedge_offset=0,
    stock_vix_adjustment=0.7,
    index_action="sell"
):
    """
    Get the recent volatility of the index and its constituents
    """
    if return_all is False:
        simulate_backtest = False

    index = vs.Index(index_symbol)
    constituents, weights = index.get_constituents(cutoff_pct=90)
    weights = [w / sum(weights) for w in weights]

    dc = DataClient(api_key=__import__("os").environ["EOD_API_KEY"])

    index_data = dc.get_data(symbol=index_symbol)
    index_monthly_data = analyser(index_data, frequency="M-THU")
    index_monthly_data = index_monthly_data[["close", "abs_change"]]
    index_monthly_data.columns = ["index_close", "index_abs_change"]

    if simulate_backtest:
        if index_symbol == "BANKNIFTY":
            index_ivs = pd.read_csv(
                "data/banknifty_ivs.csv",
                parse_dates=True,
                index_col="date",
                dayfirst=True,
            )
            index_ivs.index = pd.to_datetime(index_ivs.index)
            index_ivs = index_ivs.resample("D").ffill()
            index_monthly_data = index_monthly_data.merge(
                index_ivs, left_index=True, right_index=True, how="left"
            )
            index_monthly_data["index_iv"] = index_monthly_data["close"].fillna(
                method="ffill"
            )
            index_monthly_data.drop(columns=["close"], inplace=True)
            index_monthly_data["iv_diff_from_mean"] = (
                index_monthly_data["index_iv"] / index_monthly_data["index_iv"].mean()
            )
            index_monthly_data["time_to_expiry"] = (
                index_monthly_data.index.to_series().diff().dt.days / 365
            )

            index_hedge_action = "buy" if index_action == "sell" else "sell"

            # The main strike
            simulated_data = simulate_strike_premium_payoff(index_monthly_data["index_close"],
                                                            index_monthly_data["index_iv"],
                                                            index_monthly_data["time_to_expiry"], strike_offset, 100,
                                                            label="index", action=index_action)
            index_monthly_data = index_monthly_data.merge(
                simulated_data, left_index=True, right_index=True, how="left"
            )

            index_monthly_data["index_initial_premium_pct"] = (
                    index_monthly_data["index_initial_premium"] / index_monthly_data["index_close"]
            )

            # The hedge strike
            simulated_data = simulate_strike_premium_payoff(index_monthly_data["index_close"],
                                                            index_monthly_data["index_iv"],
                                                            index_monthly_data["time_to_expiry"], hedge_offset, 100,
                                                            label="index_hedge", action=index_hedge_action)

            index_monthly_data = index_monthly_data.merge(simulated_data, left_index=True, right_index=True, how="left")

            index_monthly_data["index_hedge_initial_premium_pct"] = (
                    index_monthly_data["index_hedge_initial_premium"] / index_monthly_data["index_close"]
            )

            index_monthly_data["index_bep_pct"] = (
                    index_monthly_data["index_initial_premium_pct"] -
                    index_monthly_data["index_hedge_initial_premium_pct"]
            )

        else:
            raise NotImplementedError

    constituent_dfs = []
    for i, constituent in enumerate(constituents):
        constituent_data = dc.get_data(symbol=constituent)
        constituent_monthly_data = analyser(constituent_data, frequency="M-THU")
        constituent_monthly_data = constituent_monthly_data[["close", "abs_change"]]
        constituent_monthly_data.columns = [
            f"{constituent}_close",
            f"{constituent}_abs_change",
        ]
        constituent_monthly_data[f"{constituent}_abs_change_weighted"] = (
            constituent_monthly_data[f"{constituent}_abs_change"] * weights[i]
        )

        if simulate_backtest:
            constituent_monthly_data[f"{constituent}_iv"] = index_monthly_data[
                "iv_diff_from_mean"
            ] * (
                (constituent_monthly_data[f"{constituent}_abs_change"].mean() - stock_vix_adjustment) * 4.4
            )  # the adjustment factor is to account for the spurious volatility on account of splits

            constituent_action = "buy" if index_action == "sell" else "sell"
            constituent_hedge_action = "sell" if constituent_action == "buy" else "sell"

            # The main strike
            simulated_data = simulate_strike_premium_payoff(constituent_monthly_data[f"{constituent}_close"],
                                                            constituent_monthly_data[f"{constituent}_iv"],
                                                            index_monthly_data["time_to_expiry"], strike_offset, 5,
                                                            label=constituent, action=constituent_action)
            constituent_monthly_data = constituent_monthly_data.merge(
                simulated_data, left_index=True, right_index=True, how="left"
            )

            constituent_monthly_data[f"{constituent}_initial_premium_pct"] = (
                constituent_monthly_data[f"{constituent}_initial_premium"] /
                constituent_monthly_data[f"{constituent}_close"]
            )

            # The hedge strike
            simulated_data = simulate_strike_premium_payoff(constituent_monthly_data[f"{constituent}_close"],
                                                            constituent_monthly_data[f"{constituent}_iv"],
                                                            index_monthly_data["time_to_expiry"], hedge_offset, 5,
                                                            label=f'{constituent}_hedge',
                                                            action=constituent_hedge_action)
            constituent_monthly_data = constituent_monthly_data.merge(
                simulated_data, left_index=True, right_index=True, how="left"
            )

            constituent_monthly_data[f"{constituent}_hedge_initial_premium_pct"] = (
                constituent_monthly_data[f"{constituent}_hedge_initial_premium"] /
                constituent_monthly_data[f"{constituent}_close"]
            )

            constituent_monthly_data[f"{constituent}_bep_pct"] = (
                constituent_monthly_data[f"{constituent}_initial_premium_pct"]
                - constituent_monthly_data[f"{constituent}_hedge_initial_premium_pct"]
            )

            constituent_monthly_data[f"{constituent}_total_payoff"] = (
                constituent_monthly_data[f"{constituent}_payoff"]
                + constituent_monthly_data[f"{constituent}_hedge_payoff"]
            )
            constituent_monthly_data[f"{constituent}_total_payoff_pct"] = (
                constituent_monthly_data[f"{constituent}_total_payoff"] /
                constituent_monthly_data[f"{constituent}_close"]
            )
            constituent_monthly_data[f"{constituent}_total_payoff_pct_weighted"] = (
                constituent_monthly_data[f"{constituent}_total_payoff_pct"] * weights[i]
            )

        constituent_dfs.append(constituent_monthly_data)

    index_monthly_data = index_monthly_data.merge(
        pd.concat(constituent_dfs, axis=1), left_index=True, right_index=True, how="inner"
    )
    index_monthly_data = index_monthly_data.copy()
    index_monthly_data["sum_constituent_movement"] = index_monthly_data.filter(
        regex="abs_change_weighted"
    ).sum(axis=1)
    index_monthly_data["ratio_of_movements"] = (
        index_monthly_data["sum_constituent_movement"]
        / index_monthly_data["index_abs_change"]
    )

    if simulate_backtest:
        index_monthly_data["index_total_payoff"] = (
                index_monthly_data["index_payoff"] + index_monthly_data["index_hedge_payoff"]
        )
        index_monthly_data["index_total_payoff_pct"] = (
                index_monthly_data["index_total_payoff"] / index_monthly_data["index_close"]
        )
        index_monthly_data["sum_constituent_payoff_pct"] = index_monthly_data.filter(
            regex="total_payoff_pct_weighted"
        ).sum(axis=1)

        index_monthly_data['total_combined_payoff_pct'] = (
                index_monthly_data["index_total_payoff_pct"] + index_monthly_data["sum_constituent_payoff_pct"]
        )

    if return_all:
        return index_monthly_data
    else:
        summary_df = index_monthly_data[
            ["index_abs_change", "sum_constituent_movement", "ratio_of_movements"]
        ]
        return summary_df


def get_greenlit_kite(
        kite_api_key,
        kite_api_secret,
        kite_user_id,
        kite_password,
        kite_auth_key,
        chrome_path=None
):
    if chrome_path is None:
        driver = webdriver.Chrome()
    else:
        driver = webdriver.Chrome(chrome_path)

    authkey_obj = pyotp.TOTP(kite_auth_key)
    kite = KiteConnect(api_key=kite_api_key)
    login_url = kite.login_url()

    driver.get(login_url)
    wait = WebDriverWait(driver, 10)  # waits for up to 10 seconds

    userid = wait.until(EC.presence_of_element_located((By.ID, 'userid')))
    userid.send_keys(kite_user_id)

    password = wait.until(EC.presence_of_element_located((By.ID, 'password')))
    password.send_keys(kite_password)

    submit = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'button-orange')))
    submit.click()

    sleep(10)  # wait for the OTP input field to be clickable
    otp_input = wait.until(EC.element_to_be_clickable((By.TAG_NAME, 'input')))
    otp_input.send_keys(authkey_obj.now())

    # wait until the URL changes
    wait.until(url_changes(driver.current_url))

    # now you can safely get the current URL
    current_url = driver.current_url

    split_url = current_url.split('=')
    request_token = None
    for i, string in enumerate(split_url):
        if 'request_token' in string:
            request_token = split_url[i + 1]
            request_token = request_token.split('&')[0] if '&' in request_token else request_token
            break

    driver.quit()

    if request_token is None:
        raise Exception('Request token not found')

    data = kite.generate_session(request_token, api_secret=kite_api_secret)
    kite.set_access_token(data["access_token"])

    return kite


def get_1m_data(kite, symbol, path='C:\\Users\\Administrator\\'):

    def fetch_minute_data_from_kite(_kite, _token, _from_date, _to_date):
        date_format = '%Y-%m-%d %H:%M:%S'
        _prices = _kite.historical_data(_token, from_date=_from_date.strftime(date_format),
                                        to_date=_to_date.strftime(date_format), interval='minute')
        return _prices

    instruments = kite.instruments(exchange='NSE')
    token = [instrument['instrument_token'] for instrument in instruments if instrument['tradingsymbol'] == symbol][0]

    try:
        main_df = pd.read_csv(
            f'{path}{symbol}_onemin_prices.csv', index_col=0, parse_dates=True
        )
        from_date = main_df.index[-1] + timedelta(minutes=1)
    except FileNotFoundError:
        print(f"No existing data for {symbol}, starting from scratch.")
        main_df = False
        from_date = datetime(2015, 1, 1, 9, 16)

    end_date = datetime.now()
    mainlist = []

    fetch_data_partial = partial(fetch_minute_data_from_kite, kite, token)

    while from_date < end_date:
        to_date = from_date + timedelta(days=60)
        prices = fetch_data_partial(from_date, to_date)
        if len(prices) < 2:
            print(f'No data for {from_date.strftime("%Y-%m-%d %H:%M:%S")} to {to_date.strftime("%Y-%m-%d %H:%M:%S")}')
            return None
        mainlist.extend(prices)
        from_date += timedelta(days=60)

    df = pd.DataFrame(mainlist).set_index('date')
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')]
    df = df[(df.index.time >= time(9, 15)) & (df.index.time <= time(15, 30))]
    df.to_csv(f'{path}{symbol}_onemin_prices.csv', mode='a', header=not isinstance(main_df, pd.DataFrame))
    print(f'Finished fetching data for {symbol}. Fetched data from {df.index[0]} to {df.index[-1]}')
    full_df = pd.concat([main_df, df]) if isinstance(main_df, pd.DataFrame) else df
    return full_df


def backtest_intraday_trend(one_min_df, beta=1, trend_threshold=1, eod_client=None):

    one_min_df = one_min_df.copy()
    if one_min_df.index.name == 'date':
        one_min_df = one_min_df.reset_index()
    one_min_df = one_min_df[(one_min_df['date'].dt.time > time(9, 15)) & (one_min_df['date'].dt.time < time(15, 30))]

    unavailable_dates = [
        datetime(2015, 2, 28).date(),
        datetime(2016, 10, 30).date(),
        datetime(2019, 10, 27).date(),
        datetime(2020, 2, 1).date(),
        datetime(2020, 11, 14).date()
    ]

    # Fetching vix data and calculating beta
    if eod_client is None:
        client = DataClient(api_key=__import__("os").environ.get("EOD_API_KEY"))
    else:
        client = eod_client

    vix = client.get_data("VIX", return_columns=["open", "close"])
    vix = vix.resample('B').ffill()
    vix['open'] = vix['open'] * beta
    vix['close'] = vix['close'] * beta

    one_min_df.drop(one_min_df[one_min_df['date'].dt.date.isin(unavailable_dates)].index, inplace=True)
    open_prices = one_min_df.groupby(one_min_df['date'].dt.date).close.first().to_frame()
    open_data = open_prices.merge(vix['open'].to_frame(), left_index=True, right_index=True)
    open_data['threshold_movement'] = (open_data['open'] / 48) * trend_threshold
    one_min_df[['day_open', 'open_vix', 'threshold_movement']] = open_data.loc[one_min_df['date'].dt.date].values
    one_min_df['change_from_open'] = ((one_min_df['close']/one_min_df['day_open']) - 1)*100

    def calculate_daily_trade_data(group):
        result_dict = {
            'returns': 0,
            'trigger_time': np.nan,
            'trigger_price': np.nan,
            'stop_loss_price': np.nan
        }
        # Find the first index where the absolute price change crosses the threshold
        idx = group[abs(group['change_from_open']) >= group['threshold_movement']].first_valid_index()
        if idx is not None:
            # Record the price and time of crossing the threshold
            cross_price = group.loc[idx, 'close']
            cross_time = group.loc[idx, 'date']

            # Determine the direction of the movement
            direction = np.sign(group.loc[idx, 'change_from_open'])

            # Calculate the stoploss price
            stoploss_price = cross_price * (1 - 0.003 * direction)

            # Check if the stoploss was breached after crossing the threshold
            future_prices = group.loc[idx:, 'close']
            if (
                    direction == 1 and future_prices.min() <= stoploss_price) \
                    or (direction == -1 and future_prices.max() >= stoploss_price
            ):
                result_dict['returns'] = -0.30
            else:
                # Return the change from the entry price to the last price of the day
                if direction == 1:
                    result_dict['returns'] = ((group['close'].iloc[-1] - cross_price) / cross_price)*100
                else:
                    result_dict['returns'] = ((group['close'].iloc[-1] - cross_price) / cross_price)*-100

            result_dict.update({
                'trigger_time': cross_time,
                'trigger_price': cross_price,
                'stop_loss_price': stoploss_price
            })
        return result_dict

    # Applying the function to each day's worth of data
    returns = one_min_df.groupby(one_min_df['date'].dt.date).apply(calculate_daily_trade_data)
    returns = returns.apply(pd.Series)

    # creating a new column with the date
    one_min_df.rename(columns={'date': 'timestamp'}, inplace=True)
    one_min_df['date'] = one_min_df['timestamp'].dt.date

    # merging the returns with the original data
    one_min_df = one_min_df.merge(returns, left_on='date', right_index=True)

    return one_min_df


def daywise_returns_of_intraday_trend(df):

    daywise_summary = df.groupby(df["timestamp"].dt.date).agg(
        {'returns': 'first', 'trigger_time': 'first', 'trigger_price': 'first', 'stop_loss_price': 'first'})
    daywise_summary.index = pd.to_datetime(daywise_summary.index)
    daywise_summary = nav_drawdown_analyser(daywise_summary, column_to_convert='returns', profit_in_pct=True)
    return daywise_summary


def calculate_intraday_return_drivers(symbol_one_min_df, day_wise_summary_df, rolling_days=60):

    symbol_one_min_df = symbol_one_min_df[
        (symbol_one_min_df.index.time > time(9, 15)) & (symbol_one_min_df.index.time < time(15, 30))
        ]
    minute_vol = symbol_one_min_df.groupby(symbol_one_min_df.index.date).apply(
        lambda x: x['close'].pct_change().abs().mean() * 100).to_frame()
    minute_vol.index = pd.to_datetime(minute_vol.index)
    minute_vol.columns = ['minute_vol']

    open_to_close_trend = symbol_one_min_df.close.groupby(symbol_one_min_df.index.date).apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100).abs().to_frame()
    open_to_close_trend.index = pd.to_datetime(open_to_close_trend.index)
    open_to_close_trend.columns = ['open_to_close_trend']
    drivers_of_returns = minute_vol.merge(open_to_close_trend, left_index=True, right_index=True)
    drivers_of_returns = drivers_of_returns.merge(
        day_wise_summary_df.returns.to_frame(), left_index=True, right_index=True
    )

    # Rolling the minute vol and open to close trend
    rolling_days = rolling_days
    drivers_of_returns['minute_vol_rolling'] = drivers_of_returns['minute_vol'].rolling(rolling_days,
                                                                                        min_periods=1).mean()
    drivers_of_returns['open_to_close_trend_rolling'] = drivers_of_returns['open_to_close_trend'].rolling(
        rolling_days, min_periods=1).mean()
    drivers_of_returns['ratio'] = drivers_of_returns['open_to_close_trend'] / drivers_of_returns['minute_vol']
    drivers_of_returns['rolling_ratio'] = drivers_of_returns['open_to_close_trend_rolling'] / drivers_of_returns[
        'minute_vol_rolling']
    drivers_of_returns['returns_rolling'] = (drivers_of_returns['returns'] / 100 + 1).cumprod()

    return drivers_of_returns


def nav_drawdown_analyser(df, column_to_convert='profit', base_price_col='close', nav_start=100, profit_in_pct=False):
    """ Supply an analysed dataframe with a column that has the profit/loss in percentage or absolute value.
    Params:
    df: Dataframe with the column to be converted to NAV
    column_to_convert: Column name to be converted to NAV (default: 'profit')
    nav_start: Starting NAV (default: 100)
    profit_in_pct: If the column is in percentage or absolute value (default: False)
    """

    df = df.copy(deep=True)
    if column_to_convert in df.columns:
        if profit_in_pct == False:
            df['profit_pct'] = (df[column_to_convert] / df[base_price_col]) * 100
        else:
            df['profit_pct'] = df[column_to_convert]
        df['strat_nav'] = ((df.profit_pct + 100) / 100).dropna().cumprod() * nav_start
        df['cum_max'] = df.strat_nav.cummax()
        df['drawdown'] = ((df.strat_nav / df.cum_max) - 1) * 100
        df['rolling_cagr'] = df[:-1].apply(lambda row: ((df.strat_nav[-1] / row.strat_nav) ** (1 / (
                (df.iloc[-1].name - row.name).days / 365)) - 1) * 100, axis=1)

        # Drawdown ID below
        df['drawdown_checker'] = np.where(df.drawdown != 0, 1, 0)
        df['change_in_trend'] = df.drawdown_checker.ne(df.drawdown_checker.shift(1))
        # df['streak_id'] = df.change_in_trend.cumsum()
        df['start_of_drawdown'] = np.where((df.change_in_trend == True) &
                                           (df.drawdown_checker == 1), True, False)
        df['end_of_drawdown'] = np.where((df.change_in_trend == True) &
                                         (df.drawdown_checker == 0), True, False)
        df['drawdown_id'] = df.start_of_drawdown.cumsum()
        df.drawdown_id = np.where(df.drawdown_checker == 1, df.drawdown_id, None)
        return df.drop(['start_of_drawdown', 'end_of_drawdown', 'drawdown_checker', 'change_in_trend'], axis=1)
    else:
        print('No profit column found in df.')
