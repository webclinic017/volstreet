import autotrader.autotradingfunctions as atf
from autotrader.exceptions import ApiKeyNotFound
from eod import EodHistoricalData
import pandas as pd
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR
import numpy as np


class DataClient:
    def __init__(self, api_key):
        if api_key is None:
            raise ApiKeyNotFound("EOD API Key not found")
        self.api_key = api_key
        self.client = EodHistoricalData(api_key=api_key)

    def get_data(self, symbol, from_date="2011-01-01", return_columns=None):
        symbol_dict = {
            "NIFTY": "NSEI.INDX",
            "BANKNIFTY": "NSEBANK.INDX",
            "FINNIFTY": "CNXFIN.INDX",
        }
        symbol = symbol.upper()
        name = symbol.split(".")[0]
        if "." not in symbol:
            if symbol in symbol_dict:
                symbol = symbol_dict[symbol]
            else:
                symbol = symbol + ".NSE"

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


def get_summary_ratio(target_symbol, benchmark_symbol, frequency='D', periods_to_avg=50):

    try:
        dc = DataClient(__import__('os').getenv('EOD_API_KEY'))
    except ApiKeyNotFound:
        return None

    benchmark = dc.get_data(benchmark_symbol)
    target = dc.get_data(target_symbol)
    benchmark = analyser(benchmark, frequency=frequency)
    target = analyser(target, frequency=frequency)
    ratio = ratio_analysis(target, benchmark, periods_to_avg=periods_to_avg)
    return ratio.loc['Summary', 'Ratio']


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

        if df.index[-1].replace(hour=15, minute=30) > atf.currenttime():
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
        lambda x: atf.findstrike(x * (1 + strike_offset), base)
    )
    data["put_strike"] = data["close"].apply(
        lambda x: atf.findstrike(x * (1 - strike_offset), base)
    )
    data["outcome_spot"] = data["close"].shift(-1)
    data["initial_premium"] = data.apply(
        lambda row: atf.calc_combined_premium(
            row.close,
            row.iv / 100,
            row.time_to_expiry,
            callstrike=row.call_strike,
            putstrike=row.put_strike,
        ),
        axis=1,
    )
    data["outcome_premium"] = data.apply(
        lambda row: atf.calc_combined_premium(
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

    index = atf.Index(index_symbol)
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
