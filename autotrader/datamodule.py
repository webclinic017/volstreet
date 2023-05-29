import autotrader.autotradingfunctions as atf
from eod import EodHistoricalData
import pandas as pd
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR
import numpy as np


class DataClient:
    def __init__(self, api_key):
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
            df = df.loc[dates[0] : dates[1]]
        else:
            df = df.loc[dates[0]]

    if frequency is None or frequency == "D" or frequency == "B":
        custom_frequency = "B"
        multiplier = 24
        df = df.resample("B").ffill()

    else:
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

    df.loc[:, "change"] = df.close.pct_change() * 100
    df.loc[:, "open_change"] = ((df.open / df.close.shift(1)) - 1) * 100
    df.loc[:, "abs_change"] = abs(df.change)
    df.loc[:, "abs_open_change"] = abs(df.open_change)
    df.loc[:, "realized_vol"] = df.abs_change * multiplier

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


def ratio_analysis(x_df, y_df, periods_to_avg, return_summary=False):
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

    return ratio_summary


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

    def _calculate_streak_summary(dataframe, frequency):
        df = (
            dataframe.reset_index()
            .groupby("streak_id")
            .agg({"date": ["min", "max"], "streak_count": "max"})
            .reset_index()
        )
        df.columns = ["streak_id", "start_date", "end_date", "streak_count"]

        # Check if there is an ongoing streak
        current_streak = (
            df.iloc[-1].streak_count
            if pd.Timestamp.today().date() <= df.iloc[-1].end_date.date()
            else False
        )

        return {
            "freq": frequency,  # Use the given freq value instead of df.iloc[-1].name
            "longest_streak": df.streak_count.max(),
            "longest_streak_start": df.start_date[df.streak_count.idxmax()],
            "longest_streak_end": df.end_date[df.streak_count.idxmax()],
            "current_streak": current_streak,
            "dataframe": df,
        }

    def print_streak_summary(streak_summary):
        print(
            f"{streak_summary['freq']} - longest streak: {streak_summary['longest_streak']} from {streak_summary['longest_streak_start']:%d %b %Y} to {streak_summary['longest_streak_end']:%d %b %Y}"
        )
        if streak_summary["current_streak"]:
            print(f"Current streak: {streak_summary['current_streak']}\n")

    freqs = generate_frequency(freq)

    streaks = []
    for freq in freqs:
        df = analyser(instrument, frequency=freq)
        df = generate_streak(df, query)
        streak_summary = _calculate_streak_summary(df, freq)
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


def compute_strike_and_premium(
    close: pd.Series,
    iv: pd.Series,
    time_to_expiry: pd.Series,
    strike_offset: float,
    base: float = 100,
    label: str = "",
    transaction="buy",
):
    if label:
        label = f"{label}_"

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
        if transaction == "sell"
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
    index_symbol, return_all=False, simulate_backtest=False, strike_offset=0
):
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
                "BANKNIFTY_IV_DATA.csv",
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
            simulated_data = compute_strike_and_premium(
                index_monthly_data["index_close"],
                index_monthly_data["index_iv"],
                index_monthly_data["time_to_expiry"],
                strike_offset,
                100,
                label="index",
            )
            index_monthly_data = index_monthly_data.merge(
                simulated_data, left_index=True, right_index=True, how="left"
            )
        else:
            raise NotImplementedError

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

        index_monthly_data = index_monthly_data.merge(
            constituent_monthly_data, left_index=True, right_index=True, how="inner"
        )
        if simulate_backtest:
            index_monthly_data[f"{constituent}_iv"] = index_monthly_data[
                "iv_diff_from_mean"
            ] * (
                (index_monthly_data[f"{constituent}_abs_change"].mean() - 0.5) * 4.4
            )  # 0.5 is the adjustment factor for the mean to account for splits
            simulated_data = compute_strike_and_premium(
                index_monthly_data[f"{constituent}_close"],
                index_monthly_data[f"{constituent}_iv"],
                index_monthly_data["time_to_expiry"],
                strike_offset,
                5,
                label=constituent,
            )
            index_monthly_data = index_monthly_data.merge(
                simulated_data, left_index=True, right_index=True, how="left"
            )
    index_monthly_data["sum_constituent_movement"] = index_monthly_data.filter(
        regex="weighted"
    ).sum(axis=1)
    index_monthly_data["ratio_of_movements"] = (
        index_monthly_data["sum_constituent_movement"]
        / index_monthly_data["index_abs_change"]
    )
    if return_all:
        return index_monthly_data
    else:
        summary_df = index_monthly_data[
            ["index_abs_change", "sum_constituent_movement", "ratio_of_movements"]
        ]
        return summary_df
