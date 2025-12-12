# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


import yfinance as yf, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def fetch_closes(tickers, start, end):
  data = {}
  for t in tickers:
      tk = yf.Ticker(t)
      df = tk.history(start=start, end=end, auto_adjust=True)   # daily OHLCV with DatetimeIndex
      # keep only adjusted close for returns analysis
      df = df[['Close','Volume']].rename(columns={'Close':'Close_'+t})
      df_clean = df[df["Volume"] > 0].copy()
      data[t] = df_clean
      # join into one DataFrame of close prices
  closes = pd.concat([data[t]['Close_'+t] for t in tickers], axis=1)
  closes.index = pd.to_datetime(closes.index)
  return(closes)

def fetch_closes_sum(tickers,weights, start, end):
  data = {}
  for t in tickers:
      tk = yf.Ticker(t)
      df = tk.history(start=start, end=end, auto_adjust=True)   # daily OHLCV with DatetimeIndex
      # keep only adjusted close for returns analysis
      df = df[['Close','Volume']].rename(columns={'Close':'Close_'+t})
      df_clean = df[df["Volume"] > 0].copy()
      data[t] = df_clean
      # join into one DataFrame of close prices
  closes = pd.concat([data[t]['Close_'+t] for t in tickers], axis=1)
  closes = closes.dot(weights)
  closes.index = pd.to_datetime(closes.index)
  return(pd.DataFrame(closes))



def simple_returns(prices):
    return prices.pct_change().dropna()

def cum_return_from_returns(returns):
    return (1 + returns).cumprod() - 1

def annualize_mean_std(ret_series, periods=252):
    mean = ret_series.mean() * periods
    std = ret_series.std() * np.sqrt(periods)
    return mean, std

def max_drawdown(price_series):
    roll_max = price_series.cummax()
    drawdown = (price_series - roll_max) / roll_max
    dd = drawdown.min()
    # find start and end dates of that drawdown
    end = drawdown.idxmin()
    start = price_series[:end].idxmax()
    return dd, start, end

def metrics_in_window(prices, returns, ticker, start, end):
  s = start; e = end
  price_w = prices.loc[s:e, ticker]
  ret_w = returns.loc[s:e, ticker]
  cumr = float((1 + ret_w).prod() - 1) if not ret_w.empty else np.nan
  ann_mean, ann_std = annualize_mean_std(ret_w.dropna())
  sharpe = ann_mean / ann_std if ann_std!=0 else np.nan
  dd, dd_start, dd_end = max_drawdown(price_w) if len(price_w)>0 else (np.nan,None,None)
  return dict(cumulative_return=cumr, annual_return=ann_mean, annual_vol=ann_std,
              sharpe=sharpe, max_drawdown=dd, dd_start=dd_start, dd_end=dd_end,
              n_days = (price_w.index[-1]-price_w.index[0]).days+1 if len(price_w)>0 else 0)
  



# ---------- UI helpers ----------
@st.cache_data(show_spinner=False)
def get_closes(tickers, start, end):
    return fetch_closes(tickers, start, end)

@st.cache_data(show_spinner=False)
def get_portfolio_closes(tickers, weights, start, end):
    return fetch_closes_sum(tickers, weights, start, end)

def make_cum_traces(df, name_prefix=None):
    # df: DataFrame of cumulative returns
    traces = []
    for col in df.columns:
        traces.append(go.Scatter(x=df.index, y=df[col].values, name=(col if not name_prefix else f"{name_prefix}-{col}"), mode="lines"))
    return traces

# ---------- Streamlit layout ----------
st.set_page_config(layout="wide", page_title="Returns Dashboard")
st.title("Portfolio vs Tick ers — Returns Dashboard")

with st.sidebar:
    tickers_input = st.text_input("Tickers to compare (comma separated)", value="SPY, QQQ, GLD")
    start = st.date_input("Start date", value=datetime(2010, 1, 1),
        min_value = datetime(2000, 1, 1))
    end = st.date_input("End date")
    portfolio_tickers_input = st.text_input("Portfolio tickers (comma)", value="AAPL,MSFT")
    weights_input = st.text_input("Portfolio weights (comma)", value="0.5,0.5")
    run = st.button("Refresh")

if not run:
    st.info("Set parameters in the sidebar and click Refresh.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
portfolio_tickers = [t.strip().upper() for t in portfolio_tickers_input.split(",") if t.strip()]
weights = [float(w.strip()) for w in weights_input.split(",")]

if len(portfolio_tickers) != len(weights):
    st.error("Number of portfolio tickers must match weights.")
    st.stop()

# Fetch data (cached)
closes = get_closes(tickers, start, end)
closes_port = get_portfolio_closes(portfolio_tickers, weights, start, end)

# Compute returns
returns = simple_returns(closes)
# ensure portfolio closes are a Series with a name
if hasattr(closes_port, "columns") and len(closes_port.columns) == 1:
    closes_port = closes_port.iloc[:, 0]
returns_port = simple_returns(closes_port)

# cumulative returns
cum_returns = cum_return_from_returns(returns)
cum_returns_port = cum_return_from_returns(returns_port.dropna())

# Top row: cumulative returns figure
st.subheader("Cumulative Returns")
fig = go.Figure()
fig.add_traces(make_cum_traces(cum_returns))
# add portfolio trace
fig.add_trace(go.Scatter(x=cum_returns_port.index, y=cum_returns_port.values,
                         name="Portfolio", mode="lines", line=dict(width=3, dash="dash")))
fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Cumulative return")
st.plotly_chart(fig, use_container_width=True)

# Row with rolling volatility
st.subheader("30-day Rolling Annualized Volatility")
rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
rolling_vol_port = returns_port.rolling(window=30).std() * np.sqrt(252)
fig2 = go.Figure()
for col in rolling_vol.columns:
    fig2.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[col], name=col))
# for col in rolling_vol_port.columns:
#     fig2.add_trace(go.Scatter(x=rolling_vol_port.index, y=rolling_vol_port[col], name='Portfolio'))
st.plotly_chart(fig2, use_container_width=True)

# Correlation heatmap (plotly)
st.subheader("Return Correlation (incl. Portfolio)")
all_returns = pd.concat([returns, returns_port.rename("PORTFOLIO")], axis=1)
corr = all_returns.corr()
fig3 = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix")
st.plotly_chart(fig3, use_container_width=True)


st.success("Dashboard updated")
