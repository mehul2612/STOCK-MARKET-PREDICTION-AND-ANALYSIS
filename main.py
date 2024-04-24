import streamlit as st
st.set_page_config(layout="wide")
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

custom_css = """
    <style>
        body {
            font-family: 'Dancing Script', cursive;
            font-size: 16px;
            color: white;
            text-align: left;
        }
        h1 {
            font-family: 'Dancing Script', cursive;
            font-size: 36px;
            color: white;
        }
        h2 {
            font-family: 'Dancing Script', cursive;
            font-size: 28px;
            color: white;
        }
        h3 {
            font-family: 'Dancing Script', cursive;
            font-size: 24px;
            color: white;
        }
        body {
            text-align: left;
        }
        .write {
            font-family: 'Dancing Script', cursive;
            font-size: 16px;
            color: white;
        
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


app_name="Stock Market Forecasting App"
st.title(app_name)
st.subheader("This is created to forecast the stock market price of selected company")
#image from online source
st.image("https://media.istockphoto.com/id/1369016721/vector/finance-background-illustration-with-abstract-stock-market-information-and-charts-over-world.jpg?s=612x612&w=0&k=20&c=MJ1PYhhX7IOfdXfR7crnlW2vJI94lE9KpI3tKY2joHg=")

#input from user for start and end date
start= datetime.date(2024, 1, 1)
end = datetime.date(2024,4,16)
st.sidebar.header("Select the parameter from below")
start_date = st.sidebar.date_input('Start Date', start)
end_date=st.sidebar.date_input('End Date',end)

#add ticker symbol list
ticker_list=["AAPL","MSFT","GOOGL","GOOG","META","TSLA","NVDA","TSLA","ADBE","PYPL","INTC",
             "CMCSA","NFLX","PEP"
             ] 
ticker=st.sidebar.selectbox("Select the compnay",ticker_list)

pricing_data,news=st.tabs(['Stocks Pricing Data','Top Stocks News'])

with pricing_data:
    # Fetch data from user input using library
    data=yf.download(ticker,start=start_date,end=end_date)
    data.insert(0,"Date",data.index,True)
    data.reset_index(drop=True,inplace=True)
    st.write('  ### Data from',start_date,'to',end_date)
    st.write(data)


    #plot the data
    st.header("Data Visualization")
    st.subheader('Plot the data')
    st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and selet your specific column")
    # fig=px.line(data,x="Date",y=data.columns,title="Closing price of stock",width=1000,height=600)
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     increasing_line_color='cyan',
                                     decreasing_line_color='magenta')])
    fig.update_layout(width=1000)
    st.plotly_chart(fig)

   


    # add a select box to select column from data
    column = st.selectbox("Select the column to be used for forecasting", data.columns[1:])

    # subsetting the data
    selected_data = data[["Date", column]]
    # st.write("### Selected Data")

    # Plot line chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Line(x=selected_data["Date"], y=selected_data[column], mode='lines+markers', name='Line Chart',line=dict(color='lightgreen')))
    fig.update_layout(title=f'Line Chart for {column}', xaxis_title='Date', yaxis_title=column, width=500, height=400)

    # Display the selected data and line chart side by side
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Selected Data")
        st.write(selected_data)

    with col2:
        st.write("### Line Chart")
        st.plotly_chart(fig)


    # ADF test check stationarity
    st.header("Is data Stationary?")
    st.write("**Note:** If p-value is less than 0.05, then data is stationary")
    st.write(adfuller(data[column])[1]<0.05)

    
    st.header("Decomposition of the data")
    decomposition=seasonal_decompose(data[column],model='additive',period=12)

    
    # Plot Trend 
    fig_trend = px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x':'Date','y':'Price'})
    fig_trend.update_traces(line=dict(color='#FF5733')) 
    st.plotly_chart(fig_trend)

    
    fig_seasonality = px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1000, height=400, labels={'x':'Date','y':'Price'})
    fig_seasonality.update_traces(line=dict(color='#33FF57')) 
    st.plotly_chart(fig_seasonality)

    
    fig_residual = px.line(x=data["Date"], y=decomposition.resid, title='Residual', width=1000, height=400, labels={'x':'Date','y':'Price'})
    fig_residual.update_traces(line=dict(color='#5733FF'))  
    st.plotly_chart(fig_residual)

    # Run the model
    #user input for three parameters of the model and seasonal
    p=st.slider('Select the value of p',0,5,2)
    d=st.slider('Select the value of d',0,5,1)
    q=st.slider('Select the value of q',0,5,2)
    seasonal_order=st.number_input('Select the value of seasonal p',0,24,12)

    model=sm.tsa.statespace.SARIMAX(data[column],order=[p,d,q],seasonal_order=[p,d,q,seasonal_order])
    model=model.fit()

    # #print model summary
    # st.header('Model Summary')
    # st.write(model.summary())
    # st.write("------")
    # Calculate MACD
    short_period = 12
    long_period = 26
    signal_period = 9

    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line

# Plot MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data['Date'], y=macd_line, mode='lines', name='MACD Line'))
    fig_macd.add_trace(go.Scatter(x=data['Date'], y=signal_line, mode='lines', name='Signal Line'))
    fig_macd.add_trace(go.Bar(x=data['Date'], y=macd_histogram, name='MACD Histogram'))
    fig_macd.update_layout(title='MACD (Moving Average Convergence Divergence)',
                        xaxis_title='Date',
                        yaxis_title='MACD',
                        width=1000,
                        height=400)
    st.plotly_chart(fig_macd)

    # Calculate Bollinger Bands
    window = 20
    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=window).std()

    # Plot Bollinger Bands
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig_bb.add_trace(go.Scatter(x=data['Date'], y=data['Upper Band'], mode='lines', name='Upper Band', line=dict(color='green')))
    fig_bb.add_trace(go.Scatter(x=data['Date'], y=data['Middle Band'], mode='lines', name='Middle Band', line=dict(color='blue')))
    fig_bb.add_trace(go.Scatter(x=data['Date'], y=data['Lower Band'], mode='lines', name='Lower Band', line=dict(color='red')))
    fig_bb.update_layout(title='Bollinger Bands',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        width=1000,
                        height=400)
    st.plotly_chart(fig_bb)


    # predict the feature values(Forecasting)
    forecast_period=st.number_input('## Select the number of days to forecast',1,365,10)
    prediction=model.get_prediction(start=len(data),end=len(data)+forecast_period-1)
    prediction=prediction.predicted_mean

    prediction.index=pd.date_range(start=end_date, periods=len(prediction),freq='D')
    prediction=pd.DataFrame(prediction)
    prediction.insert(0,'Date',prediction.index)
    prediction.reset_index(drop=True,inplace=True)
 

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Candlestick(x=data['Date'],
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'],
                                        increasing_line_color='cyan',
                                        decreasing_line_color='magenta',
                                        name="Actual"))


    fig_forecast.add_trace(go.Candlestick(x=prediction['Date'],
                                        open=prediction['predicted_mean'],
                                        high=prediction['predicted_mean'],
                                        low=prediction['predicted_mean'],
                                        close=prediction['predicted_mean'],
                                        increasing_line_color='green',
                                        decreasing_line_color='red',
                                        name="Forecast"))


    fig_forecast.update_layout(title='Actual vs Forecasted',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            width=1000,
                            height=400)


    st.plotly_chart(fig_forecast)

        
    short_window = 50
    long_window = 200

    # Compute short and long moving averages
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    # Generate buy/sell signals
    data['Signal'] = 0
    data['Signal'][short_window:] = \
        np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)

    # Calculate daily returns
    data['Return'] = data['Close'].pct_change()

    # Calculate strategy returns
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Return']

    # Assess performance
    strategy_return = data['Strategy_Return'].cumsum().iloc[-1]

    if strategy_return > 0:
        decision = "Buy"
    elif strategy_return < 0:
        decision = "Sell"
    else:
        decision = "Hold"

    st.markdown(f"<span style='font-size: 20px;'>It is recommended to: {decision}</span>", unsafe_allow_html=True)



with news:
    from stocknews import StockNews
    import streamlit as st

    

    st.header('Latest News of Stocks')
    sn = StockNews(ticker_list, save_news=False)
    df_news = sn.read_rss()

    
    st.markdown(
        """
        <style>
        .buy-button { background-color: green; color: white; }
        .sell-button { background-color: red; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

    for i in range(50):
        # st.subheader(f'News {i+1}')
        st.subheader(df_news['title'][i])
        st.write(df_news['published'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        # st.write(f'Title Sentiment: {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        # st.write(f'News Sentiment: {news_sentiment}')

        
        if news_sentiment >= 0.5:
            st.markdown(f'<button class="buy-button">{f"Good to Buy"}</button>', unsafe_allow_html=True)
        else:
            st.markdown(f'<button class="sell-button">{f"Bad to Buy"}</button>', unsafe_allow_html=True)




st.write('----------------------------------------------------------------------------------------------')
st.write("### About me:")
st.write("<p style='color:red; font-weight:bold; font-size:50px;'>Mehul Chauhan</p>",unsafe_allow_html=True)
st.write("## Connect with me on social media")
import streamlit as st

linkedin_url = "https://img.icons8.com/color/48/000000/linkedin.png"
github_url = "https://img.icons8.com/color/48/000000/github.png"
instagram_url = "https://img.icons8.com/color/48/000000/instagram-new.png"
linkedin_redirect_url = "https://www.linkedin.com/in/mehul-chauhan-5950481ba/"
github_redirect_url = "https://github.com/mehul2612"
instagram_redirect_url = "https://www.instagram.com/mehul._.26/?next=%2F"

st.markdown(
    f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
    f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>'
    f'<a href="{instagram_redirect_url}"><img src="{instagram_url}" width="60" height="60"></a>',
    unsafe_allow_html=True
)
st.write('----------------------------------------------------------------------------------------------')

