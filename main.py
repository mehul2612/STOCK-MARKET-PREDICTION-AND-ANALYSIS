import streamlit as st
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
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

app_name="Stock Market Forecasting App"
st.title(app_name)
st.subheader("This is created to forecast the stock market price of selected company")
#image from online source
st.image("https://media.istockphoto.com/id/1369016721/vector/finance-background-illustration-with-abstract-stock-market-information-and-charts-over-world.jpg?s=612x612&w=0&k=20&c=MJ1PYhhX7IOfdXfR7crnlW2vJI94lE9KpI3tKY2joHg=")

#input from user for start and end date
start= datetime.date(2020, 1, 1)
end = datetime.date(2021, 12, 31)
st.sidebar.header("Select the parameter from below")
start_date = st.sidebar.date_input('Start Date', start)
end_date=st.sidebar.date_input('End Date',end)

#add ticker symbol list
ticker_list=["AAPL","MSFT","GOOGL","GOOG","META","TSLA","NVDA","TSLA","ADBE","PYPL","INTC",
             "CMCSA","NFLX","PEP"
             ] 
ticker=st.sidebar.selectbox("Select the compnay",ticker_list)

pricing_data,news=st.tabs(['Pricing Data','Top 10 News'])

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
    fig=px.line(data,x="Date",y=data.columns,title="Closing price of stock",width=1000,height=600)
    st.plotly_chart(fig)

    # add a select box to select column from data
    # column=st.selectbox("Select the column to be used for forecasting",data.columns[1:])

    # #subsetting the data
    # data=data[["Date",column]]
    # st.write("Selected Data")
    # st.write(data)
    # ...

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

    #decompose the data
    st.header("Decomposition of the data")
    decomposition=seasonal_decompose(data[column],model='additive',period=12)
    # st.write(decomposition.plot())
    #make the same plot in plotly
    # st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x':'Date','y':'Price'}))
    # st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal, title='Seasonality', width=1000, height=400, labels={'x':'Date','y':'Price'}))
    # st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid, title='Residual', width=1000, height=400, labels={'x':'Date','y':'Price'}))
    # Plot Trend with a specific color
    fig_trend = px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x':'Date','y':'Price'})
    fig_trend.update_traces(line=dict(color='#FF5733'))  # Use a different color for trend
    st.plotly_chart(fig_trend)

    # Plot Seasonality with a specific color
    fig_seasonality = px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1000, height=400, labels={'x':'Date','y':'Price'})
    fig_seasonality.update_traces(line=dict(color='#33FF57'))  # Use a different color for seasonality
    st.plotly_chart(fig_seasonality)

    # Plot Residual with a specific color
    fig_residual = px.line(x=data["Date"], y=decomposition.resid, title='Residual', width=1000, height=400, labels={'x':'Date','y':'Price'})
    fig_residual.update_traces(line=dict(color='#5733FF'))  # Use a different color for residual
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

    # predict the feature values(Forecasting)
    forecast_period=st.number_input('## Select the number of days to forecast',1,365,10)
    prediction=model.get_prediction(start=len(data),end=len(data)+forecast_period-1)
    prediction=prediction.predicted_mean

    prediction.index=pd.date_range(start=end_date, periods=len(prediction),freq='D')
    prediction=pd.DataFrame(prediction)
    prediction.insert(0,'Date',prediction.index)
    prediction.reset_index(drop=True,inplace=True)
    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Display the table in the first column
    with col1:
        st.write("## Predicted Data",prediction)

    # Display the plot in the second column
    with col2:
        st.write("## Actual Data",data)

    # lets plot the data
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=data["Date"], y=data[column], name="Actual",line=dict(color='yellow')))
    fig_forecast.add_trace(go.Scatter(x=prediction['Date'], y=prediction['predicted_mean'], name="Forecast", line=dict(color='green')))

    fig_forecast.update_layout(title='Actual vs Forecasted', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
    st.plotly_chart(fig_forecast)

    # add buttons to show and hide seperate plots
    show_plots=False
    if st.button("Show Seperate Plots"):
        if not show_plots:
            st.write(px.line(x=data['Date'],y=data[column],title='Actual',width=1200,height=400,labels={'x':'Date','y':'Price'}))
            st.write(px.line(x=prediction['Date'],y=prediction['predicted_mean'],title='Forecasted',width=1200,height=400,labels={'x':'Date','y':'Price'}))
            show_plots=True
        else:
            show_plots=False

    # add hide plot button
    hide_plots=False
    if st.button("Hide Separate Plots"):
        if not hide_plots:
            hide_plots=True
        else:
            hide_plots=False



from stocknews import StockNews
with news:
    st.header(f'News of {ticker}')
    sn=StockNews(ticker_list,save_news=False)
    df_news=sn.read_rss()
    for i in range(15):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment=df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment=df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')


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
