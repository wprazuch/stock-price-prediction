# Stock Price Prediction

![Stock Market](static/stock_market.jpg)

The repository contains a solution for the price prediction problem for stock data. Using deep neural networks, a model was created that quite nicely predicts stock price data.

## Data
In this mini project, Tata stock data was used ([link](https://data-flair.training/blogs/download-tata-global-beverages-stocks-data/)). The data is in csv format, contains Opening, Closing, High and Low prices for trading days between 2014 and 2018.

## Description
You can check the workflow in `notebook.ipynb` file. This notebook starts from loading and visualization, then moves to preprocessing/preparation, next it describes how to build the model and finally shows some results. In the notebook, some pickled objects will be generated which may be used later on.

In `app.py` there is a Dash app that visualizes predictions and ground truth values of the Tata stock prices. You may launch it by typing:
```
python app.py
```
and then open the app in the browser.

### Endnote
You may check some of my other projects [here](https://wprazuch.github.io/).
