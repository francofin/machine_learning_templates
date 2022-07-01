class Mlgorithmic:
    
    def __init__(self):
        import pandas as pd
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        from sqlalchemy import create_engine
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from statsmodels.tsa.seasonal import seasonal_decompose
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from collections import Counter
        import nltk
        import nltk.corpus as corpus
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.stem import WordNetLemmatizer 
        from IPython.display import Image
       
        
        self.pd = pd
        self.np = np
        self.os = os
        self.mse = mean_squared_error
        self.r2 = r2_score
        self.cf = confusion_matrix
        self.sklearn_cr = classification_report
        self.tts = train_test_split
        self.label_encoder = LabelEncoder()
        self.standard_scaler = StandardScaler()
        self.accuracy = accuracy_score
        self.counter = Counter
        self.Image = Image
        self.corpus = corpus
        self.nltk = nltk
        self.sent_tokenize = sent_tokenize
        self.word_tokenize = word_tokenize
        self.lemmatizer = WordNetLemmatizer()
        self.plt = plt
        self.mpl = mpl
        

        
    def load_data(self, path_to_file, index='', date_treatment=True):
        from pathlib import Path
        data = Path(path_to_file)
        if path_to_file[-3:] == 'csv':
            if index:
                dataframe = self.pd.read_csv(data, index_col=index, parse_dates=date_treatment)
            else:
                dataframe = self.pd.read_csv(data)
        else:
            if index:
                dataframe = self.pd.read_excel(data, index_col=index, parse_dates=date_treatment)
            else:
                dataframe = self.pd.read_excel(data)

        self.dataframe=dataframe
                
        return dataframe

    def get_returns(self, dataframe):
        """
        Requires data to be numerical either float or int
        """
        try:
            data=dataframe
            returns = dataframe.pct_change()
            data['Returns'] = returns
        except:
            raise TypeError("unsupported operand type(s) for /: 'str' and 'float', check columns after loading data \n" 
                            "this error possibly occurs because one or more of the columns in the dataset has strings as its type")
            
        return data
    
    def get_data_statistics(self, dataframe, rolling_window=20, index=''):
        """
        Requires numerical data, if non numerical data is provided then the data will be discarded and not used. if you wish to use 
        non numerical data in the case of NLP, first use vectorization or some other method which encodes text data. 
        """

        try:
            raw_data = dataframe
            data_with_returns = self.get_returns(raw_data)
            data_with_returns["1+return"] = data_with_returns+1
            data_with_standard_deviation = data_with_returns.rolling(window=rolling_window).std()
            data_with_rolling_returns = data_with_standard_deviation["1+return"].rolling(window=rolling_window).cumprod()-1
            return data_with_rolling_returns
        except:
            raise TypeError("unsupported operand type(s) for /: 'str' and 'float', check columns after loading data \n" 
                            "this error possibly occurs because one or more of the columns in the dataset has strings as its type")



    def database_connection_service(self, db_name):
        return create_engine(db_name)


    def get_query(self, query: str, engine):
        if not query:
            raise AttributeError("The query must be a string of an sql type. Please check the syntax for your query.")
        
        try:
            print(engine)
            user_engine = engine
        except:
            print("An sql databse connection must be made and passed into the get_qquery function. Use database_connection_service to create a conenction to yoyu databae if yoyu have not made one or pass your database into the prompt")
            user_db = input("please enter the database you wish to source data from, must be in the form postgresql://postgres:postgres@localhost:5432/estate_db")
            try:
                user_db_connection = self.database_connection_service(user_db)
            except:
                print("please renter your engine database correctly, trouble reading your previous connection")
                self.get_query(query, engine)
        
        data_fetched = pd.read_sql(query, user_db_connection)
        return data_fetched
        
    def decompose(self, data, method='seasonal', model='Additive'):
        '''
        params: data is a signle dataset that can be used in the analysis
        params: model is the Additive or Multiplicative model. 
        params: method is seasonal for seasonal decompose or hp for hodrick prescott. 
        '''
        from statsmodels.tsa.seasonal import seasonal_decompose
        import statsmodels.api as sm
        
        if method == 'seasonal':
            decomposed = seasonal_decompose(data, model=model)
            return decomposed
        elif model =='hp':
            ts_noise, ts_trend = sm.tsa.filters.hpfilter(df['close'])
            return ts_noise, ts_trend
        
    
    def run_adfuller(self, data):
        '''
        Presence of unit root means the time series is non-stationary.
        p < 0.05 to reject the null of unit root. If not rejected, the time series is non stationary.
        If rejected, the series is stationary. p value is result[1]
        if p < 0.05, stationary else non stationary
        '''
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(data)
        return result
    
    def arma_arima(self, data, ar=1, ma=1, i=1, lags=0, model='arma', run_acf_pacf=False):
        '''
        AR(p) makes predictions using previous values of the dependent variable, while the MA uses the mean and previous errors. 
        param: lags only needed if run_acf is true and sets the lags paramater on the pplot. 
        PACF to determine AR and ACF plot to determine MA and i
        
        '''
        from statsmodels.tsa.arima_model import ARMA
        from statsmodels.tsa.arima_model import ARIMA
        
        if model == 'arma':
            model = ARMA(data, order=(ar,ma))
            results = model.fit()
        elif model == 'arima':
            model = ARIMA(data, order=(ar, i, ar))
            results = model.fit()
        if run_acf:
            import statsmodels as sm
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            plot_acf(data, lags=lags);
            
            sm.graphics.tsaplots.plot_pacf(data, lags=lags, zero=False);
            
        return results
    
    def arch_garch_model(self, data, start_date_for_forecast='', forecast_horizon=10, ar=1, ma=1, mean="Zero", vol="GARCH", run_forecast=False):
        '''
        Forecast horizon by default is 10
        param: start date for forecast is string date representation.
        '''
        from arch import arch_model
        
        model = arch_model(returns, mean=mean, vol=vol, p=ar, q=ma)
        res = model.fit(disp="off")
        
        if run_forecast:
            forecasts = res.forecast(start=start_date_for_forecast, horizon=forecast_horizon)
            return res, forecast
        else:
            pass
        
        
        return res
    
    def runLinearRegression(self, data_x, data_y):
        '''
        Returns a tupe, of 8 values. 
        Data may require shaping. [samples, features]
        '''
        
        from sklearn.metrics import mean_squared_error, r2_score
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        import numpy as np
        model = LinearRegression()
        model.fit(data_x, data_y)
        coeff = model.coef_
        intercept = model.intercept_
        predicted_y_values = model.predict(X)
        
        score = model.score(data_x, data_y, sample_weight=None)
        r2 = r2_score(data_y, predicted_y_values)
        mse = mean_squared_error(data_y, predicted_y_values)
        rmse = np.sqrt(mse)
        std = np.std(y)
        
        return coeff, intercept, score, r2, mse, rmse, std
    
    def split_test_and_train_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
        
        return X_train, X_test, y_train, y_test
    
    def run_logistic_regression(self, data_x_train, data_y_train, data_x_test, data_y_test, solver='lbfgs'):
        '''
        solver params are based on scklearn parameters. 
        Model requires both training data and test data that have already been split.
        '''
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(solver=solver, random_state=1)
        classifier.fit(data_x_train, data_y_train)
        
        print(f"Training Data Score: {classifier.score(data_x_train, data_y_train)}")
        print(f"Testing Data Score: {classifier.score(data_x_test, data_y_test)}")
        predictions = classifier.predict(data_x_test)
        results = pd.DataFrame({"Prediction": predictions, "Actual": data_y_test})
    
    def encode_data(self, data, columns_to_encode):
        X_binary_encoded = pd.get_dummies(data, columns=columns_to_encode)
        
    def get_confusion_matrix_and_classification_report(self, data_y_test, predictions, target_names=''):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        
        cf_matrix = confusion_matrix(data_y_test, predictions)
        classification_output = classification_report(data_y_test, predictions, target_names=target_names)
        
        return cf_matrix, classification_output