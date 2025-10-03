from typing import final
import pandas as pd


class DataSource:
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    def __init__(self, df: pd.DataFrame):
        self.data = df
        if not all(col in self.data.columns for col in self.required_columns):
            raise ValueError(f"Dataframe requires the following columns: {self.required_columns}")
        self.current_index = len(self.data)
        self.open_data = self.data['Open'].to_numpy()
        self.high_data = self.data['High'].to_numpy()
        self.low_data = self.data['Low'].to_numpy()
        self.close_data = self.data['Close'].to_numpy()
        self.volume_data = self.data['Volume'].to_numpy()

    @final
    def __len__(self):
        return len(self.data)
    
    @property
    def Index(self):
        return self.data.index
    
    @property
    def Open(self):
        return self.open_data[:self.current_index]
    
    @property
    def High(self):
        return self.high_data[:self.current_index]
    
    @property
    def Low(self):
        return self.low_data[:self.current_index]
    
    @property
    def Close(self):
        return self.close_data[:self.current_index]
    
    @property
    def Volume(self):
        return self.volume_data[:self.current_index]

class CSVDataSource(DataSource):
    def __init__(self, pathname: str):
        data = pd.read_csv(pathname, index_col=0, parse_dates=[0])
        super().__init__(data)