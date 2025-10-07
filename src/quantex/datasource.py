import math
from typing import final
import numpy as np
import pandas as pd

class DataSource:
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    def __init__(self, df: pd.DataFrame, train_test_split: bool = False, mode: str = "train"):
        self.data = df
        if (train_test_split and mode == "train"):
            index = math.floor(len(df.index) * 0.8)
            self.data = self.data.iloc[:index]
        elif (train_test_split and mode == "test"):
            index = math.floor(len(df.index) * 0.8)
            self.data = self.data.iloc[index:]
        if not all(col in self.data.columns for col in self.required_columns):
            raise ValueError(f"Dataframe requires the following columns: {self.required_columns}")
        self.current_index = len(self.data)
        self.open_data = np.ascontiguousarray(self.data['Open'].to_numpy(), dtype=np.float64)
        self.high_data = np.ascontiguousarray(self.data['High'].to_numpy(), dtype=np.float64)
        self.low_data = np.ascontiguousarray(self.data['Low'].to_numpy(), dtype=np.float64)
        self.close_data = np.ascontiguousarray(self.data['Close'].to_numpy(), dtype=np.float64)
        self.volume_data = np.ascontiguousarray(self.data['Volume'].to_numpy(), dtype=np.float64)

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
    
    @property
    def COpen(self) -> np.float64:
        return self.open_data[self.current_index]
    
    @property
    def CHigh(self) -> np.float64:
        return self.high_data[self.current_index]
    
    @property
    def CLow(self) -> np.float64:
        return self.low_data[self.current_index]
    
    @property
    def CClose(self) -> np.float64:
        return self.close_data[self.current_index]
    
    @property
    def CVolume(self) -> np.float64:
        return self.volume_data[self.current_index]

class CSVDataSource(DataSource):
    def __init__(self, pathname: str, train_test_split: bool = False, mode: str = "train"):
        data = pd.read_csv(pathname, index_col=0, parse_dates=[0])
        super().__init__(data, train_test_split, mode)