#!/usr/bin/env python
# coding: utf-8

#Users can also create their own processor by inheriting the base class of Processor
#using the base portion from qlib credit to Microsoft

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Qlib's default data pre-process (qlib.data.dataset.processor) contains:
# 1. Sample process (DropnaProcessor, DropnaLabel)
# 2. Feature process (DropCol, FilterCol)
# 3. Irregularity value process (TanhProcess, ProcessInf)
# 4. Missing value process (Fillna, CSZFillna)
# 5. Normalization process (MinMaxNorm, ZScoreNorm, RobustZScoreNorm, CSZScoreNorm, CSRankNorm)

# For example, in Alpha158 and Alpha360, the training and validation data called DropnaLabel and CSZScoreNorm
# and testing data called ProcessInf, ZScoreNorm and Fillna (more detail in qlib.contrib.data.handler).

# One way of customize data pre-process is to change the value of ”data_handler_config“ (a dictionary)， adding two keys called
# "learn_processors" and "infer_processors", with their value being a list of dictionaries of processor wished to be utilized along any possible arguments.

# Here's an example:
data_handler_config = {
    ...
    "infer_processors": [
        {
            "class": "FilterCol"
            "kwargs": {"col_list": ["col1", "col2", "col3"]},
        },
        {
            "class": "ProcessorName"
            "kwargs": {"field_group": "label"},
        },
    ],

    "learn_processors": [
        "ProcessorName",
        {"class": "ProcessorName", "kwargs": {"field_group": "label"}}
    ],
}

# In addition to customizing data_handler_config, we could also create our own processor by INHERITING the base class of the Processor.
# Here's an example:
class DropZeroProcessor(Processor):
    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        return df[(df.T != 0).all()]
# The above code is a processor that drops all rows which contains 0.
# Customized processor could also be called in data_handler_config using the same format described above.

# Lastly, we could ultimately create our own Processor class (we are inheriting from the qlib processor until now).
# For this Processor class, we should include all the functions that our children processor (DropZeroProcessor for example) might call.
# Here's an example:
class Processor_new(Serializable)  # Note that qlib Processor interits Serializable, we should probably do the same. TODO: why is that?
    def fit(self, df: pd.DataFrame = None):
        pass

    @abc.abstractmethod  # TODO: this is a decorator. What function does it have and why use it here?
    def __call__(self, df: pd.DataFrame):
        pass

    def otherFunctions(self):
        pass
# The functions in Processor class actually serve as a "virtual function" in C/C++ and should be implemented by its children.
# That's why they return True or just pass.


# Above are codes and notes I wrote for creating a new Processor before Wednesday
# ---------------------------------------------------------------------------------------


import abc
import numpy as np
import pandas as pd
import copy


from qlib.log import TimeInspector
from datetime import datetime
from qlib.data.dataset.utils import fetch_df_by_index
from qlib.utils.serial import Serializable
from qlib.utils.paral import datetime_groupby_apply
from tsmoothie.smoother import *

EPS = 1e-12


class Processor(Serializable):
    def fit(self, df: pd.DataFrame = None):
        """
        learn data processing parameters
        Parameters
        ----------
        df : pd.DataFrame
            When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.
        """
        pass

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        process the data
        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside
        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """
        pass

    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference
        Some processors are not usable for inference.
        Returns
        -------
        bool:
            if it is usable for infenrece.
        """
        return True


class TanhProcess(Processor):
    """ Use tanh to process noise data"""

    def __call__(self, df):
        def tanh_denoise(data):
            mask = data.columns.get_level_values(1).str.contains("LABEL")
            col = df.columns[~mask]
            data[col] = data[col] - 1
            data[col] = np.tanh(data[col])

            return data

        return tanh_denoise(df)


class Fillna(Processor):
    """Process NaN"""

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            cols = get_group_columns(df, self.fields_group)
            df.fillna({col: self.fill_value for col in cols}, inplace=True)
        return df


class ProcessInf(Processor):
    """Process infinity  """

    def __call__(self, df):
        def replace_inf(data):
            def process_inf(df):
                for col in df.columns:
                    # FIXME: Such behavior is very weird
                    df[col] = df[col].replace([np.inf, -np.inf], df[col][~np.isinf(df[col])].mean())
                return df

            data = datetime_groupby_apply(data, process_inf)
            data.sort_index(inplace=True)
            return data

        return replace_inf(df)

class DropZeroProcessor(Processor):
    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        return df[(df.T != 0).all()]

class TsmoothieProcessor(Processor):
    def __call__(self, df: pd.DataFrame):
        smoother = ConvolutionSmoother(window_len=30, window_type='ones')
        smoother.smooth(df["AAPL"])
    
class FormatLevelTwo(Processor):
    def __call__(self,df):
        def extract(row):
            output = {}
            for d in row.values:
                if str(type(d)) == "<class 'str'>":
                    d = ast.literal_eval(d)
                for k in d.keys():
                    output[k] = d[k]
            return output
        def convert(timestamp):
            if timestamp != 0:
                timestamp = timestamp/1000
                newtime = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %I:%M:%S")
            else:
                newtime = np.nan
            return newtime
        def process_timestamps(df):
            df.bidtimestamp = df.bidtimestamp.apply(convert)
            df.asktimestamp = df.asktimestamp.apply(convert)
            return df
        def process_leveltwo(df):
            df = df[pd.notnull(df['0'])]
            df.columns = ["index","dict"]
            df = df.groupby('index')['dict'].apply(extract).unstack().fillna(0)
            bid_ask = df.groupby("index")["data"].apply(extract).unstack().fillna(0)
            df = df.join(bid_ask).drop("data",1)
            tempdf = df.copy()
            x = [{'price': np.nan, 'size': np.nan, 'timestamp': np.nan}]
            tempdf['bids'] = df.bids.apply(lambda y: x if len(y)==0 else y)
            tempdf['asks'] = df.asks.apply(lambda y: x if len(y)==0 else y)
            tempdf['bids'] = tempdf['bids'].str[0]
            tempdf['asks'] = tempdf['asks'].str[0]
            bidinfo = tempdf.groupby("index")["bids"].apply(extract).unstack().fillna(0)
            bidinfo.columns = ["bidprice","bidsize", "bidtimestamp"]
            askinfo = tempdf.groupby("index")["asks"].apply(extract).unstack().fillna(0)
            askinfo.columns = ["askprice","asksize", "asktimestamp"]
            tempdf = tempdf.join(bidinfo).join(askinfo).drop("bids",1).drop("asks",1)
            process_timestamps(tempdf)
            df = tempdf.copy()
            return(tempdf)
        return process_leveltwo(df)

# In[ ]:
