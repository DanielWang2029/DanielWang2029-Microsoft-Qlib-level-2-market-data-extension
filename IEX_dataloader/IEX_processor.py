# !/usr/bin/env python
# coding: utf-8

# In[5]:


# Users can also create their own processor by inheriting the base class of Processor
# using the base portion from qlib credit to Microsoft


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import numpy as np
import pandas as pd
import copy

from qlib.log import TimeInspector
from qlib.data.dataset.utils import fetch_df_by_index
from qlib.utils.serial import Serializable
from qlib.utils.paral import datetime_groupby_apply

EPS = 1e-12


class Processor():
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

    def tanh_denoise(data):
        mask = data.columns.get_level_values(1).str.contains("LABEL")
        col = df.columns[~mask]
        data[col] = data[col] - 1
        data[col] = np.tanh(data[col])

        return data

    return tanh_denoise(df)

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

# In[ ]: