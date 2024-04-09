#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:26:12 2024

@author: danieldabbah
"""
import torch
import pandas as pd
import pre_process


cols_to_exclude = ['Unnamed: 0', 'OBJECTID', 'FOD_ID', 'FPA_ID',
                   'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID',
                   'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID',
                   'MTBS_FIRE_NAME', 'COMPLEX_NAME', 'OWNER_CODE', 'Shape']

cols_to_use = ['SOURCE_SYSTEM_TYPE', 'DISCOVERY_DATE',
               'DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_DESCR',
               'CONT_DATE', 'CONT_DOY', 'CONT_TIME',   'STATE', ]


cols_to_check_value_counts = ['SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY',
                              'NWCG_REPORTING_UNIT_ID',
                              'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT',
                              'SOURCE_REPORTING_UNIT_NAME', 'FIRE_CODE',
                              'FIRE_NAME', 'OWNER_DESCR', 'COUNTY', 'FIPS_CODE', 'FIPS_NAME',
                              'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'FIRE_YEAR'
                              ]

if __name__ == '__main__':
    k = 1000
    df = pd.read_csv("data/train.csv.gz", usecols=cols_to_use)[:9*k]
    validation = pd.read_csv("data/validation.csv.gz",
                             usecols=cols_to_use)[:3*k]
    test = pd.read_csv("data/test_1.csv.gz", usecols=cols_to_use)[:k]

    b = df[:10]
    df = pre_process.pre_process_time_cols(df)
    a = df[:500]

    df.columns

    df["text"][0]
    validation = pre_process.pre_process_time_cols(validation)
    test = pre_process.pre_process_time_cols(test)
    # TODO: later improve the model by add text features from cols_to_check_values_counts
    k = test[:200]
    # create hugging face data sets
    # start tokenize like they did in the book


# len(cols_to_check_value_counts) + len(cols_to_exclude) + len(cols_to_use)
    len(test['label'].value_counts())
    len(validation['label'].value_counts())
    len(df['label'].value_counts())

    from datasets import Dataset, Features, ClassLabel, Value

    unique_classes = sorted(df['label'].unique())

    unique_classes
    features = Features({
        'label': ClassLabel(names=unique_classes),
        'text': Value('string')
        # Include other columns as needed, e.g., 'text': Value('string')
    })

    features
    train_ds = Dataset.from_pandas(df, features=features)
    val_ds = Dataset.from_pandas(validation, features=features)
    test_ds = Dataset.from_pandas(test, features=features)

    from datasets import DatasetDict

    dataset_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds
    })


dataset_dict
