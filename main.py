#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:26:12 2024

@author: danieldabbah
"""
import torch
import pandas as pd
import pre_process

df = pd.read_csv("data/train.csv.gz")


cols_to_exclude = ['Unnamed: 0', 'OBJECTID', 'FOD_ID', 'FPA_ID',
                   'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID',
                   'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID',
                   'MTBS_FIRE_NAME', 'COMPLEX_NAME', 'OWNER_CODE', 'Shape']

cols_to_use = ['SOURCE_SYSTEM_TYPE', 'FIRE_YEAR', 'DISCOVERY_DATE',
               'DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_DESCR',
               'CONT_DATE', 'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE',
               'FIRE_SIZE_CLASS',  'STATE', ]


cols_to_check_value_counts = ['SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY',
                              'NWCG_REPORTING_UNIT_ID',
                              'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT',
                              'SOURCE_REPORTING_UNIT_NAME', 'FIRE_CODE',
                              'FIRE_NAME', 'OWNER_DESCR', 'COUNTY', 'FIPS_CODE', 'FIPS_NAME',
                              'LATITUDE', 'LONGITUDE'
                              ]

if __name__ == '__main__':

    df = pd.read_csv("data/train.csv.gz", usecols=cols_to_use)
    b = df[:10]
    df = pre_process.pre_process_time_cols(df)
    a = df[:500]

    # TODO: later improve the model by add text features from cols_to_check_values_counts
    # finish state col and decide what to do with fire size values.
    # create one line of text
    # maybe use fire size
# len(cols_to_check_value_counts) + len(cols_to_exclude) + len(cols_to_use)
