#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:26:12 2024

@author: danieldabbah
"""
import torch
import pandas as pd

df = pd.read_csv("data/train.csv.gz")

a = df[:10]
cols_to_exclude = ['Unnamed: 0', 'OBJECTID', 'FOD_ID', 'FPA_ID',
                   'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID']

cols_to_use = ['SOURCE_SYSTEM_TYPE',]


cols_to_check_value_counts = ['SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY',
                              'NWCG_REPORTING_UNIT_ID',
                              'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT',
                              'SOURCE_REPORTING_UNIT_NAME',]

a.columns

# , 'FIRE_CODE', 'FIRE_NAME',
#     'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID', 'MTBS_FIRE_NAME',
#     'COMPLEX_NAME', 'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY',
#     'DISCOVERY_TIME', 'STAT_CAUSE_DESCR', 'CONT_DATE', 'CONT_DOY',
#     'CONT_TIME', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'LATITUDE', 'LONGITUDE',
#     'OWNER_CODE', 'OWNER_DESCR', 'STATE', 'COUNTY', 'FIPS_CODE',
#     'FIPS_NAME', 'Shape'],


# COLUMNS_TO_USE = ['FIRE_SIZE', 'FIRE_SIZE_CLASS', 'FIRE_YEAR', 'DISCOVERY_DATE',
#                   'DISCOVERY_DOY', 'DISCOVERY_TIME', 'CONT_DATE', 'CONT_DOY',
#                   'CONT_TIME', 'STAT_CAUSE_DESCR',  # The most important Feature
# ]
