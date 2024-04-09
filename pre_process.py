#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:35:07 2024

@author: danieldabbah
"""
import pandas as pd
import numpy as np
from datetime import datetime
import datetime
import calendar


def fill_and_convert_julian_to_gregorian(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Preprocesses a pandas DataFrame by calculating the average Julian date for non-null values in a specified column,
    filling null values with this average, and converting Julian dates to Gregorian dates.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: the name of the column containing Julian dates to process.

    Returns:|
    - A pandas DataFrame with the specified column's null values filled with the average Julian date and converted to Gregorian dates.
    """

    # Step 1: Calculate the average Julian date for non-null values
    average_julian_date = df[column_name].dropna().mean()

    # Step 2: Fill nulls in the specified column with the calculated average
    df[column_name].fillna(average_julian_date, inplace=True)

    # Step 3: Apply the Julian to Gregorian conversion logic to the column
    df[column_name] = df[column_name].apply(lambda julian_date: datetime.datetime.fromtimestamp(
        (julian_date - 2440587.5) * 86400.0).strftime('%Y-%m-%d'))

    return df


def impute_and_format_doy(df, column_name='CONT_DOY'):
    """
    Imputes missing values in the day-of-year column and formats the column by converting to integers.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: String name of the column with day-of-year values.

    Returns:
    - A modified pandas DataFrame with missing day-of-year values imputed and all values formatted as integers.
    """

    # Calculate the median day-of-year, ignoring NaN values
    median_doy = df[column_name].median()

    # Fill NaN values with the median day-of-year
    df[column_name].fillna(median_doy, inplace=True)

    # Convert day-of-year values to integers to remove any decimals
    df[column_name] = df[column_name].astype(int)

    return df


def categorize_and_impute_time(df, time_column):
    # Convert all non-NaN values to strings for consistent processing
    df[time_column] = df[time_column].apply(
        lambda x: f"{int(x):04}" if not pd.isnull(x) else np.nan)

    # Calculate the median time, ignoring NaN values
    median_time = df[time_column].dropna().astype(int).median()

    # Convert the median time to a string in HHMM format
    median_time_str = f"{int(median_time):04}"

    # Fill NaN values with the median time
    df[time_column].fillna(median_time_str, inplace=True)

    # Convert time to HH:MM format for readability
    df[time_column] = df[time_column].apply(lambda x: f"{x[:2]}:{x[2:]}")
    # Function to categorize time into part of the day

    def categorize_time(time_str):
        hour = int(time_str[:2])
        if 5 <= hour <= 11:
            return 'Morning'
        elif 12 <= hour <= 16:
            return 'Afternoon'
        elif 17 <= hour <= 20:
            return 'Evening'
        else:  # Covers night and early morning hours
            return 'Night'

    # Apply categorization to each time
    df['CATEGORY_'+time_column] = df[time_column].apply(categorize_time)
    return df


def get_season_optimized(date):
    month = date.month
    day = date.day
    if month in (1, 2, 12):
        season = 'Winter'
    elif month in (3, 4, 5):
        season = 'Spring'
    elif month in (6, 7, 8):
        season = 'Summer'
    else:
        season = 'Fall'  # Renaming Autumn to Fall

    # Adjusting for days in transitional months
    if month == 3 and day >= 21:
        season = 'Spring'
    elif month == 6 and day >= 21:
        season = 'Summer'
    elif month == 9 and day >= 23:
        season = 'Fall'
    elif month == 12 and day >= 21:
        season = 'Winter'

    return season


# Function to convert minutes into days, hours, and minutes
def convert_minutes_to_textual_representation(minutes):
    days = int(minutes // 1440)
    hours = int((minutes % 1440) // 60)
    minutes_left = int((minutes % 1440) % 60)

    result = ""
    if days > 0:
        result += f"{days} day{'s' if days > 1 else ''}, "
    if hours > 0 or days > 0:  # Include hours if there are also days
        result += f"{hours} hour{'s' if hours != 1 else ''}, "
    result += f"{minutes_left} minute{'s' if minutes_left != 1 else ''}"

    return result


def pre_process_time_cols(df):

    # convering from julian to gregorian
    # this if for time when the fire was discover
    df = fill_and_convert_julian_to_gregorian(df, 'DISCOVERY_DATE')
    df = impute_and_format_doy(df, 'DISCOVERY_DOY')
    df['disc_time'] = df['DISCOVERY_TIME']
    df = categorize_and_impute_time(df, 'DISCOVERY_TIME')

    # Fill NaN values with the median:
    df['disc_time'].fillna(df['disc_time'].median(), inplace=True)

    # this is when the fire was under control
    df = fill_and_convert_julian_to_gregorian(df, 'CONT_DATE')
    df = impute_and_format_doy(df, 'CONT_DOY')
    df = categorize_and_impute_time(df, 'CONT_TIME')

    # Applying the optimized function to the dataframe
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])

    # finding the season of the year
    df['SEASON'] = df['DISCOVERY_DATE'].apply(get_season_optimized)

    # the time difrence when the fire was discover to when it is under control
    df['DAYS_DIFFERENCE'] = df.apply(lambda row: row['CONT_DOY'] - row['DISCOVERY_DOY'] if row['CONT_DOY']
                                     >= row['DISCOVERY_DOY'] else (365 - row['DISCOVERY_DOY']) + row['CONT_DOY'], axis=1)
    # Step 1: Convert DISCOVERY_TIME and CONT_TIME to timedelta objects
    df['DISCOVERY_TIME_DELTA'] = pd.to_timedelta(
        df['DISCOVERY_TIME'].str.strip() + ':00')
    df['CONT_TIME_DELTA'] = pd.to_timedelta(
        df['CONT_TIME'].str.strip() + ':00')

    # Step 2: Calculate the initial time difference in minutes
    df['TIME_DIFF_MINUTES'] = (
        df['CONT_TIME_DELTA'] - df['DISCOVERY_TIME_DELTA']).dt.total_seconds() / 60

    # Step 3: Adjust for the difference in days, assuming 1440 minutes in a day (24 hours * 60 minutes)
    df['ADJUSTED_TIME_DIFF_MINUTES'] = df.apply(lambda row: row['TIME_DIFF_MINUTES'] if row['CONT_DOY'] >= row['DISCOVERY_DOY']
                                                else row['TIME_DIFF_MINUTES'] + (1440 * (row['CONT_DOY'] - row['DISCOVERY_DOY'] + 365)), axis=1)
    # Step 2: Calculate the initial time difference in minutes
    df['TIME_DIFF_MINUTES'] = df['TIME_DIFF_MINUTES'] + \
        df['DAYS_DIFFERENCE']*24*60
    df = df.drop(['CONT_TIME_DELTA', 'DISCOVERY_TIME_DELTA',
                 'ADJUSTED_TIME_DIFF_MINUTES', 'CATEGORY_CONT_TIME', 'DAYS_DIFFERENCE'], axis=1)

    df = df.drop(columns=['CONT_DATE', 'CONT_DOY',
                 'CONT_TIME', 'DISCOVERY_DOY', 'disc_time'])

    df['discovery_d_of_month'] = df['DISCOVERY_DATE'].dt.day
    df['discovery_month'] = df['DISCOVERY_DATE'].dt.month
    df = df.drop(['DISCOVERY_DATE'], axis=1)
    df = df.drop(['DISCOVERY_TIME'], axis=1)

    def day_suffix(day):
        if 0 < day and day < 4:
            return str(day) + ["st", "nd", "rd"][day - 1]
        elif day < 32:
            return str(day) + "th"
        else:
            raise Exception("Invalid day!")

    # Combine month and day with textual representations and correct suffixes
    df['Date'] = df.apply(
        lambda x: calendar.month_name[x['discovery_month']] + " " +
        day_suffix(x['discovery_d_of_month']), axis=1)

    df = df.drop(columns=['discovery_d_of_month', 'discovery_month'])

    # if the fire duration is 0, I decided to use arbitray 10 minutes duration.
    df.loc[df['TIME_DIFF_MINUTES'] == 0] = 10
    df['Fire duration'] = df['TIME_DIFF_MINUTES'].apply(
        convert_minutes_to_textual_representation)

    df = df.drop(columns=['TIME_DIFF_MINUTES'])

    state_mapping = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    df["STATE"] = df["STATE"].map(state_mapping)
    df.rename(columns={'CATEGORY_DISCOVERY_TIME': 'Time of day',
              'SOURCE_SYSTEM_TYPE': 'Source System',
                       'STATE': 'State', 'SEASON': 'Season',
                       'STAT_CAUSE_DESCR': 'label'}, inplace=True)

    features_cols = [col for col in df.columns if col != 'label']
    df['label'] = df['label'].astype(str)

    def get_text_samle(x):
        sample = ""
        for col in features_cols[:-1]:
            sample += f"{col} is: {x[col]}, "
        col = features_cols[-1]
        sample += f"{col} is: {x[col]}"
        return sample

    df["text"] = df.apply(get_text_samle, axis=1)
    df = df.drop(columns=features_cols)

    return df
