# data_cleaning.py
import pandas as pd
import numpy as np

def clean_2018_data(df):
    # Đổi tên các cột và xóa các cột không cần thiết
    df = df.rename(columns={'Your level': 'level', 'Current Salary': 'salary'})
    df = df.drop(['Salary two years ago', 'Are you getting any Stock Options?'], axis=1)
    
    # Thêm các cột thiếu với giá trị NaN
    df['main tech'] = np.nan
    df['Number of vacation days'] = np.nan
    df['Сontract duration'] = np.nan
    return df

def clean_2019_data(df):
    # Đổi tên các cột và xóa các cột không cần thiết
    df = df.rename(columns={
        'Zeitstempel': 'Timestamp',
        'Position (without seniority)': 'Position',
        'Seniority level': 'level',
        'Yearly brutto salary (without bonus and stocks)': 'salary',
        'Yearly brutto salary (without bonus and stocks) one year ago. Only answer if staying in same country': 'Salary one year ago',
        'Your main technology / programming language': 'main tech'
    })
    df = df.drop([
        'Company name ',
        'Yearly bonus',
        'Yearly stocks',
        'Yearly bonus one year ago. Only answer if staying in same country',
        'Yearly stocks one year ago. Only answer if staying in same country',
        'Number of home office days per month',
        '0',
        'Company business sector'
    ], axis=1)
    return df

def clean_2020_data(df):
    # Đổi tên các cột và xóa các cột không cần thiết
    df = df.rename(columns={
        'Total years of experience': 'Years of experience',
        'Seniority level': 'level',
        'Yearly brutto salary (without bonus and stocks) in EUR': 'salary',
        'Annual brutto salary (without bonus and stocks) one year ago. Only answer if staying in the same country': 'Salary one year ago',
        'Your main technology / programming language': 'main tech'
    })
    df = df.drop([
        'Yearly bonus + stocks in EUR',
        'Annual bonus+stocks one year ago. Only answer if staying in same country',
        'Have you lost your job due to the coronavirus outbreak?',
        'Have you been forced to have a shorter working week (Kurzarbeit)? If yes, how many hours per week',
        'Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR',
        'Employment status',
        'Years of experience in Germany',
        'Other technologies/programming languages you use often'
    ], axis=1)
    return df
