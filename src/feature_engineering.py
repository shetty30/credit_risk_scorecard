import pandas as pd
import numpy as np


def engineer_features(df):
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs()
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs()
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / 365
    df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'] / 365
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['EMPLOYMENT_RATIO'] = df['EMPLOYMENT_YEARS'] / df['AGE_YEARS']
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_MIN'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['BUREAU_ENQUIRY_TOTAL'] = (
        df['AMT_REQ_CREDIT_BUREAU_DAY'] +
        df['AMT_REQ_CREDIT_BUREAU_WEEK'] +
        df['AMT_REQ_CREDIT_BUREAU_MON'] +
        df['AMT_REQ_CREDIT_BUREAU_QRT']
    )
    df['HAS_SOCIAL_CIRCLE_DEFAULT'] = (
        df['DEF_30_CNT_SOCIAL_CIRCLE'] > 0
    ).astype(int)
    return df


def winsorize(df, cols, lower=0.01, upper=0.99):
    for col in cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def compute_woe(df, feature, target, bins=10):
    binned = pd.qcut(df[feature], q=bins, duplicates='drop')
    grouped = df.groupby(binned)[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']
    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()
    grouped['dist_events'] = grouped['events'] / total_events
    grouped['dist_non_events'] = grouped['non_events'] / total_non_events
    grouped['woe'] = np.log(
        grouped['dist_events'] / grouped['dist_non_events'].replace(0, np.nan)
    )
    grouped['iv'] = (
        grouped['dist_events'] - grouped['dist_non_events']
    ) * grouped['woe']
    iv_total = grouped['iv'].sum()
    return grouped, iv_total