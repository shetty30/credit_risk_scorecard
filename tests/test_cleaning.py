import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.data_cleaning import LoanDataCleaner


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'TARGET': [0, 1, 0, 1, 0],
        'CODE_GENDER': ['M', 'F', 'XNA', 'M', 'F'],
        'FLAG_OWN_CAR': ['Y', 'N', 'Y', 'N', 'Y'],
        'FLAG_OWN_REALTY': ['Y', 'Y', 'N', 'N', 'Y'],
        'OCCUPATION_TYPE': ['Laborers', None, 'Sales staff', None, 'Drivers'],
        'NAME_TYPE_SUITE': ['Unaccompanied', None, 'Family', 'Unaccompanied', None],
        'ORGANIZATION_TYPE': ['Business Entity Type 3', 'Transport: type 3',
                              'Government', 'Construction', 'School'],
        'NAME_CONTRACT_TYPE': ['Cash loans', 'Revolving loans',
                               'Cash loans', 'Cash loans', 'Revolving loans'],
        'NAME_INCOME_TYPE': ['Working', 'Pensioner', 'Working',
                             'Commercial associate', 'Working'],
        'NAME_EDUCATION_TYPE': ['Secondary / secondary special',
                                'Higher education', 'Secondary / secondary special',
                                'Higher education', 'Secondary / secondary special'],
        'NAME_FAMILY_STATUS': ['Married', 'Single / not married',
                               'Married', 'Married', 'Single / not married'],
        'NAME_HOUSING_TYPE': ['House / apartment', 'With parents',
                              'House / apartment', 'House / apartment', 'With parents'],
        'WEEKDAY_APPR_PROCESS_START': ['MONDAY', 'TUESDAY', 'WEDNESDAY',
                                       'THURSDAY', 'FRIDAY'],
        'EXT_SOURCE_2': [0.5, 0.3, None, 0.7, 0.4],
        'EXT_SOURCE_3': [0.6, None, 0.4, 0.8, None],
        'AMT_REQ_CREDIT_BUREAU_DAY': [0, None, 1, 0, None],
        'AMT_REQ_CREDIT_BUREAU_WEEK': [0, None, 0, 1, None],
        'AMT_REQ_CREDIT_BUREAU_MON': [1, None, 0, 0, None],
        'AMT_REQ_CREDIT_BUREAU_QRT': [0, None, 0, 0, None],
        'AMT_REQ_CREDIT_BUREAU_HOUR': [0, None, 0, 0, None],
        'AMT_REQ_CREDIT_BUREAU_YEAR': [2, None, 1, 3, None],
        'HIGH_NULL_COL': [None, None, None, None, 0.1],
        'AMT_INCOME_TOTAL': [100000, 200000, 150000, 80000, 120000],
        'AMT_CREDIT': [500000, 300000, 400000, 200000, 350000],
        'AMT_ANNUITY': [25000, 15000, 20000, 10000, 17000],
        'AMT_GOODS_PRICE': [450000, 280000, 380000, 180000, 320000],
        'CNT_FAM_MEMBERS': [2, 1, 3, 2, 4],
        'DAYS_BIRTH': [-10000, -15000, -12000, -8000, -20000],
        'DAYS_EMPLOYED': [-2000, -500, -3000, -1000, -4000],
        'DEF_30_CNT_SOCIAL_CIRCLE': [0, 1, 0, 2, 0],
        'OBS_30_CNT_SOCIAL_CIRCLE': [2, 3, 1, 4, 2],
        'DEF_60_CNT_SOCIAL_CIRCLE': [0, 0, 0, 1, 0],
        'OBS_60_CNT_SOCIAL_CIRCLE': [1, 2, 1, 3, 1],
        'DAYS_LAST_PHONE_CHANGE': [-100, -200, -150, -50, -300],
    })


def test_removes_high_null_columns(sample_df):
    cleaner = LoanDataCleaner(null_threshold=0.40)
    result = cleaner.remove_high_nulls(sample_df)
    assert 'HIGH_NULL_COL' not in result.columns


def test_gender_xna_replaced(sample_df):
    cleaner = LoanDataCleaner()
    result = cleaner.clean_gender(sample_df)
    assert 'XNA' not in result['CODE_GENDER'].values


def test_occupation_nulls_filled(sample_df):
    cleaner = LoanDataCleaner()
    result = cleaner.fill_occupation(sample_df)
    assert result['OCCUPATION_TYPE'].isnull().sum() == 0


def test_org_risk_tiers_created(sample_df):
    cleaner = LoanDataCleaner()
    cleaner.build_org_risk_tiers(sample_df)
    result = cleaner.apply_org_risk_tiers(sample_df)
    assert 'ORG_RISK_TIER' in result.columns
    assert set(result['ORG_RISK_TIER'].unique()).issubset(
        {'High Risk', 'Medium Risk', 'Low Risk'}
    )


def test_no_nulls_after_fit_transform(sample_df):
    cleaner = LoanDataCleaner(null_threshold=0.40)
    result = cleaner.fit_transform(sample_df)
    assert result.isnull().sum().sum() == 0


def test_ext_source_missing_flag_created(sample_df):
    cleaner = LoanDataCleaner(null_threshold=0.40)
    result = cleaner.fit_transform(sample_df)
    assert 'EXT_SOURCE_3_MISSING' in result.columns


def test_weekday_column_dropped(sample_df):
    cleaner = LoanDataCleaner(null_threshold=0.40)
    result = cleaner.fit_transform(sample_df)
    assert 'WEEKDAY_APPR_PROCESS_START' not in result.columns


def test_binary_encoding(sample_df):
    cleaner = LoanDataCleaner(null_threshold=0.40)
    result = cleaner.fit_transform(sample_df)
    assert result.dtypes.get('FLAG_OWN_CAR') != object