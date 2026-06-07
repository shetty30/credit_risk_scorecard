import pandas as pd
import numpy as np


class LoanDataCleaner:

    def __init__(self, null_threshold=0.40):
        self.null_threshold = null_threshold
        self.drop_cols_ = []
        self.median_values_ = {}
        self.high_risk_orgs_ = []
        self.medium_risk_orgs_ = []

    def remove_high_nulls(self, df):
        null_pct = df.isnull().mean()
        self.drop_cols_ = null_pct[null_pct > self.null_threshold].index.tolist()
        return df.drop(columns=self.drop_cols_)

    def clean_gender(self, df):
        df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'F')
        return df

    def fill_occupation(self, df):
        df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')
        return df

    def fill_suite(self, df):
        df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna(
            df['NAME_TYPE_SUITE'].mode()[0]
        )
        return df

    def build_org_risk_tiers(self, df):
        org_dr = df.groupby('ORGANIZATION_TYPE')['TARGET'].mean()
        self.high_risk_orgs_ = org_dr[org_dr > 0.10].index.tolist()
        self.medium_risk_orgs_ = org_dr[
            (org_dr >= 0.07) & (org_dr <= 0.10)
        ].index.tolist()
        return self

    def apply_org_risk_tiers(self, df):
        def categorize(org):
            if org in self.high_risk_orgs_:
                return 'High Risk'
            elif org in self.medium_risk_orgs_:
                return 'Medium Risk'
            return 'Low Risk'
        df['ORG_RISK_TIER'] = df['ORGANIZATION_TYPE'].apply(categorize)
        return df

    def encode_categoricals(self, df):
        binary_map = {'Y': 1, 'N': 0}
        df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map(binary_map)
        df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map(binary_map)
        df = pd.get_dummies(df,
                            columns=['NAME_CONTRACT_TYPE', 'CODE_GENDER',
                                     'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                                     'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                                     'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
                                     'ORG_RISK_TIER'],
                            drop_first=True)
        df = df.drop(columns=['ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START'],
                     errors='ignore')
        return df

    def impute_numerics(self, df, fit=True):
        amt_req_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
        df[amt_req_cols] = df[amt_req_cols].fillna(0)
        df['EXT_SOURCE_3_MISSING'] = df['EXT_SOURCE_3'].isnull().astype(int)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if c != 'TARGET']
        if fit:
            self.median_values_ = df[num_cols].median().to_dict()
        for col in num_cols:
            df[col] = df[col].fillna(self.median_values_.get(col, 0))
        return df

    def fit_transform(self, df):
        df = self.remove_high_nulls(df)
        df = self.clean_gender(df)
        df = self.fill_occupation(df)
        df = self.fill_suite(df)
        self.build_org_risk_tiers(df)
        df = self.apply_org_risk_tiers(df)
        df = self.encode_categoricals(df)
        df = self.impute_numerics(df, fit=True)
        return df

    def transform(self, df):
        df = df.drop(columns=self.drop_cols_, errors='ignore')
        df = self.clean_gender(df)
        df = self.fill_occupation(df)
        df = self.fill_suite(df)
        df = self.apply_org_risk_tiers(df)
        df = self.encode_categoricals(df)
        df = self.impute_numerics(df, fit=False)
        return df