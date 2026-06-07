CREATE TABLE fact_loans (
    loan_id             VARCHAR(20) PRIMARY KEY,
    borrower_id         VARCHAR(20) NOT NULL,
    product_id          VARCHAR(10) NOT NULL,
    date_id             INT NOT NULL,
    loan_amount         DECIMAL(15,2),
    funded_amount       DECIMAL(15,2),
    interest_rate       DECIMAL(5,2),
    term_months         INT,
    installment         DECIMAL(10,2),
    default_flag        TINYINT,
    days_past_due       INT,
    recovery_amount     DECIMAL(15,2)
);

CREATE TABLE dim_borrowers (
    borrower_id         VARCHAR(20) PRIMARY KEY,
    annual_income       DECIMAL(15,2),
    employment_length   INT,
    home_ownership      VARCHAR(20),
    state               CHAR(2),
    credit_score        INT,
    open_accounts       INT,
    delinquencies_2yr   INT,
    revol_utilization   DECIMAL(5,2),
    total_accounts      INT,
    public_records      INT
);

CREATE TABLE dim_products (
    product_id          VARCHAR(10) PRIMARY KEY,
    loan_purpose        VARCHAR(50),
    grade               CHAR(1),
    sub_grade           VARCHAR(3)
);

CREATE TABLE dim_time (
    date_id             INT PRIMARY KEY,
    issue_date          DATE,
    year                INT,
    quarter             INT,
    month               INT,
    vintage_year        INT,
    vintage_quarter     VARCHAR(7)
);
