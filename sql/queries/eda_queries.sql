SELECT 
    p.grade,
    COUNT(*) AS total_loans,
    ROUND(SUM(f.default_flag) * 100.0 / COUNT(*), 2) AS default_rate_pct
FROM fact_loans f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY p.grade
ORDER BY p.grade;

SELECT 
    CASE 
        WHEN b.credit_score < 580 THEN 'Poor (< 580)'
        WHEN b.credit_score < 670 THEN 'Fair (580-669)'
        WHEN b.credit_score < 740 THEN 'Good (670-739)'
        WHEN b.credit_score < 800 THEN 'Very Good (740-799)'
        ELSE 'Exceptional (800+)'
    END AS credit_band,
    COUNT(*) AS total_loans,
    ROUND(SUM(f.default_flag) * 100.0 / COUNT(*), 2) AS default_rate_pct,
    AVG(f.interest_rate) AS avg_rate
FROM fact_loans f
JOIN dim_borrowers b ON f.borrower_id = b.borrower_id
GROUP BY credit_band
ORDER BY default_rate_pct DESC;

SELECT 
    p.loan_purpose,
    COUNT(*) AS total_loans,
    ROUND(SUM(f.default_flag) * 100.0 / COUNT(*), 2) AS default_rate_pct,
    SUM(f.loan_amount) AS total_exposure
FROM fact_loans f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY p.loan_purpose
ORDER BY default_rate_pct DESC;

SELECT 
    CASE
        WHEN b.employment_length < 1 THEN '< 1 year'
        WHEN b.employment_length < 3 THEN '1-3 years'
        WHEN b.employment_length < 5 THEN '3-5 years'
        WHEN b.employment_length < 10 THEN '5-10 years'
        ELSE '10+ years'
    END AS emp_band,
    COUNT(*) AS total_loans,
    ROUND(SUM(f.default_flag) * 100.0 / COUNT(*), 2) AS default_rate_pct
FROM fact_loans f
JOIN dim_borrowers b ON f.borrower_id = b.borrower_id
GROUP BY emp_band
ORDER BY default_rate_pct DESC;

SELECT
    t.vintage_quarter,
    COUNT(*) AS originations,
    SUM(f.default_flag) AS defaults,
    ROUND(SUM(f.default_flag) * 100.0 / COUNT(*), 2) AS default_rate_pct
FROM fact_loans f
JOIN dim_time t ON f.date_id = t.date_id
GROUP BY t.vintage_quarter
ORDER BY t.vintage_quarter;

SELECT TOP 10
    b.state,
    COUNT(*) AS loan_count,
    ROUND(SUM(f.default_flag) * 100.0 / COUNT(*), 2) AS default_rate_pct,
    SUM(f.loan_amount) AS total_exposure
FROM fact_loans f
JOIN dim_borrowers b ON f.borrower_id = b.borrower_id
GROUP BY b.state
ORDER BY total_exposure DESC;