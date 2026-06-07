CREATE VIEW v_geo_concentration AS
SELECT 
    b.state,
    COUNT(*) AS loan_count,
    SUM(f.default_flag) AS defaults,
    SUM(f.loan_amount) AS total_exposure,
    ROUND(SUM(f.default_flag) * 100.0 / COUNT(*), 2) AS default_rate_pct,
    ROUND(SUM(f.loan_amount) * 100.0 / SUM(SUM(f.loan_amount)) OVER(), 2) AS exposure_pct
FROM fact_loans f
JOIN dim_borrowers b ON f.borrower_id = b.borrower_id
GROUP BY b.state
ORDER BY default_rate_pct DESC;