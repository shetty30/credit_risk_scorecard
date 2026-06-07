CREATE VIEW v_default_by_grade AS
SELECT 
    p.grade,
    COUNT(*) AS total_loans,
    SUM(f.default_flag) AS defaults,
    ROUND(SUM(f.default_flag) * 100.0 / COUNT(*), 2) AS default_rate_pct,
    AVG(f.interest_rate) AS avg_interest_rate,
    AVG(b.credit_score) AS avg_credit_score,
    SUM(f.loan_amount) AS total_exposure
FROM fact_loans f
JOIN dim_borrowers b ON f.borrower_id = b.borrower_id
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY p.grade
ORDER BY p.grade;