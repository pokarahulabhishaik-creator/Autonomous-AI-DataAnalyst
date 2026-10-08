"""
Analytics Knowledge Base for AutoAnalyst AI.
Contains curated statistical, analytical, and business interpretation guidelines.
"""

KNOWLEDGE_DOCUMENTS = [
    {
        "id": "stat_corr_causation",
        "category": "statistics",
        "title": "Correlation Does Not Imply Causation",
        "content": (
            "Principle: A correlation (positive or negative Pearson/Spearman r) indicates a mathematical association, "
            "not a causal mechanism. Confounding variables, reverse causality, or coincidence can produce high correlation. "
            "Guidance: Always explicitly state that correlation indicates co-movement or statistical association, never claim "
            "one variable causes another without controlled A/B experiment data or verified causal inference models."
        )
    },
    {
        "id": "stat_p_value",
        "category": "statistics",
        "title": "Statistical Significance and P-Values",
        "content": (
            "Principle: In hypothesis tests (t-test, ANOVA, chi-square), a p-value < 0.05 indicates that the observed difference "
            "is unlikely to have occurred solely due to random sampling chance under the null hypothesis. "
            "Guidance: A statistically significant difference is not necessarily practically or economically significant. "
            "Always report effect size or magnitude alongside the p-value."
        )
    },
    {
        "id": "stat_simpsons_paradox",
        "category": "statistics",
        "title": "Simpson's Paradox and Aggregation Bias",
        "content": (
            "Principle: A trend or relationship observed in an aggregated dataset can reverse or disappear when the data is disaggregated "
            "by subgroups (e.g. regions, customer tiers, product categories). "
            "Guidance: Before drawing sweeping conclusions from dataset-wide averages, drill down into major categorical dimensions "
            "to verify whether the trend holds consistently across segments."
        )
    },
    {
        "id": "stat_outliers",
        "category": "statistics",
        "title": "Outlier Detection and Handling",
        "content": (
            "Principle: Outliers are values that fall outside expected distribution ranges (e.g. beyond 1.5 * IQR or > 3 standard deviations). "
            "Guidance: Never delete outliers indiscriminately. Distinguish between data entry errors and genuine extreme business events "
            "(such as enterprise whale purchases, massive order returns, or severe operational failures). "
            "Use median and interquartile range (IQR) for skewed distributions instead of mean and standard deviation."
        )
    },
    {
        "id": "kpi_profit_margin",
        "category": "business_kpi",
        "title": "Profit Margin and Discount Sensitivity",
        "content": (
            "Principle: Profit Margin = (Profit / Revenue) * 100%. Aggressive promotional discounting often increases top-line sales volume "
            "while devastating bottom-line net profit. "
            "Guidance: When analyzing declining profits alongside growing or stable sales, inspect the Discount column and Cost/Margin ratios. "
            "If discounts exceed gross margins, sales volume increases result in accelerating net dollar losses."
        )
    },
    {
        "id": "kpi_churn_retention",
        "category": "business_kpi",
        "title": "Customer Churn and Lifetime Value (LTV)",
        "content": (
            "Principle: Churn Rate = Customers Lost / Total Customers. Monthly recurring revenue retention is heavily influenced by "
            "contract length, onboarding support tickets, and pricing tier stability. "
            "Guidance: Customers on month-to-month contracts with high support ticket volumes within their first 90 days represent "
            "the highest churn hazard. Focus proactive interventions on this cohort."
        )
    },
    {
        "id": "ml_evaluation",
        "category": "machine_learning",
        "title": "ML Model Selection and Evaluation Metrics",
        "content": (
            "Principle: In regression, R-squared measures variance explained (0 to 1), while RMSE measures absolute dollar/unit error. "
            "In classification with imbalanced classes (e.g. Churn or Fraud), Accuracy is misleading; rely on F1-Score, Precision, and Recall. "
            "Guidance: Always test on unseen holdout data (Train-Test Split). Document model limitations and feature importance rank."
        )
    },
    {
        "id": "ml_clustering",
        "category": "machine_learning",
        "title": "Customer Segmentation via Clustering",
        "content": (
            "Principle: K-Means partitions data into k distinct clusters based on Euclidean distance across normalized numerical features. "
            "Guidance: Evaluate cluster separation using the Silhouette Score (values closer to +1 indicate tight, distinct clusters). "
            "Characterize each cluster by its mean profile (e.g. 'High Spend, Low Frequency' vs 'Low Spend, High Frequency')."
        )
    },
    {
        "id": "viz_chart_selection",
        "category": "visualization",
        "title": "Visualization Selection Framework",
        "content": (
            "Guidelines: "
            "1. Time Series / Trends: Continuous line chart with date on X-axis. "
            "2. Category Comparison / Rankings: Bar chart (horizontal if category labels are long). "
            "3. Numerical Distributions: Histogram with bins or box plot showing quartiles and outliers. "
            "4. Relationships / Correlation: Scatter plot with trendline or correlation heatmap matrix. "
            "5. Composition / Proportions: Donut chart or treemap for hierarchical data (limit to < 7 slices)."
        )
    },
    {
        "id": "data_quality_framework",
        "category": "data_engineering",
        "title": "Data Quality Scoring Framework",
        "content": (
            "Principle: Data Quality Score is a weighted composite of Completeness (missing value ratio), Uniqueness (duplicate row ratio), "
            "Validity (proper data types and formatting), and Consistency (absence of impossible extreme values). "
            "Guidance: High quality datasets (>85%) allow confident automated analysis. Lower scores require explicit cleaning and cautionary disclaimers."
        )
    }
]
