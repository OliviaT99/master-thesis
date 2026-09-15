| **Variable** | **Source** | **Definition** | **Type** |
| --- | --- | --- | --- |
| `age` | clients | Customer age | numeric |
| `client_id` | clients | Customer identification number | string |
| `first_issue_date` | clients | Loyalty card issue date and time | datetime |
| `first_redeem_date` | clients | Date and time the loyalty card was activated through a qualifying purchase | datetime |
| `gender` | clients | Customer gender | categorical |
| `issue_redeem_delay` | clients | Time between loyalty card issue and first redemption | numeric |
| `treatment_flg` | uplift_train | SMS treatment assignment | binary |
| `target` | uplift_train | Post-treatment purchase outcome | binary |
| `n_transactions` | purchases | Number of transactions per customer | count |
| `total_spending` | purchases | Total spending per customer | numeric |
| `avg_basket_value` | purchases | Average spending per transaction | numeric |
| `median_basket_value` | purchases | Median spending per transaction | numeric |
| `avg_items_per_basket` | purchases | Average number of items per transaction | numeric |
| `median_items_per_basket` | purchases | Median number of items per transaction | numeric |
| `n_stores` | purchases | Number of distinct stores visited | count |
| `loyalty_points_earned` | purchases | Total loyalty points earned | numeric |
| `loyalty_points_spent` | purchases | Total loyalty points spent | numeric |
| `prop_transactions_points_earned` | purchases | Proportion of transactions in which loyalty points were earned | numeric |
| `prop_transactions_points_spent` | purchases | Proportion of transactions in which loyalty points were spent | numeric |
| `first_purchase_date` | purchases | Date and time of the customer's first purchase | datetime |
| `last_purchase_date` | purchases | Date and time of the customer's last purchase | datetime |
| `n_distinct_products` | purchases | Number of distinct products purchased | count |
| `observation_period_days` | purchases | Number of days between the customer's first and last purchase | numeric |
| `observation_period_months` | purchases | Observation period in months, calculated from the first and last purchase | numeric |
| `transactions_last_30d` | purchases | Number of transactions in the 30 days before the reference date | count |
| `spending_last_30d` | purchases | Total spending in the 30 days before the reference date | numeric |
| `transactions_last_60d` | purchases | Number of transactions in the 60 days before the reference date | count |
| `spending_last_60d` | purchases | Total spending in the 60 days before the reference date | numeric |
| `transactions_last_90d` | purchases | Number of transactions in the 90 days before the reference date | count |
| `spending_last_90d` | purchases | Total spending in the 90 days before the reference date | numeric |
| `mean_days_between_purchases` | purchases | Mean number of days between consecutive purchases | numeric |
| `median_days_between_purchases` | purchases | Median number of days between consecutive purchases | numeric |
| `min_days_between_purchases` | purchases | Minimum number of days between consecutive purchases | numeric |
| `max_days_between_purchases` | purchases | Maximum number of days between consecutive purchases | numeric |
| `std_days_between_purchases` | purchases | Standard deviation of days between consecutive purchases | numeric |
