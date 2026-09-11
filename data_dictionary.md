| **Variable**      | **Source**   | **Definition**                 | **Type**    |
| ----------------  | ------------ | ------------------------------ | ----------- |
| `age`             | clients      | customer age                   | numeric     |
| `client_id`       | clients      | customer identification number | string      |
| `first_issue_date`| clients      | loyalty card issue date & time | string      |
| `first_redeem_date`| clients      | date & time loyalty card was activated through a qualifying purchase| string |
| `gender`          | clients      | customer gender                | categorical |
|** `issue_redeem_delay`| clients | time delay between card issue and first activation | |
| `treatment_flg`   | uplift_train | SMS assignment                 | binary      |
| `target`          | uplift_train | post-treatment purchase        | binary      |
|**`n_transactions`  | purchases    | historical transaction count   | numeric     |
| | | | |
