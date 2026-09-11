from pathlib import Path

from sklift.datasets import fetch_x5


RAW_DIR = Path("data/raw")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    x5 = fetch_x5()

    # Save clients
    x5["data"]["clients"].to_csv(
        RAW_DIR / "clients.csv",
        index=False,
    )

    # Reconstruct and save uplift training data
    uplift_train = x5["data"]["train"].copy()
    uplift_train["treatment_flg"] = x5["treatment"].values
    uplift_train["target"] = x5["target"].values

    uplift_train.to_csv(
        RAW_DIR / "uplift_train.csv",
        index=False,
    )

    # Save purchases
    #x5["data"]["purchases"].to_csv(
    #    RAW_DIR / "purchases.csv",
    #    index=False,
    #)

    print("Saved:")
    print(f"  {RAW_DIR / 'clients.csv'}")
    print(f"  {RAW_DIR / 'uplift_train.csv'}")
    #print(f"  {RAW_DIR / 'purchases.csv'}")


if __name__ == "__main__":
    main()