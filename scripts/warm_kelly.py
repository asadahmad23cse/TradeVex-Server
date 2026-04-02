from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk.kelly_warm_start import BTCKellyWarmStart


def main() -> None:
    kelly = BTCKellyWarmStart()
    kelly.load_from_history("data/signal_history.json")

    print("\n=== Kelly Warm-Start Report ===")
    print(f"Total trades loaded: {kelly.total_trades}")
    print(f"Buckets populated:   {len(kelly.buckets)}")
    print("\nBucket breakdown:")
    for key, stats in sorted(kelly.buckets.items()):
        print(f"\n  Bucket: {key}")
        print(f"    Trades     : {stats['total']}")
        print(f"    Win Rate   : {stats['win_rate']:.1%}")
        print(f"    Avg Win    : +{stats['avg_win_pct']:.2f}%")
        print(f"    Avg Loss   : -{stats['avg_loss_pct']:.2f}%")
        print(f"    Expectancy : {stats['expectancy']:+.3f}%")
        print(f"    Smoothing  : {stats['smoothing_weight']:.0%} empirical")

    print("\n--- Sample Position Sizes ---")
    test_cases = [
        ("LONG", 85, "BULLISH TREND", "STRONG"),
        ("LONG", 72, "BULLISH TREND", "MODERATE"),
        ("LONG", 60, "SIDEWAYS", "WEAK"),
        ("SHORT", 85, "BEARISH TREND", "STRONG"),
        ("SHORT", 72, "BEARISH TREND", "MODERATE"),
    ]
    for sig, conf, regime, strength in test_cases:
        r = kelly.compute_btc_position(sig, conf, regime)
        print(
            f"  {sig} {strength:8} ({conf}% | {regime:15}): "
            f"{r['position_size_pct']:.2f}% | "
            f"Method: {r['method']}"
        )

    kelly.save_buckets()
    print("\nSaved -> data/kelly_buckets.json")


if __name__ == "__main__":
    main()
