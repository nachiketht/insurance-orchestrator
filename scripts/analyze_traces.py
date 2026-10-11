"""Write docs/cost_latency.md from the saved local traces."""

from underwriting.analysis import write_report


def main() -> None:
    print(write_report())


if __name__ == "__main__":
    main()
