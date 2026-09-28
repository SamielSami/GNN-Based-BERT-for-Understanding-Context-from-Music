"""Print the available Task 1 commands."""


def main() -> None:
    print(
        "Task 1 modules:\n"
        "  python -m src.task1.audit    # audit MusicCaps metadata/availability\n"
        "  python -m src.task1.data     # ID-first, training-fitted AudioSet CSV\n"
        "  python -m src.task1.train    # train/evaluate BERT\n"
        "  python -m src.task1.predict  # run checkpoint inference\n"
        "  python -m src.task1.evaluate # audit and recompute held-out reports\n"
        "\nSee docs/task1/README.md for complete commands and architecture."
    )


if __name__ == "__main__":
    main()
