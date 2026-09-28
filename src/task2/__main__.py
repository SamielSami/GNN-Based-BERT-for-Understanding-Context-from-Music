"""Print Task 2 entry points."""


def main() -> None:
    print(
        "Task 2 modules:\n"
        "  python -m src.task2.manifest    # build an audio/label manifest\n"
        "  python -m src.task2.preprocess  # cache features, graphs, and mels\n"
        "  python -m src.task2.graphs      # build one graph for inspection\n"
        "  python -m src.task2.train       # train GNN, MLP, or CNN\n"
        "  python -m src.task2.evaluate    # evaluate a saved checkpoint\n"
        "  python -m src.task2.audit       # verify all saved graphs\n"
        "  python -m src.task2.visualize   # export 20 graph comparisons\n"
        "  python -m src.task2.experiment  # run the five-model controlled suite\n"
        "\nSee docs/task2/README.md for the complete workflow."
    )


if __name__ == "__main__":
    main()
