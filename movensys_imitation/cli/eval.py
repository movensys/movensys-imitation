"""``movensys-eval``: LeRobot's ``lerobot-eval`` with the project policies registered."""

from movensys_imitation.cli._lerobot import run


def main() -> None:
    run("eval")


if __name__ == "__main__":
    main()
