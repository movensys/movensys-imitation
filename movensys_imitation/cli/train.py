"""``movensys-train``: LeRobot's ``lerobot-train`` with the project policies registered."""

from movensys_imitation.cli._lerobot import run


def main() -> None:
    run("train")


if __name__ == "__main__":
    main()
