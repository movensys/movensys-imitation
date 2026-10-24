from glob import glob
import os

from setuptools import find_packages, setup

package_name = "movensys_imitation_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config", "dobot_cr3a"),
         glob("config/dobot_cr3a/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mohsin Ali (MVSK)",
    maintainer_email="mali@movensys.com",
    description="PushT imitation-learning nodes for the Dobot CR3A (guard, recorder, manager, "
                "scorer, policy runner).",
    license="MIT License",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Added in plan Phase 2:
            "planar_guard = movensys_imitation_ros.planar_guard:main",
            # "episode_recorder = movensys_imitation_ros.episode_recorder:main",
            # "episode_manager = movensys_imitation_ros.episode_manager:main",
            # "pusht_scorer = movensys_imitation_ros.pusht_scorer:main",
            # "policy_runner = movensys_imitation_ros.policy_runner:main",
        ],
    },
)
