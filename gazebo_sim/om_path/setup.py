from setuptools import find_packages, setup

package_name = "om_path"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/om_path"]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OpenMind",
    maintainer_email="hello@openmind.com",
    description="Lidar-based obstacle avoidance path computation for OM1 simulation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "om_path = om_path.om_path_node:main",
        ],
    },
)
