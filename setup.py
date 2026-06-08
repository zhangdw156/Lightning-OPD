# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import platform
import sys

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


# Custom wheel class to modify the wheel name
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"

        if platform.system() == "Linux":
            platform_tag = "manylinux1_x86_64"
        else:
            platform_tag = platform.system().lower()

        return python_version, abi_tag, platform_tag


setup(cmdclass={"bdist_wheel": bdist_wheel})
