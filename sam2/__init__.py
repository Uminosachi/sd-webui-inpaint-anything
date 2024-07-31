# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from hydra import initialize_config_dir, initialize_config_module  # noqa: F401

inpa_basedir = os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
configs_path = os.path.join(inpa_basedir, "sam2_configs")

initialize_config_dir(configs_path, version_base="1.2")
# initialize_config_module("sam2_configs", version_base="1.2")
