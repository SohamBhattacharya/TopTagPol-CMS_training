# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import setuptools


setuptools.setup(
    name="my_pr_curve",
    version="1.0.0",
    description="Sample TensorBoard plugin.",
    packages=["my_pr_curve"],
    package_data={
        "my_pr_curve": ["tf_pr_curve_dashboard/**"],
    },
    entry_points={
        "tensorboard_plugins": [
            "my_plugins = my_pr_curve.pr_curves_plugin:PrCurvesPlugin",
        ],
    },
)

