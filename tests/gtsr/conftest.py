# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)
