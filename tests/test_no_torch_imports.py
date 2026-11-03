import subprocess
import sys

import pytest

DATA_MODULES = [
    "sprout.data.check_splits",
    "sprout.data.class_report",
    "sprout.data.field",
    "sprout.data.grouping",
    "sprout.data.inventory",
    "sprout.data.masks",
    "sprout.data.splits",
    "sprout.segment.data",
    "sprout.segment.crop",
]


@pytest.mark.parametrize("module", DATA_MODULES)
def test_importing_a_data_module_does_not_load_torch(module):
    code = f"import sys, {module}; sys.exit(1 if 'torch' in sys.modules else 0)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, f"{module} loaded torch\n{done.stderr}"
