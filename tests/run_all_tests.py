import unittest
import os
import re
import sys


def get_module_name_by_path(path):
    curdir = os.getcwd()
    assert path.startswith(curdir + "/")
    assert path.endswith(".py")
    return path[len(curdir) + 1 : -3].replace("/", ".")


def run_all_tests(base_dir):
    test_file_pattern = re.compile(r"^test_.+\.py$")
    errors = []
    n_test = 1

    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            for root, _, files in os.walk(base_dir):
                for filename in files:
                    if re.match(test_file_pattern, filename):
                        filepath = os.path.join(root, filename)
                        module_name = get_module_name_by_path(filepath)
                        print(f"{n_test:3d} {module_name}", file=sys.stdout, flush=True)
                        module_errors, module_failures = run_tests_in_file(module_name)
                        errors.extend(module_errors)
                        errors.extend(module_failures)
                        n_test += 1
        finally:
            sys.stderr = old_stderr

    if len(errors) == 0:
        print("OK", file=sys.stderr)
    else:
        print(f"{len(errors)} ERRORS: ", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)


def run_tests_in_file(module_name):
    suite = unittest.TestSuite()

    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(module_name, globals(), locals(), ["suite"])
        suitefn = getattr(mod, "suite")
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        tests = unittest.defaultTestLoader.loadTestsFromName(module_name)
        suite.addTest(tests)

    res = unittest.TextTestRunner().run(suite)
    return res.errors, res.failures


if __name__ == "__main__":
    run_all_tests(base_dir=os.path.join(os.getcwd()))
