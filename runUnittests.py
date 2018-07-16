import unittest

from tests.tMPCTracker import TestTracker
runner = unittest.TextTestRunner()
result = runner.run(unittest.makeSuite(TestTracker))
