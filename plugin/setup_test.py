"""Tests that setup.py correctly parses requirements.in and configures packaging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl.testing import absltest

from google3.third_party.xprof.plugin import setup


class SetupTest(absltest.TestCase):

  def test_find_requirements_file_exists(self):
    req_file = setup._find_requirements_file()
    self.assertTrue(os.path.exists(req_file))
    self.assertTrue(os.path.isfile(req_file))
    self.assertTrue(req_file.endswith('requirements.in'))

  def test_required_packages_not_empty_and_valid(self):
    self.assertIsInstance(setup.REQUIRED_PACKAGES, list)
    self.assertNotEmpty(setup.REQUIRED_PACKAGES)
    for req in setup.REQUIRED_PACKAGES:
      self.assertIsInstance(req, str)
      self.assertTrue(req.strip())
      self.assertNotIn('#', req)
      self.assertFalse(req.startswith('-'))

  def test_required_packages_contains_expected_dependencies(self):
    expected_deps = [
        'absl-py >= 2.1.0',
        'gviz_api >= 1.10.0',
        'setuptools >= 70.1.1',
        'fsspec[gcs] >= 2024.10.0',
        'cheroot >= 10.0.1',
        'etils[epath] >= 1.0.0',
        'werkzeug >= 0.11.15',
        'protobuf >= 3.19.6',
        'six >= 1.10.0',
        'google-cloud-storage >= 3.12.0',
        'urllib3 >= 2.7.0',
        'fire >= 0.4.0',
    ]
    for dep in expected_deps:
      self.assertIn(
          dep,
          setup.REQUIRED_PACKAGES,
          f'Expected dependency {dep} not found in REQUIRED_PACKAGES:'
          f' {setup.REQUIRED_PACKAGES}',
      )

  def test_parse_requirements_custom_content(self):
    test_content = (
        '# Header comment\n'
        '\n'
        'absl-py >= 2.1.0\n'
        '  fsspec[gcs] >= 2024.10.0  # Inline comment\n'
        '# Another comment\n'
        '--extra-index-url https://example.com/pypi\n'
        '-f ./wheels\n'
        'setuptools >= 70.1.1\n'
        '\n'
        'etils[epath] >= 1.0.0\n'
    )
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.in', delete=False
    ) as temp_file:
      temp_file.write(test_content)
      temp_path = temp_file.name

    try:
      parsed = setup.parse_requirements(temp_path)
      expected = [
          'absl-py >= 2.1.0',
          'fsspec[gcs] >= 2024.10.0',
          'setuptools >= 70.1.1',
          'etils[epath] >= 1.0.0',
      ]
      self.assertEqual(parsed, expected)
    finally:
      if os.path.exists(temp_path):
        os.remove(temp_path)

  def test_parse_requirements_file_not_found(self):
    with self.assertRaises(FileNotFoundError):
      setup.parse_requirements('/nonexistent/path/to/requirements.in')

  def test_get_readme(self):
    readme = setup.get_readme()
    self.assertIsInstance(readme, str)
    self.assertNotEmpty(readme)


if __name__ == '__main__':
  absltest.main()
