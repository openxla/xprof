"""Tests that setup.py parses requirements.in and configures packaging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from importlib import metadata as importlib_metadata
import io
import os
import sys
import tempfile
import types
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from packaging import metadata as packaging_metadata
from packaging import version as packaging_version
import setuptools
from setuptools import warnings as setuptools_warnings
from setuptools.command import bdist_wheel as setuptools_bdist_wheel

from google3.third_party.xprof.plugin import setup
from google3.third_party.xprof.tbp_package import setup as tbp_setup

# Constructed dynamically so automated text rewrites under plugin/ do not alter
# the expected compatibility package name.
_TBP_PROJECT_NAME = '_'.join(('tensorboard', 'plugin', 'profile'))
_SETUP_MODULES = types.MappingProxyType({
    'xprof': setup,
    'tbp': tbp_setup,
})

# Fixed subset of setuptools hooks exercised when running Distribution in test
# environments where setuptools dist-info metadata is absent (not the full set
# of installed setuptools entry points).
_SETUPTOOLS_ENTRY_POINT_SPECS = (
    (
        'distutils.setup_keywords',
        'install_requires',
        'setuptools.dist:check_requirements',
    ),
    (
        'distutils.setup_keywords',
        'extras_require',
        'setuptools.dist:check_extras',
    ),
    (
        'distutils.setup_keywords',
        'entry_points',
        'setuptools.dist:check_entry_points',
    ),
    (
        'distutils.setup_keywords',
        'package_data',
        'setuptools.dist:check_package_data',
    ),
    (
        'distutils.setup_keywords',
        'packages',
        'setuptools.dist:check_packages',
    ),
    (
        'distutils.setup_keywords',
        'python_requires',
        'setuptools.dist:check_specifier',
    ),
    (
        'setuptools.finalize_distribution_options',
        'keywords',
        'setuptools.dist:Distribution._finalize_setup_keywords',
    ),
    (
        'setuptools.finalize_distribution_options',
        'parent_finalize',
        'setuptools.dist:_Distribution.finalize_options',
    ),
)


def _build_synthetic_setuptools_entry_points() -> (
    importlib_metadata.EntryPoints
):
  """Builds synthetic setuptools EntryPoints from the fixed spec table."""
  return importlib_metadata.EntryPoints(
      importlib_metadata.EntryPoint(name=name, value=value, group=group)
      for group, name, value in _SETUPTOOLS_ENTRY_POINT_SPECS
  )


class _FakeWheelBdistWheel:
  """Minimal stand-in for legacy wheel.bdist_wheel.bdist_wheel."""

  def __init__(self):
    self.root_is_pure = True

  def finalize_options(self):
    self.root_is_pure = True

  def get_tag(self):
    return ('cp310', 'cp310', 'linux_x86_64')


class SetupTest(parameterized.TestCase):

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
        'fsspec[gcs] >= 2024.10.0',
        'cheroot >= 10.0.1',
        'etils[epath] >= 1.0.0',
        'werkzeug >= 3.1.9',
        'protobuf >= 3.19.6',
        'six >= 1.10.0',
        'google-cloud-storage >= 3.12.0',
        'urllib3 >= 2.8.0',
        'fire >= 0.4.0',
        'ml_dtypes >= 0.2.0',
        'numpy >= 1.24.0',
    ]
    for dep in expected_deps:
      self.assertIn(
          dep,
          setup.REQUIRED_PACKAGES,
          f'Expected dependency {dep} not found in REQUIRED_PACKAGES:'
          f' {setup.REQUIRED_PACKAGES}',
      )

  def test_console_scripts_entry_points(self):
    self.assertIn('console_scripts', setup.ENTRY_POINTS)
    self.assertIn(
        'xprof = xprof.cli.xprof_cli:main',
        setup.ENTRY_POINTS['console_scripts'],
    )
    self.assertIn(
        'xparity = xprof.xparity.xparity_cli:main',
        setup.ENTRY_POINTS['console_scripts'],
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

  def test_package_data_contains_skills(self):
    self.assertIn('xprof', setup.PACKAGE_DATA)
    self.assertIn('skills/**', setup.PACKAGE_DATA['xprof'])

  def test_package_data_contains_hlo_descriptor_set(self):
    self.assertIn(
        'cli/internal/oss/hlo_descriptor_set.pb', setup.PACKAGE_DATA['xprof']
    )

  def test_skills_markdown_files_present_in_source_tree(self):
    skills_dir = os.path.join(
        os.path.dirname(__file__), '..', 'skills', 'xprof'
    )
    skill_md = os.path.join(skills_dir, 'SKILL.md')
    roofline_md = os.path.join(
        skills_dir, 'references', 'get_roofline_model.md'
    )
    collect_md = os.path.join(skills_dir, 'references', 'collect_profile.md')
    self.assertTrue(os.path.isfile(skill_md), f'Missing {skill_md}')
    self.assertTrue(os.path.isfile(roofline_md), f'Missing {roofline_md}')
    self.assertTrue(os.path.isfile(collect_md), f'Missing {collect_md}')
    with open(roofline_md, 'r', encoding='utf-8') as f:
      content = f.read()
    self.assertNotIn('/google/bin/releases', content)
    self.assertNotIn('.par', content)
    self.assertIn('"program":', content)
    self.assertIn('"device_info":', content)
    self.assertIn('"top_operations":', content)

    with open(collect_md, 'r', encoding='utf-8') as f:
      collect_content = f.read()
    self.assertNotIn('/google/bin/releases', collect_content)
    self.assertNotIn('.par', collect_content)
    self.assertNotIn('.trace.json.gz', collect_content)
    self.assertNotIn('torch.profiler', collect_content)
    self.assertIn('jax.profiler', collect_content)
    self.assertIn('torch_xla', collect_content)
    self.assertIn('xp.start_trace', collect_content)
    self.assertIn('tensorflow', collect_content)
    self.assertIn('curl -X POST', collect_content)
    self.assertNotIn('curl -X GET', collect_content)

    xparity_skills_dir = os.path.join(
        os.path.dirname(__file__), '..', 'skills', 'xparity'
    )
    xparity_skill_md = os.path.join(xparity_skills_dir, 'SKILL.md')
    xparity_num_md = os.path.join(
        xparity_skills_dir, 'references', 'numerical_correctness.md'
    )
    self.assertTrue(
        os.path.isfile(xparity_skill_md), f'Missing {xparity_skill_md}'
    )
    self.assertTrue(os.path.isfile(xparity_num_md), f'Missing {xparity_num_md}')
    for md_path in (xparity_skill_md, xparity_num_md):
      with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
      self.assertNotIn('/google/bin/releases', md_content)
      self.assertNotIn('.par', md_content)
      self.assertNotIn('arca9-local', md_content)

  def test_oss_tools_package_files_present_in_source_tree(self):
    oss_tools_dir = os.path.join(
        os.path.dirname(__file__),
        'xprof',
        'cli',
        'tools',
        'oss',
    )
    init_py = os.path.join(oss_tools_dir, '__init__.py')
    graph_viewer_py = os.path.join(oss_tools_dir, 'get_graph_viewer_tool.py')
    kernel_utilization_py = os.path.join(
        oss_tools_dir, 'get_kernel_utilization_tool.py'
    )
    upload_trace_py = os.path.join(oss_tools_dir, 'upload_trace_tool.py')
    perf_counters_py = os.path.join(
        os.path.dirname(oss_tools_dir), 'get_perf_counters_tool.py'
    )
    self.assertTrue(os.path.isfile(init_py), f'Missing {init_py}')
    self.assertTrue(
        os.path.isfile(graph_viewer_py), f'Missing {graph_viewer_py}'
    )
    self.assertTrue(
        os.path.isfile(kernel_utilization_py),
        f'Missing {kernel_utilization_py}',
    )
    self.assertTrue(
        os.path.isfile(upload_trace_py), f'Missing {upload_trace_py}'
    )
    self.assertTrue(
        os.path.isfile(perf_counters_py), f'Missing {perf_counters_py}'
    )
    for detect_tool in (
        'detect_layout_mismatch_copies_tool.py',
        'detect_unfused_reshapes_tool.py',
        'detect_unfused_updates_tool.py',
        'detect_unnecessary_convert_dynamic_scale_tool.py',
        'detect_unnecessary_convert_reduce_tool.py',
    ):
      detect_tool_py = os.path.join(os.path.dirname(oss_tools_dir), detect_tool)
      self.assertTrue(
          os.path.isfile(detect_tool_py), f'Missing {detect_tool_py}'
      )

  def _ensure_setuptools_entry_points(self):
    if importlib_metadata.entry_points(group='distutils.setup_keywords'):
      return
    entry_points = _build_synthetic_setuptools_entry_points()
    for entry_point in entry_points:
      self.assertTrue(
          callable(entry_point.load()),
          f'Synthetic entry point {entry_point.value!r} is not callable',
      )
    self.enter_context(
        mock.patch.object(
            importlib_metadata,
            'entry_points',
            autospec=True,
            side_effect=entry_points.select,
        )
    )

  @contextlib.contextmanager
  def _assert_no_packaging_warnings(self):
    with warnings.catch_warnings(record=True) as recorded_warnings:
      warnings.simplefilter('always')
      yield
    relevant_warnings = [
        w
        for w in recorded_warnings
        if issubclass(
            w.category,
            (
                setuptools_warnings.SetuptoolsDeprecationWarning,
                DeprecationWarning,
                FutureWarning,
                UserWarning,
            ),
        )
    ]
    self.assertEmpty(
        relevant_warnings,
        f'Unexpected warnings: {[str(w.message) for w in relevant_warnings]}',
    )

  def _build_validated_metadata(self, setup_kwargs):
    self._ensure_setuptools_entry_points()
    buf = io.StringIO()
    with self._assert_no_packaging_warnings():
      dist = setuptools.Distribution(setup_kwargs)
      dist.parse_config_files(filenames=[])
      dist.metadata.write_pkg_file(buf)

    return packaging_metadata.Metadata.from_email(buf.getvalue(), validate=True)

  def test_custom_bdist_wheel_registered_in_cmdclass(self):
    bdist_wheel_cls = setup.CustomBdistWheel
    if bdist_wheel_cls is None:
      self.fail('CustomBdistWheel should not be None')
    self.assertTrue(
        issubclass(bdist_wheel_cls, setuptools_bdist_wheel.bdist_wheel)
    )
    self.assertIs(setup.cmdclass['bdist_wheel'], bdist_wheel_cls)

  def test_custom_bdist_wheel_prefers_setuptools_over_wheel(self):
    fake_wheel_mod = types.ModuleType('wheel.bdist_wheel')
    setattr(fake_wheel_mod, 'bdist_wheel', _FakeWheelBdistWheel)
    with mock.patch.dict(sys.modules, {'wheel.bdist_wheel': fake_wheel_mod}):
      preferred_cls = setup._make_custom_bdist_wheel()

    if preferred_cls is None:
      self.fail('Resolved CustomBdistWheel should not be None')
    self.assertTrue(
        issubclass(preferred_cls, setuptools_bdist_wheel.bdist_wheel)
    )
    self.assertFalse(issubclass(preferred_cls, _FakeWheelBdistWheel))

  def test_custom_bdist_wheel_resolves_without_wheel_or_warnings(self):
    with mock.patch.dict(
        sys.modules, {'wheel': None, 'wheel.bdist_wheel': None}
    ):
      with self._assert_no_packaging_warnings():
        isolated_cls = setup._make_custom_bdist_wheel()

    if isolated_cls is None:
      self.fail('Isolated CustomBdistWheel should not be None')
    self.assertTrue(
        issubclass(isolated_cls, setuptools_bdist_wheel.bdist_wheel)
    )

  def test_custom_bdist_wheel_falls_back_to_wheel_bdist_wheel(self):
    fake_wheel_mod = types.ModuleType('wheel.bdist_wheel')
    setattr(fake_wheel_mod, 'bdist_wheel', _FakeWheelBdistWheel)
    with mock.patch.dict(
        sys.modules,
        {
            'setuptools.command.bdist_wheel': None,
            'wheel.bdist_wheel': fake_wheel_mod,
        },
    ):
      fallback_cls = setup._make_custom_bdist_wheel()

    if fallback_cls is None:
      self.fail('Fallback CustomBdistWheel should not be None')
    self.assertTrue(issubclass(fallback_cls, _FakeWheelBdistWheel))
    cmd = fallback_cls()
    cmd.finalize_options()
    self.assertIs(cmd.root_is_pure, False)
    self.assertEqual(cmd.get_tag(), ('py3', 'none', 'linux_x86_64'))

  def test_custom_bdist_wheel_returns_none_when_unavailable(self):
    with mock.patch.dict(
        sys.modules,
        {
            'setuptools.command.bdist_wheel': None,
            'wheel.bdist_wheel': None,
        },
    ):
      self.assertIsNone(setup._make_custom_bdist_wheel())

  def test_custom_bdist_wheel_marks_non_pure_and_overrides_tag(self):
    bdist_wheel_cls = setup.CustomBdistWheel
    if bdist_wheel_cls is None:
      self.fail('CustomBdistWheel should not be None')

    dist = setuptools.Distribution({'name': 'xprof', 'version': '2.24.2'})
    cmd = bdist_wheel_cls(dist)
    cmd.plat_name = 'linux_x86_64'
    cmd.dist_info_dir = self.create_tempdir().full_path
    cmd.ensure_finalized()

    self.assertIs(cmd.root_is_pure, False)
    self.assertEqual(cmd.get_tag(), ('py3', 'none', 'linux_x86_64'))

  def test_xprof_setup_kwargs_options(self):
    kwargs = setup.get_setup_kwargs()

    self.assertNotIn('tests_require', kwargs)
    self.assertEqual(kwargs['extras_require']['test'], setup.TEST_PACKAGES)
    self.assertIs(kwargs['cmdclass']['bdist_wheel'], setup.CustomBdistWheel)

  def test_tbp_setup_kwargs_options(self):
    kwargs = tbp_setup.get_setup_kwargs()

    self.assertEqual(tbp_setup.PROJECT_NAME, _TBP_PROJECT_NAME)
    self.assertEqual(tbp_setup.VERSION, '0.0.0')
    self.assertEqual(kwargs['install_requires'], ['xprof==0.0.0'])

  def _patch_packaging_env(
      self,
      *,
      setuptools_version: str,
      dist_has_license_attr: bool | None = None,
      missing_packaging_module: str | None = None,
  ):
    self.enter_context(
        mock.patch.object(setuptools, '__version__', setuptools_version)
    )
    if dist_has_license_attr is not None:
      if dist_has_license_attr:
        metadata = types.SimpleNamespace(license_expression=None)
      else:
        metadata = types.SimpleNamespace()
      fake_dist = types.SimpleNamespace(metadata=metadata)
      self.enter_context(
          mock.patch.object(
              setuptools, 'Distribution', autospec=True, return_value=fake_dist
          )
      )
    else:
      self.enter_context(
          mock.patch.object(
              setuptools,
              'Distribution',
              autospec=True,
              side_effect=AssertionError(
                  'Distribution() should not be instantiated for PEP 440'
                  ' version checks or when packaging modules are unavailable'
              ),
          )
      )
    if missing_packaging_module is not None:
      self.enter_context(
          mock.patch.dict(sys.modules, {missing_packaging_module: None})
      )

  @parameterized.product(
      (
          dict(
              setuptools_version='77.0.3',
              dist_has_license_attr=None,
          ),
          dict(
              setuptools_version='80.9.0.post1',
              dist_has_license_attr=None,
          ),
          dict(
              setuptools_version='1!0.1.0',
              dist_has_license_attr=None,
          ),
          dict(
              setuptools_version='0.dev0+unknown',
              dist_has_license_attr=True,
          ),
          dict(
              setuptools_version='not-a-version',
              dist_has_license_attr=True,
          ),
      ),
      package_name=('xprof', 'tbp'),
  )
  def test_setup_kwargs_pep639_license(
      self,
      package_name,
      setuptools_version,
      dist_has_license_attr,
  ):
    self._patch_packaging_env(
        setuptools_version=setuptools_version,
        dist_has_license_attr=dist_has_license_attr,
    )
    kwargs = _SETUP_MODULES[package_name].get_setup_kwargs()

    self.assertNotIn('license', kwargs)
    self.assertEqual(kwargs['license_expression'], 'Apache-2.0')
    self.assertEmpty([
        classifier
        for classifier in kwargs['classifiers']
        if classifier.startswith('License ::')
    ])

  @parameterized.product(
      (
          dict(
              setuptools_version='77.0.2',
              dist_has_license_attr=None,
              missing_packaging_module=None,
          ),
          dict(
              setuptools_version='77.0.3.dev0',
              dist_has_license_attr=None,
              missing_packaging_module=None,
          ),
          dict(
              setuptools_version='77.0.3rc1',
              dist_has_license_attr=None,
              missing_packaging_module=None,
          ),
          dict(
              setuptools_version='69.5.1',
              dist_has_license_attr=None,
              missing_packaging_module=None,
          ),
          dict(
              setuptools_version='0.dev0+unknown',
              dist_has_license_attr=False,
              missing_packaging_module=None,
          ),
          dict(
              setuptools_version='not-a-version',
              dist_has_license_attr=False,
              missing_packaging_module=None,
          ),
          dict(
              setuptools_version='77.0.3',
              dist_has_license_attr=None,
              missing_packaging_module='packaging.licenses',
          ),
          dict(
              setuptools_version='77.0.3',
              dist_has_license_attr=None,
              missing_packaging_module='packaging.version',
          ),
          dict(
              setuptools_version='0.dev0+unknown',
              dist_has_license_attr=None,
              missing_packaging_module='packaging.licenses',
          ),
      ),
      package_name=('xprof', 'tbp'),
  )
  def test_setup_kwargs_legacy_license_fallback(
      self,
      package_name,
      setuptools_version,
      dist_has_license_attr,
      missing_packaging_module,
  ):
    self._patch_packaging_env(
        setuptools_version=setuptools_version,
        dist_has_license_attr=dist_has_license_attr,
        missing_packaging_module=missing_packaging_module,
    )
    kwargs = _SETUP_MODULES[package_name].get_setup_kwargs()

    self.assertNotIn('license_expression', kwargs)
    self.assertEqual(kwargs['license'], 'Apache 2.0')
    self.assertEqual(
        [
            classifier
            for classifier in kwargs['classifiers']
            if classifier.startswith('License ::')
        ],
        ['License :: OSI Approved :: Apache Software License'],
    )

  def test_synthetic_setuptools_entry_points_resolve_to_callables(self):
    entry_points = _build_synthetic_setuptools_entry_points()
    self.assertNotEmpty(entry_points)
    for entry_point in entry_points:
      with self.subTest(group=entry_point.group, name=entry_point.name):
        self.assertTrue(
            callable(entry_point.load()),
            f'Synthetic entry point {entry_point.value!r} is not callable',
        )

  def test_distribution_metadata_valid_pep639_and_zero_warnings(self):
    parsed = self._build_validated_metadata(setup.get_setup_kwargs())

    self.assertGreaterEqual(
        packaging_version.Version(parsed.metadata_version),
        packaging_version.Version('2.4'),
    )
    self.assertEqual(parsed.license_expression, 'Apache-2.0')
    self.assertIsNone(parsed.license)
    self.assertEmpty([
        classifier
        for classifier in parsed.classifiers or ()
        if classifier.startswith('License ::')
    ])
    self.assertEqual(parsed.name, 'xprof')
    self.assertEqual(parsed.provides_extra, ['test'])
    self.assertNotEmpty(parsed.requires_dist)

  def test_tbp_distribution_metadata_valid_pep639_and_zero_warnings(self):
    parsed = self._build_validated_metadata(tbp_setup.get_setup_kwargs())

    self.assertGreaterEqual(
        packaging_version.Version(parsed.metadata_version),
        packaging_version.Version('2.4'),
    )
    self.assertEqual(parsed.license_expression, 'Apache-2.0')
    self.assertIsNone(parsed.license)
    self.assertEmpty([
        classifier
        for classifier in parsed.classifiers or ()
        if classifier.startswith('License ::')
    ])
    self.assertEqual(parsed.name, _TBP_PROJECT_NAME)
    self.assertEqual(
        [str(req) for req in parsed.requires_dist or ()],
        ['xprof==0.0.0'],
    )
    self.assertNotEmpty(parsed.description)

  def test_tbp_get_long_description_reads_sibling_readme(self):
    pkg_dir = self.create_tempdir()
    pkg_dir.create_file('README.md', content='Sibling TBP README')
    with mock.patch.object(
        tbp_setup, '__file__', os.path.join(pkg_dir.full_path, 'setup.py')
    ):
      self.assertEqual(tbp_setup.get_long_description(), 'Sibling TBP README')

  def test_tbp_get_long_description_reads_parent_readme(self):
    root_dir = self.create_tempdir()
    root_dir.create_file('README.md', content='Parent repo README')
    pkg_dir = root_dir.mkdir('tbp_package')
    with mock.patch.object(
        tbp_setup, '__file__', os.path.join(pkg_dir.full_path, 'setup.py')
    ):
      self.assertEqual(tbp_setup.get_long_description(), 'Parent repo README')

  def test_tbp_get_long_description_falls_back_when_readme_missing(self):
    root_dir = self.create_tempdir()
    pkg_dir = root_dir.mkdir('tbp_package')
    with mock.patch.object(
        tbp_setup, '__file__', os.path.join(pkg_dir.full_path, 'setup.py')
    ):
      self.assertEqual(
          tbp_setup.get_long_description(), 'XProf Profiler Plugin'
      )


if __name__ == '__main__':
  absltest.main()
