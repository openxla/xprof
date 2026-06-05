from __future__ import annotations

import pathlib

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import
from xprof import profile_data


def _create_test_xspace() -> xplane_pb2.XSpace:
  return xplane_pb2.XSpace(
      planes=[
          xplane_pb2.XPlane(
              name="host_marker",
              stats=[
                  xplane_pb2.XStat(metadata_id=3, int64_value=42),
              ],
              lines=[
                  xplane_pb2.XLine(
                      name="line1",
                      timestamp_ns=1000,
                      events=[
                          xplane_pb2.XEvent(
                              metadata_id=1,
                              offset_ps=500_000,
                              duration_ps=200_000,
                              stats=[
                                  xplane_pb2.XStat(
                                      metadata_id=2, str_value="stat_val"
                                  ),
                              ],
                          ),
                      ],
                  ),
              ],
              event_metadata={1: xplane_pb2.XEventMetadata(name="event_name")},
              stat_metadata={
                  2: xplane_pb2.XStatMetadata(name="stat_name"),
                  3: xplane_pb2.XStatMetadata(name="plane_stat_name"),
              },
          ),
      ]
  )


class ProfileDataTest(parameterized.TestCase):

  def test_parse_empty_xspace(self):
    xspace = xplane_pb2.XSpace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      self.assertEmpty(pd.planes)

  def test_parse_plane_data(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      planes = pd.planes
      self.assertLen(planes, 1)
      plane = planes[0]
      self.assertEqual(plane.name, "host_marker")
      self.assertCountEqual(plane.stats, [("plane_stat_name", "42")])

  def test_parse_line_data(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      plane = pd.planes[0]
      lines = plane.lines
      self.assertLen(lines, 1)
      line = lines[0]
      self.assertEqual(line.name, "line1")

  def test_parse_event_count(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      events = pd.planes[0].lines[0].events
      self.assertLen(events, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="name",
          attr_name="name",
          expected_val="event_name",
          assert_method_name="assertEqual",
      ),
      dict(
          testcase_name="start_ns",
          attr_name="start_ns",
          expected_val=1500.0,
          assert_method_name="assertAlmostEqual",
      ),
      dict(
          testcase_name="duration_ns",
          attr_name="duration_ns",
          expected_val=200.0,
          assert_method_name="assertAlmostEqual",
      ),
      dict(
          testcase_name="stats",
          attr_name="stats",
          expected_val=(("stat_name", "stat_val"),),
          assert_method_name="assertCountEqual",
      ),
  )
  def test_parse_event_properties(
      self, attr_name, expected_val, assert_method_name
  ):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      event = pd.planes[0].lines[0].events[0]
      actual_val = getattr(event, attr_name)
      assert_method = getattr(self, assert_method_name)
      assert_method(actual_val, expected_val)

  def test_parse_from_file_with_path_object(self):
    xspace = xplane_pb2.XSpace(planes=[xplane_pb2.XPlane(name="test_plane")])

    temp_file = self.create_tempfile()
    temp_file.write_bytes(xspace.SerializeToString())

    # Pass as pathlib.Path object to verify PathLike support.
    path_obj = pathlib.Path(temp_file.full_path)
    with profile_data.ProfileData.from_file(path_obj) as pd:
      self.assertLen(pd.planes, 1)
      self.assertEqual(pd.planes[0].name, "test_plane")

  def test_data_context_manager(self):
    xspace = xplane_pb2.XSpace(planes=[xplane_pb2.XPlane(name="test_plane")])

    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      self.assertFalse(pd.closed)

    self.assertTrue(pd.closed)

  def test_plane_context_manager(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      plane = pd.planes[0]
      with plane:
        self.assertFalse(plane.closed)
      self.assertTrue(plane.closed)

  def test_line_context_manager(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      line = pd.planes[0].lines[0]
      with line:
        self.assertFalse(line.closed)
      self.assertTrue(line.closed)

  def test_event_context_manager(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      event = pd.planes[0].lines[0].events[0]
      with event:
        self.assertFalse(event.closed)
      self.assertTrue(event.closed)

  def test_plane_closes_child_lines(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      plane = pd.planes[0]
      line = plane.lines[0]
      with plane:
        self.assertFalse(line.closed)
      self.assertTrue(line.closed)

  def test_line_closes_child_events(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      line = pd.planes[0].lines[0]
      event = line.events[0]
      with line:
        self.assertFalse(event.closed)
      self.assertTrue(event.closed)

  def test_closed_data_errors(self):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      pass

    self.assertTrue(pd.closed)

    # Calling properties/methods on closed objects must raise ValueError.
    with self.assertRaisesRegex(
        ValueError, "I/O operation on closed ProfileData"
    ):
      _ = pd.planes

  @parameterized.named_parameters(
      dict(
          testcase_name="plane_name",
          resource_key="plane",
          attr_name="name",
          class_name="ProfilePlane",
      ),
      dict(
          testcase_name="plane_lines",
          resource_key="plane",
          attr_name="lines",
          class_name="ProfilePlane",
      ),
      dict(
          testcase_name="plane_stats",
          resource_key="plane",
          attr_name="stats",
          class_name="ProfilePlane",
      ),
      dict(
          testcase_name="line_name",
          resource_key="line",
          attr_name="name",
          class_name="ProfileLine",
      ),
      dict(
          testcase_name="line_events",
          resource_key="line",
          attr_name="events",
          class_name="ProfileLine",
      ),
      dict(
          testcase_name="event_name",
          resource_key="event",
          attr_name="name",
          class_name="ProfileEvent",
      ),
      dict(
          testcase_name="event_start_ns",
          resource_key="event",
          attr_name="start_ns",
          class_name="ProfileEvent",
      ),
      dict(
          testcase_name="event_duration_ns",
          resource_key="event",
          attr_name="duration_ns",
          class_name="ProfileEvent",
      ),
      dict(
          testcase_name="event_stats",
          resource_key="event",
          attr_name="stats",
          class_name="ProfileEvent",
      ),
  )
  def test_closed_attributes_raise_value_error(
      self, resource_key, attr_name, class_name
  ):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      plane = pd.planes[0]
      line = plane.lines[0]
      event = line.events[0]

    resource_map = {
        "plane": plane,
        "line": line,
        "event": event,
    }
    resource = resource_map[resource_key]
    self.assertTrue(resource.closed)

    with self.assertRaisesRegex(
        ValueError, f"I/O operation on closed {class_name}"
    ):
      _ = getattr(resource, attr_name)

  @parameterized.named_parameters(
      dict(
          testcase_name="data",
          resource_key="data",
          expected_repr="ProfileData(closed=True)",
      ),
      dict(
          testcase_name="plane",
          resource_key="plane",
          expected_repr="ProfilePlane(closed=True)",
      ),
      dict(
          testcase_name="line",
          resource_key="line",
          expected_repr="ProfileLine(closed=True)",
      ),
      dict(
          testcase_name="event",
          resource_key="event",
          expected_repr="ProfileEvent(closed=True)",
      ),
  )
  def test_closed_repr(self, resource_key, expected_repr):
    xspace = _create_test_xspace()
    with profile_data.ProfileData.from_serialized_xspace(
        xspace.SerializeToString()
    ) as pd:
      plane = pd.planes[0]
      line = plane.lines[0]
      event = line.events[0]

    resource_map = {
        "data": pd,
        "plane": plane,
        "line": line,
        "event": event,
    }
    resource = resource_map[resource_key]
    self.assertEqual(repr(resource), expected_repr)


if __name__ == "__main__":
  absltest.main()
