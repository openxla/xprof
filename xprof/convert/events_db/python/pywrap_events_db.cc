/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/arrow/util/type_fwd.h"
#include "third_party/nanobind/include/nanobind/make_iterator.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/operators.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"
#include "third_party/nanobind/include/nanobind/stl/string.h"
#include "third_party/nanobind/include/nanobind/stl/string_view.h"
#include "third_party/nanobind/include/nanobind/stl/variant.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/xspace_parser.h"

namespace nb = nanobind;
using nanobind::literals::operator""_a;

namespace {

constexpr std::string_view kModuleName =
    "xprof.convert.events_db.python.events_db";

nb::object import_python_class(std::string_view name) {
  return nb::module_::import_(kModuleName.data()).attr(name.data());
}

}  // namespace

namespace xprof::events_db {
namespace {

template <typename>
constexpr bool always_false_v = false;

template <typename T>
std::vector<T> PyToVector(nb::handle h) {
  if (nb::isinstance<nb::str>(h))
    throw nb::type_error("str is not a supported sequence; use list or tuple");
  if (nb::isinstance<nb::bytes>(h))
    throw nb::type_error("bytes is not a supported FieldValue; use str");
  if (nb::isinstance<std::vector<T>>(h)) return nb::cast<std::vector<T>>(h);
  if (nb::isinstance<nb::sequence>(h)) {
    nb::sequence seq = nb::cast<nb::sequence>(h);
    const size_t count = nb::len(seq);
    std::vector<T> v;
    v.reserve(count);
    for (nb::handle item : seq) v.push_back(nb::cast<T>(item));
    return v;
  }
  throw nb::type_error("Expected a sequence");
}

// Binds `std::vector<T>` as a read-only `Sequence` view to avoid copies on
// read.
template <typename T>
void BindSequence(nb::module_& m, const char* name, nb::handle abc_sequence) {
  nb::class_<std::vector<T>> cls =
      nb::class_<std::vector<T>>(m, name)
          .def("__len__", [](const std::vector<T>& v) { return v.size(); })
          .def("__getitem__",
               [](const std::vector<T>& v, Py_ssize_t i) -> const T& {
                 Py_ssize_t size = static_cast<Py_ssize_t>(v.size());
                 if (i < 0) i += size;
                 if (i < 0 || i >= size) throw nb::index_error();
                 return v[i];
               })
          .def("__getitem__",
               [](const std::vector<T>& v, nb::slice slice) -> std::vector<T> {
                 auto [start, stop, step, length] = slice.compute(v.size());
                 // `[[maybe_unused]]` must be before `auto`
                 (void)stop;  // Unused.
                 std::vector<T> res;
                 res.reserve(length);
                 for (size_t i = 0; i < length; ++i) {
                   res.push_back(v[start]);
                   start += step;
                 }
                 return res;
               })
          .def(
              "__iter__",
              [](const std::vector<T>& v) {
                return nb::make_iterator(nb::type<std::vector<T>>(), "Iterator",
                                         v.begin(), v.end());
              },
              // Keep the container sequence (1) alive in memory for as long as
              // the returned iterator (0) exists.
              nb::keep_alive<0, 1>())
          .def("__repr__",
               [](const std::vector<T>& v) {
                 std::ostringstream oss;
                 oss << "[";
                 for (size_t i = 0; i < v.size(); ++i) {
                   if (i > 0) oss << ", ";
                   if constexpr (std::is_same_v<T, std::string>)
                     oss << "\"" << v[i] << "\"";
                   else
                     oss << v[i];
                 }
                 oss << "]";
                 return oss.str();
               })
          .def("__eq__", [](const std::vector<T>& a, nb::handle b) {
            if (nb::isinstance<std::vector<T>>(b))
              return a == nb::cast<const std::vector<T>&>(b);
            if (!nb::isinstance<nb::sequence>(b)) return false;
            nb::sequence seq = nb::cast<nb::sequence>(b);
            if (a.size() != nb::len(seq)) return false;
            try {
              for (size_t i = 0; i < a.size(); ++i)
                if (!nb::cast(a[i]).equal(seq[i])) return false;
            } catch (const nb::python_error& e) {
              if (!e.matches(PyExc_Exception)) throw;
              return false;
            }
            return true;
          });

  abc_sequence.attr("register")(cls);
}

// Converts a FieldValue variant to its corresponding Python object.
nb::object FieldValueToPyObject(nb::handle parent_py, const FieldValue& val) {
  return std::visit(
      [&](const auto& arg) -> nb::object {
        using T = std::remove_cvref_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return nb::none();
        } else if constexpr (std::is_same_v<T, bool> ||
                             std::is_same_v<T, int32_t> ||
                             std::is_same_v<T, uint32_t> ||
                             std::is_same_v<T, int64_t> ||
                             std::is_same_v<T, uint64_t> ||
                             std::is_same_v<T, double> ||
                             std::is_same_v<T, std::string>) {
          return nb::cast(arg);
        } else if constexpr (std::is_same_v<T, std::vector<int32_t>> ||
                             std::is_same_v<T, std::vector<uint32_t>> ||
                             std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<uint64_t>> ||
                             std::is_same_v<T, std::vector<double>> ||
                             std::is_same_v<T, std::vector<std::string>>) {
          // Zero-copy transient reference tied to the parent Record's lifetime.
          return nb::cast(&arg, nb::rv_policy::reference_internal, parent_py);
        } else {
          static_assert(always_false_v<T>, "Unhandled FieldValue type");
        }
      },
      val);
}

// Converts a Python object to an appropriate FieldValue alternative.
FieldValue PyObjectToFieldValue(nb::handle obj) {
  if (obj.is_none()) return std::monostate{};
  if (nb::isinstance<nb::bool_>(obj)) return nb::cast<bool>(obj);

  if (nb::isinstance<nb::int_>(obj)) {
    if (int64_t val; nb::try_cast(obj, val)) return val;
    return nb::cast<uint64_t>(obj);
  }
  if (nb::isinstance<nb::float_>(obj)) return nb::cast<double>(obj);
  if (nb::isinstance<nb::str>(obj)) return nb::cast<std::string>(obj);

  // Reject bytes explicitly so it is not treated as a sequence of integers.
  if (nb::isinstance<nb::bytes>(obj))
    throw nb::type_error("bytes is not a supported FieldValue; use str");

  if (nb::isinstance<std::vector<int32_t>>(obj))
    return nb::cast<std::vector<int32_t>>(obj);
  if (nb::isinstance<std::vector<uint32_t>>(obj))
    return nb::cast<std::vector<uint32_t>>(obj);
  if (nb::isinstance<std::vector<int64_t>>(obj))
    return nb::cast<std::vector<int64_t>>(obj);
  if (nb::isinstance<std::vector<uint64_t>>(obj))
    return nb::cast<std::vector<uint64_t>>(obj);
  if (nb::isinstance<std::vector<double>>(obj))
    return nb::cast<std::vector<double>>(obj);
  if (nb::isinstance<std::vector<std::string>>(obj))
    return nb::cast<std::vector<std::string>>(obj);

  if (nb::isinstance<nb::sequence>(obj)) {
    nb::sequence seq = nb::cast<nb::sequence>(obj);
    const size_t count = nb::len(seq);
    if (count == 0) return std::vector<int64_t>{};
    const nb::handle first = seq[0];
    if (nb::isinstance<nb::int_>(first)) return PyToVector<int64_t>(seq);
    if (nb::isinstance<nb::float_>(first)) return PyToVector<double>(seq);
    if (nb::isinstance<nb::str>(first)) return PyToVector<std::string>(seq);
  }

  try {
    return nb::cast<FieldValue>(obj);
  } catch (const nb::cast_error&) {
    throw nb::type_error("Unsupported value type for FieldValue");
  }
}

class PyRecordConsumer {
 public:
  explicit PyRecordConsumer(nb::handle target) {
    nb::gil_scoped_acquire gil;
    target_ = nb::borrow<nb::object>(target);

    if (nb::hasattr(target, "consume")) {
      nb::object consume_attr = target.attr("consume");
      if (PyCallable_Check(consume_attr.ptr()))
        consume_fn_ = std::move(consume_attr);
    }

    if (!consume_fn_.is_valid() && PyCallable_Check(target.ptr()))
      consume_fn_ = nb::borrow<nb::object>(target);

    if (!consume_fn_.is_valid()) {
      throw nb::type_error(
          "RecordConsumerRef target must be callable or provide a callable "
          "'consume' method");
    }

    if (nb::hasattr(target, "finalize")) {
      nb::object finalize_attr = target.attr("finalize");
      if (!PyCallable_Check(finalize_attr.ptr()))
        throw nb::type_error("'finalize' attribute must be callable");
      finalize_fn_ = std::move(finalize_attr);
      has_finalize_ = true;

      try {
        const nb::object inspect_mod = nb::module_::import_("inspect");
        const nb::object sig = inspect_mod.attr("signature")(finalize_fn_);
        const nb::object params = sig.attr("parameters");
        finalize_takes_arg_ = nb::len(params) > 0;
      } catch (const nb::python_error&) {
        finalize_takes_arg_ = true;
      }
    }
  }

  PyRecordConsumer(const PyRecordConsumer& other) {
    if (nb::is_alive()) {
      nb::gil_scoped_acquire gil;
      target_ = other.target_;
      consume_fn_ = other.consume_fn_;
      finalize_fn_ = other.finalize_fn_;
    }
    has_finalize_ = other.has_finalize_;
    finalize_takes_arg_ = other.finalize_takes_arg_;
  }

  PyRecordConsumer(PyRecordConsumer&& other) noexcept = default;
  PyRecordConsumer& operator=(const PyRecordConsumer&) = delete;
  PyRecordConsumer& operator=(PyRecordConsumer&&) = delete;

  ~PyRecordConsumer() {
    if (nb::is_alive()) {
      nb::gil_scoped_acquire gil;
      target_.reset();
      consume_fn_.reset();
      finalize_fn_.reset();
    } else {
      target_.release();
      consume_fn_.release();
      finalize_fn_.release();
    }
  }

  absl::StatusOr<StepControl> operator()(Record& record) const {
    return Consume(record);
  }

  absl::StatusOr<StepControl> Consume(Record& record) const noexcept {
    try {
      return InvokeConsume(record);
    } catch (const std::exception& e) {
      return absl::UnknownError(e.what());
    }
  }

  StepControl PyConsume(Record& record) const { return InvokeConsume(record); }

  absl::Status Finalize(
      const absl::StatusOr<ParseStatus>& result) const noexcept {
    try {
      InvokeFinalize(result);
      return absl::OkStatus();
    } catch (const std::exception& e) {
      return absl::UnknownError(e.what());
    }
  }

  void PyFinalize(nb::handle result = nb::none()) const {
    nb::gil_scoped_acquire gil;
    if (!result.is_none() && !nb::isinstance<ParseStatus>(result) &&
        !PyExceptionInstance_Check(result.ptr())) {
      throw nb::type_error(
          "finalize argument must be a ParseStatus, an Exception, or None");
    }
    if (!has_finalize_) return;

    if (!finalize_takes_arg_) {
      if (!PyExceptionInstance_Check(result.ptr())) finalize_fn_();
      return;
    }

    if (result.is_none())
      finalize_fn_(nb::cast(ParseStatus::kComplete));
    else
      finalize_fn_(result);
  }

  RecordConsumerRef AsRef() const noexcept { return RecordConsumerRef(*this); }

  nb::object target() const {
    nb::gil_scoped_acquire gil;
    return target_;
  }

 private:
  StepControl InvokeConsume(Record& record) const {
    nb::gil_scoped_acquire gil;
    nb::object py_record = nb::cast(&record, nb::rv_policy::reference);
    nb::object py_result = consume_fn_(py_record);

    if (py_result.is_none()) return StepControl::kContinue;
    if (nb::isinstance<nb::bool_>(py_result))
      return nb::cast<bool>(py_result) ? StepControl::kContinue
                                       : StepControl::kStop;
    if (nb::isinstance<StepControl>(py_result))
      return nb::cast<StepControl>(py_result);
    throw nb::type_error(
        "Record consumer must return StepControl, bool, or None");
  }

  void InvokeFinalize(const absl::StatusOr<ParseStatus>& result) const {
    if (!has_finalize_) return;

    nb::gil_scoped_acquire gil;
    if (finalize_takes_arg_) {
      if (result.ok()) {
        finalize_fn_(nb::cast(*result));
      } else {
        const absl::Status& status = result.status();
        const nb::handle exc = nb::handle(PyExc_RuntimeError);
        if (status.message().empty())
          finalize_fn_(exc(status.ToString()));
        else
          finalize_fn_(exc(status.message()));
      }
    } else {
      if (result.ok()) finalize_fn_();
    }
  }

  nb::object target_;
  nb::object consume_fn_;
  nb::object finalize_fn_;
  bool has_finalize_ = false;
  bool finalize_takes_arg_ = false;
};

static_assert(std::is_constructible_v<RecordConsumerRef, PyRecordConsumer&>);
static_assert(
    std::is_constructible_v<RecordConsumerRef, const PyRecordConsumer&>);

}  // namespace
}  // namespace xprof::events_db

namespace nanobind::detail {

template <>
struct type_caster<xprof::events_db::RecordConsumerRef> {
  static constexpr auto Name = const_name("RecordConsumerRef");
  template <typename T_>
  using Cast = movable_cast_t<T_>;
  template <typename T_>
  static constexpr bool can_cast() {
    return true;
  }

  std::optional<xprof::events_db::PyRecordConsumer> holder;
  std::optional<xprof::events_db::RecordConsumerRef> value;

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    if (src.is_none()) return false;

    if (isinstance<xprof::events_db::PyRecordConsumer>(src)) {
      xprof::events_db::PyRecordConsumer& c =
          cast<xprof::events_db::PyRecordConsumer&>(src);
      value.emplace(c.AsRef());
      return true;
    }

    if (!PyCallable_Check(src.ptr()) && !hasattr(src, "consume")) return false;

    try {
      holder.emplace(borrow<object>(src));
      value.emplace(holder->AsRef());
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

  static handle from_cpp(xprof::events_db::RecordConsumerRef src,
                         rv_policy policy,
                         cleanup_list* cleanup) noexcept = delete;

  explicit operator xprof::events_db::RecordConsumerRef*() { return &*value; }
  explicit operator xprof::events_db::RecordConsumerRef&() { return *value; }
  explicit operator xprof::events_db::RecordConsumerRef&&() {
    return (xprof::events_db::RecordConsumerRef&&)*value;
  }
};

template <>
struct type_caster<arrow::Compression::type> {
  static constexpr char kName[] = "ArrowCompressionType";
  NB_TYPE_CASTER(arrow::Compression::type, const_name(kName))

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    std::string_view name;
    try {
      if (!isinstance(src, import_python_class(kName))) return false;
      name = cast<std::string_view>(src.attr("name"));
    } catch (const std::exception&) {
      // Defensive fallback: module import and attribute access on a valid enum
      // instance are expected to succeed. This prevents unexpected runtime or
      // C-API exceptions from escaping `noexcept` into `std::terminate()`.
      return false;
    }
    for (const auto& [type, type_name] : kCompressionCodecs) {
      if (name == type_name) {
        value = type;
        return true;
      }
    }
    // Signal conversion failure to nanobind so it raises a standard Python
    // TypeError.
    return false;
  }

  static handle from_cpp(arrow::Compression::type src, rv_policy /*policy*/,
                         cleanup_list* /*cleanup*/) noexcept {
    for (const auto& [type, type_name] : kCompressionCodecs) {
      if (type == src) {
        try {
          return object(import_python_class(kName).attr(type_name.data()))
              .release();
        } catch (const std::exception&) {
          // Defensive fallback: module import and attribute lookup are expected
          // to always succeed for known enum members. This catch prevents
          // unexpected Python/C-API exceptions from escaping `noexcept` into
          // `std::terminate()`.
          return handle();
        }
      }
    }
    // Return an empty handle so nanobind signals a conversion failure.
    return handle();
  }

 private:
  static constexpr std::pair<arrow::Compression::type, std::string_view>
      kCompressionCodecs[] = {
          {arrow::Compression::UNCOMPRESSED, "UNCOMPRESSED"},
          {arrow::Compression::SNAPPY, "SNAPPY"},
          {arrow::Compression::GZIP, "GZIP"},
          {arrow::Compression::BROTLI, "BROTLI"},
          {arrow::Compression::ZSTD, "ZSTD"},
          {arrow::Compression::LZ4, "LZ4"},
          {arrow::Compression::LZ4_FRAME, "LZ4_FRAME"},
          {arrow::Compression::LZO, "LZO"},
          {arrow::Compression::BZ2, "BZ2"},
          {arrow::Compression::LZ4_HADOOP, "LZ4_HADOOP"},
      };
};

}  // namespace nanobind::detail

namespace xprof::events_db {

NB_MODULE(pywrap_events_db, m) {
  m.doc() = "Native C++ bindings for XProf Events DB Schema and Record";

  nb::object abc_sequence =
      nb::module_::import_("collections.abc").attr("Sequence");

  // Bind zero-copy sequence views for repeated fields and register with
  // Sequence.
  BindSequence<std::string>(m, "StringSequence", abc_sequence);
  BindSequence<int32_t>(m, "Int32Sequence", abc_sequence);
  BindSequence<uint32_t>(m, "Uint32Sequence", abc_sequence);
  BindSequence<int64_t>(m, "Int64Sequence", abc_sequence);
  BindSequence<uint64_t>(m, "Uint64Sequence", abc_sequence);
  BindSequence<double>(m, "DoubleSequence", abc_sequence);

  // Bind FieldIndex opaque token
  nb::class_<FieldIndex>(m, "FieldIndex",
                         "Opaque token representing a field. Wraps an integer "
                         "for O(1) hashing and lookup.")
      .def(nb::init<>())
      .def("is_valid", &FieldIndex::is_valid,
           "Returns true if the `FieldIndex` represents a valid registered "
           "field.")
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def(nb::self < nb::self)
      .def(nb::self <= nb::self)
      .def(nb::self > nb::self)
      .def(nb::self >= nb::self)
      .def("__hash__",
           [](const FieldIndex& self) -> size_t {
             return absl::Hash<FieldIndex>{}(self);
           })
      .def("__repr__", [](const FieldIndex& self) {
        return self.is_valid() ? "FieldIndex(valid)" : "FieldIndex(invalid)";
      });

  // Bind Schema
  nb::class_<Schema>(
      m, "Schema",
      "Manager for the mapping between string names and indices. Thread-safe.")
      .def(nb::init<uint32_t>(),
           "max_field_count"_a = std::numeric_limits<uint32_t>::max(),
           "Constructs a Schema with an optional maximum number of fields.")
      .def(
          "register_field_name",
          [](Schema& self, std::string_view name) {
            return self.RegisterFieldName(name);
          },
          "name"_a,
          "Registers a name and returns its unique index. If already "
          "registered, returns the existing index. If `max_field_count` names "
          "have already been registered and the input name has not been "
          "registered before, returns an invalid index.")
      .def(
          "get_field_name",
          [](const Schema& self,
             FieldIndex field) -> std::optional<std::string_view> {
            return self.GetFieldName(field);
          },
          "field"_a, "Resolves an index back to its name.")
      .def(
          "lookup_field_index",
          [](const Schema& self, std::string_view name) {
            return self.LookupFieldIndex(name);
          },
          "name"_a, "Looks up an existing field without registering it.")
      .def("__len__", &Schema::size)
      .def_prop_ro("size", &Schema::size,
                   "Returns the number of names currently registered.")
      .def_prop_ro(
          "max_field_count", &Schema::max_field_count,
          "Returns the maximum number of names that can be registered.");

  // Bind Record
  nb::class_<Record>(
      m, "Record",
      "An extensible, row-oriented record mapping field indices to dynamically "
      "typed values. Not thread-safe. When received via a consumer callback, "
      "the `Record` is transient and valid only for the duration of the call.")
      .def(nb::init<>())
      .def(
          "has_field",
          [](const Record& self, FieldIndex field) {
            return self.HasField(field);
          },
          "field"_a, "Checks if the field has a set value in the record.")
      .def(
          "__contains__",
          [](const Record& self, FieldIndex field) {
            return self.HasField(field);
          },
          "field"_a)
      .def(
          "get",
          [](nb::handle self_py, FieldIndex field,
             std::optional<nb::object> default_value) -> nb::object {
            const Record& self = nb::cast<const Record&>(self_py);
            if (!field.is_valid() || !self.HasField(field))
              return default_value.value_or(nb::none());
            return FieldValueToPyObject(self_py, self[field]);
          },
          "field"_a, "default"_a = nb::none(),
          "Retrieves the value associated with the given field. Returns "
          "default if the field is unset or missing.")
      .def(
          "__getitem__",
          [](nb::handle self_py, FieldIndex field) -> nb::object {
            const Record& self = nb::cast<const Record&>(self_py);
            if (!field.is_valid() || !self.HasField(field))
              throw nb::key_error("Field not found in Record");
            return FieldValueToPyObject(self_py, self[field]);
          },
          "field"_a,
          "Retrieves the value associated with the given field. Raises "
          "`KeyError` if unset or missing.")
      .def(
          "set",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self[field] = PyObjectToFieldValue(value);
          },
          "field"_a, "value"_a,
          "Sets the value associated with the given field.")
      .def(
          "__setitem__",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self[field] = PyObjectToFieldValue(value);
          },
          "field"_a, "value"_a,
          "Sets the value associated with the given field.")
      .def(
          "set_bool",
          [](Record& self, FieldIndex field, bool value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetBool(field, value);
          },
          "field"_a, "value"_a, "Sets a boolean field.")
      .def(
          "set_int32",
          [](Record& self, FieldIndex field, int32_t value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetInt32(field, value);
          },
          "field"_a, "value"_a, "Sets an int32 scalar field.")
      .def(
          "set_uint32",
          [](Record& self, FieldIndex field, uint32_t value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetUint32(field, value);
          },
          "field"_a, "value"_a, "Sets a uint32 scalar field.")
      .def(
          "set_int64",
          [](Record& self, FieldIndex field, int64_t value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetInt64(field, value);
          },
          "field"_a, "value"_a, "Sets an int64 scalar field.")
      .def(
          "set_uint64",
          [](Record& self, FieldIndex field, uint64_t value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetUint64(field, value);
          },
          "field"_a, "value"_a, "Sets a uint64 scalar field.")
      .def(
          "set_double",
          [](Record& self, FieldIndex field, double value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetDouble(field, value);
          },
          "field"_a, "value"_a, "Sets a double scalar field.")
      .def(
          "set_string",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            if (nb::isinstance<nb::str>(value))
              self.SetString(field, nb::cast<std::string_view>(value));
            else if (nb::isinstance<nb::bytes>(value))
              throw nb::type_error(
                  "bytes is not a supported FieldValue; use str");
            else
              throw nb::type_error("Expected str");
          },
          "field"_a, "value"_a, "Sets a string field.")
      .def(
          "set_int32_sequence",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetInt32Sequence(field, PyToVector<int32_t>(value));
          },
          "field"_a, "value"_a, "Sets an int32 sequence field.")
      .def(
          "set_uint32_sequence",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetUint32Sequence(field, PyToVector<uint32_t>(value));
          },
          "field"_a, "value"_a, "Sets a uint32 sequence field.")
      .def(
          "set_int64_sequence",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetInt64Sequence(field, PyToVector<int64_t>(value));
          },
          "field"_a, "value"_a, "Sets an int64 sequence field.")
      .def(
          "set_uint64_sequence",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetUint64Sequence(field, PyToVector<uint64_t>(value));
          },
          "field"_a, "value"_a, "Sets a uint64 sequence field.")
      .def(
          "set_double_sequence",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetDoubleSequence(field, PyToVector<double>(value));
          },
          "field"_a, "value"_a, "Sets a double sequence field.")
      .def(
          "set_string_sequence",
          [](Record& self, FieldIndex field, nb::handle value) {
            if (!field.is_valid())
              throw nb::value_error("Cannot set field on invalid FieldIndex");
            self.SetStringSequence(field, PyToVector<std::string>(value));
          },
          "field"_a, "value"_a, "Sets a string sequence field.")
      .def("__len__", &Record::size)
      .def("clear", &Record::clear, "Removes all fields from the record.")
      .def(nb::self == nb::self);

  // Bind `StepControl` enum
  nb::enum_<StepControl>(
      m, "StepControl",
      "Control-flow decision returned by the record consumer callback after "
      "processing each streamed `Record`.")
      .value("CONTINUE", StepControl::kContinue,
             "Continue parsing subsequent events.")
      .value("STOP", StepControl::kStop,
             "Clean early stop requested (e.g. limit reached or match found).");

  // Bind `ParseStatus` enum
  nb::enum_<ParseStatus>(m, "ParseStatus",
                         "Final outcome of the entire parsing operation.")
      .value("COMPLETE", ParseStatus::kComplete,
             "Scanned the entire trace to completion.")
      .value("STOPPED_EARLY", ParseStatus::kStoppedEarly,
             "Parsing stopped early and cleanly because consumer returned "
             "`StepControl.STOP`.");

  // Bind `RecordConsumerRef`
  nb::class_<PyRecordConsumer>(
      m, "RecordConsumerRef",
      "Wrapper and adapter for a record consumer and its completion "
      "lifecycle.\n\n"
      "Note: The `Record` passed to `consume` is a transient view owned by the "
      "parser and must not be retained beyond the callback invocation or "
      "shared across threads.")
      .def(nb::init<const PyRecordConsumer&>(), "other"_a,
           "Copy-constructs a RecordConsumerRef.")
      .def(nb::init<nb::handle>(), "target"_a.none(),
           "Constructs a RecordConsumerRef from a callable or object with a "
           "`consume` method.")
      .def(
          "__copy__",
          [](const PyRecordConsumer& self) { return PyRecordConsumer(self); },
          "Returns a copy of this RecordConsumerRef.")
      .def("consume", &PyRecordConsumer::PyConsume, "record"_a,
           "Processes a streamed transient `Record` and returns a "
           "`StepControl` decision. The `record` reference must not be stored "
           "beyond this call.")
      .def("__call__", &PyRecordConsumer::PyConsume, "record"_a,
           "Processes a streamed transient `Record` and returns a "
           "`StepControl` decision. The `record` reference must not be stored "
           "beyond this call.")
      .def("finalize", &PyRecordConsumer::PyFinalize,
           "result"_a.none() = nb::none(),
           "Finalizes the consumer with the given `ParseStatus` or `Exception` "
           "(defaults to `ParseStatus.COMPLETE`).")
      .def_prop_ro("target", &PyRecordConsumer::target,
                   "The underlying Python consumer target.")
      .def("__repr__", [](const PyRecordConsumer& self) {
        nb::gil_scoped_acquire gil;
        std::ostringstream oss;
        oss << "RecordConsumerRef(target="
            << nb::cast<std::string>(nb::repr(self.target())) << ")";
        return oss.str();
      });

  // Expose parse_xspace_file and parse_xspace_bytes.
  // Releases the GIL during execution so multithreaded parser worker threads
  // can acquire the GIL in InvokeConsume without deadlocking.
  m.def(
      "parse_xspace_file",
      [](std::string_view file_path, Schema& schema,
         RecordConsumerRef consumer) -> ParseStatus {
        absl::StatusOr<ParseStatus> parse_status;
        {
          nb::gil_scoped_release nogil;
          parse_status = ParseXSpace(file_path, schema, consumer);
        }
        if (parse_status.ok()) return *parse_status;
        throw std::runtime_error(std::string(parse_status.status().message()));
      },
      "file_path"_a, "schema"_a, "consumer"_a,
      "Parses an XSpace binary protobuf file into the record consumer.");

  m.def(
      "parse_xspace_bytes",
      [](nb::bytes bytes, Schema& schema,
         RecordConsumerRef consumer) -> ParseStatus {
        const void* data = bytes.data();
        const size_t size = bytes.size();
        tensorflow::profiler::XSpace xspace;
        absl::StatusOr<ParseStatus> parse_status;
        {
          nb::gil_scoped_release nogil;
          if (!xspace.ParseFromArray(data, size))
            throw std::runtime_error("Failed to parse binary XSpace protobuf.");
          parse_status = ParseXSpace(xspace, schema, consumer);
        }
        if (parse_status.ok()) return *parse_status;
        throw std::runtime_error(std::string(parse_status.status().message()));
      },
      "bytes"_a, "schema"_a, "consumer"_a,
      "Parses an in-memory binary XSpace protobuf buffer into the record "
      "consumer.");

  // Test helper for `arrow::Compression::type` `type_caster`.
  m.def("_test_roundtrip_arrow_compression",
        [](arrow::Compression::type c) { return c; });
}

}  // namespace xprof::events_db
