/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
#include "xprof/convert/data_table_utils.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "nlohmann/json.hpp"

namespace tensorflow {
namespace profiler {
namespace {
absl::TimeZone GoogleTimeZone() {
  static const auto* tz = [] {
    auto* tz = new absl::TimeZone;
    absl::LoadTimeZone("America/Los_Angeles", tz);
    return tz;
  }();
  return *tz;
}

template <typename T>
const char* GetNegStr(T* value) {
  if (*value < 0) {
    *value = -(*value);
    return "-";
  } else {
    return "";
  }
}

std::string convertBytesToHumanReadableFormat(int64_t num_bytes) {
  if (num_bytes == std::numeric_limits<int64_t>::min()) {
    // Special case for number whose absolute value is out of range.
    return "-8E";
  }

  const char* neg_str = GetNegStr(&num_bytes);

  // Special case for bytes.
  if (num_bytes < int64_t{1024}) {
    // No fractions for bytes.
    return absl::StrFormat("%s%dB", neg_str, num_bytes);
  }

  static const char units[] = "KMGTPE";  // int64 only goes up to E.
  const char* unit = units;
  while (num_bytes >= int64_t{1024} * int64_t{1024}) {
    num_bytes /= int64_t{1024};
    ++unit;
    CHECK(unit < units + ABSL_ARRAYSIZE(units));
  }

  return absl::StrFormat("%s%.*f%c", neg_str, (*unit == 'K') ? 1 : 2,
                         num_bytes / 1024.0, *unit);
}
}  // namespace

// Value
Value::Value() : null_(false) {}
bool Value::IsNull() const { return null_; }
void Value::set_null(bool null) { null_ = null; }

// TextValue
TextValue::TextValue() : value_("") { set_null(true); }
TextValue::TextValue(absl::string_view value) : value_(value) {}
const char& TextValue::GetType() const { return kTextTypeCode; }
const std::string& TextValue::GetValue() const {
  DCHECK(!IsNull()) << "This is a NULL value.";
  return value_;
}

// NumberValue
NumberValue::NumberValue() : value_(0.0) { set_null(true); }
NumberValue::NumberValue(double value) : value_(value) {}
const char& NumberValue::GetType() const { return kNumberTypeCode; }
double NumberValue::GetValue() const {
  DCHECK(!IsNull()) << "This is a NULL value.";
  return value_;
}

// BooleanValue
BooleanValue::BooleanValue() : value_(false) { set_null(true); }
BooleanValue::BooleanValue(bool value) : value_(value) {}
const char& BooleanValue::GetType() const { return kBooleanTypeCode; }
bool BooleanValue::GetValue() const {
  DCHECK(!IsNull()) << "This is a NULL value.";
  return value_;
}

// TableCell
TableCell::TableCell(bool value) : value(new BooleanValue(value)) {}
TableCell::TableCell(double value) : value(new NumberValue(value)) {}
TableCell::TableCell(absl::string_view value) : value(new TextValue(value)) {}
TableCell::TableCell(const char* value) : value(new TextValue(value)) {}

TableCell::TableCell(Value* value, absl::string_view formatted_value)
    : value(value), formatted_value(new std::string(formatted_value)) {}
TableCell::TableCell(bool value, absl::string_view formatted_value)
    : value(new BooleanValue(value)),
      formatted_value(new std::string(formatted_value)) {}
TableCell::TableCell(double value, absl::string_view formatted_value)
    : value(new NumberValue(value)),
      formatted_value(new std::string(formatted_value)) {}
TableCell::TableCell(absl::string_view value, absl::string_view formatted_value)
    : value(new TextValue(value)),
      formatted_value(new std::string(formatted_value)) {}
TableCell::TableCell(const char* value, absl::string_view formatted_value)
    : value(new TextValue(value)),
      formatted_value(new std::string(formatted_value)) {}

TableCell::TableCell(
    Value* value,
    const absl::btree_map<std::string, std::string>& custom_properties)
    : value(value), custom_properties(custom_properties) {}
TableCell::TableCell(
    bool value,
    const absl::btree_map<std::string, std::string>& custom_properties)
    : value(new BooleanValue(value)), custom_properties(custom_properties) {}
TableCell::TableCell(
    double value,
    const absl::btree_map<std::string, std::string>& custom_properties)
    : value(new NumberValue(value)), custom_properties(custom_properties) {}
TableCell::TableCell(
    absl::string_view value,
    const absl::btree_map<std::string, std::string>& custom_properties)
    : value(new TextValue(value)), custom_properties(custom_properties) {}
TableCell::TableCell(
    const char* value,
    const absl::btree_map<std::string, std::string>& custom_properties)
    : value(new TextValue(value)), custom_properties(custom_properties) {}

nlohmann::json TableCell::GetCellValue() const {
  return GetCellValueImp(value.get());
}

nlohmann::json TableCell::GetCellValueImp(const Value* value) const {
  if (value == nullptr || value->IsNull()) {
    return "";
  }
  const char type_code = value->GetType();
  if (type_code == kTextTypeCode) {
    auto text_value = dynamic_cast<const TextValue*>(value);
    if (text_value != nullptr) {
      return text_value->GetValue();
    }
  }
  if (type_code == kNumberTypeCode) {
    auto number_value = dynamic_cast<const NumberValue*>(value);
    if (number_value != nullptr) {
      return number_value->GetValue();
    }
  }
  if (type_code == kBooleanTypeCode) {
    auto boolean_value = dynamic_cast<const BooleanValue*>(value);
    if (boolean_value != nullptr) {
      return boolean_value->GetValue();
    }
  }
  return "";
}

std::string TableCell::GetCellValueStr() const {
  return GetCellValueStrImp(value.get());
}

std::string TableCell::GetCellValueStrImp(const Value* value) const {
  if (value == nullptr || value->IsNull()) {
    return "";
  }
  const char type_code = value->GetType();
  if (type_code == kTextTypeCode) {
    auto text_value = dynamic_cast<const TextValue*>(value);
    if (text_value != nullptr) {
      return text_value->GetValue();
    }
  }
  if (type_code == kNumberTypeCode) {
    auto number_value = dynamic_cast<const NumberValue*>(value);
    if (number_value != nullptr) {
      // Use six digits should be enough for human readable numbers.
      return absl::StrCat(absl::SixDigits(number_value->GetValue()));
    }
  }
  if (type_code == kBooleanTypeCode) {
    auto boolean_value = dynamic_cast<const BooleanValue*>(value);
    if (boolean_value != nullptr) {
      // Don't use StrCat here as it converts bools to "0" or "1".
      return boolean_value->GetValue() ? "TRUE" : "FALSE";
    }
  }
  return "";
}

bool TableCell::HasFormattedValue() const {
  return formatted_value != nullptr;
}

const std::string& TableCell::GetFormattedValue() const {
  return *formatted_value;
}

// TableRow
TableRow& TableRow::AddNumberCell(double value) {
  cells_.push_back(std::make_unique<TableCell>(value));
  return *this;
}

TableRow& TableRow::AddTextCell(absl::string_view value) {
  cells_.push_back(std::make_unique<TableCell>(value));
  return *this;
}

TableRow& TableRow::AddBooleanCell(bool value) {
  cells_.push_back(std::make_unique<TableCell>(value));
  return *this;
}

TableRow& TableRow::AddFormattedNumberCell(double value,
                                           absl::string_view formatted_value) {
  cells_.push_back(std::make_unique<TableCell>(value, formatted_value));
  return *this;
}

TableRow& TableRow::AddFormattedTextCell(absl::string_view value,
                                         absl::string_view formatted_value) {
  cells_.push_back(std::make_unique<TableCell>(value, formatted_value));
  return *this;
}

TableRow& TableRow::AddHexCell(uint64_t value) {
  return AddFormattedNumberCell(value, absl::StrCat("0x", absl::Hex(value)));
}

TableRow& TableRow::AddBytesCell(uint64_t value) {
  return AddFormattedNumberCell(value,
                                convertBytesToHumanReadableFormat(value));
}

TableRow& TableRow::AddFormattedDateCell(absl::Time value,
                                         absl::string_view value_format) {
  return AddFormattedNumberCell(
      absl::ToUnixSeconds(value),
      absl::FormatTime(value_format, value, GoogleTimeZone()));
}

std::vector<const TableCell*> TableRow::GetCells() const {
  std::vector<const TableCell*> cells;
  cells.reserve(cells_.size());
  for (const std::unique_ptr<TableCell>& cell : cells_) {
    cells.push_back(cell.get());
  }
  return cells;
}

void TableRow::SetCustomProperties(
    const absl::btree_map<std::string, std::string>& custom_properties) {
  custom_properties_ = custom_properties;
}

void TableRow::AddCustomProperty(std::string name, std::string value) {
  custom_properties_[name] = value;
}

const absl::btree_map<std::string, std::string>& TableRow::GetCustomProperties()
    const {
  return custom_properties_;
}

int TableRow::RowSize() const { return cells_.size(); }

// DataTable
void DataTable::AddColumn(TableColumn column) {
  table_descriptions_.push_back(std::move(column));
}

const std::vector<TableColumn>& DataTable::GetColumns() const {
  return table_descriptions_;
}

TableRow* DataTable::AddRow() {
  table_rows_.push_back(std::make_unique<TableRow>());
  return table_rows_.back().get();
}

std::vector<const TableRow*> DataTable::GetRows() const {
  std::vector<const TableRow*> rows;
  rows.reserve(table_rows_.size());
  for (const std::unique_ptr<TableRow>& row : table_rows_) {
    rows.push_back(row.get());
  }
  return rows;
}

void DataTable::AddCustomProperty(std::string name, std::string value) {
  custom_properties_[name] = value;
}

const absl::btree_map<std::string, std::string>&
DataTable::GetCustomProperties() const {
  return custom_properties_;
}

std::string DataTable::ToJson() const {
  nlohmann::json table;
  table["cols"] = nlohmann::json::array();
  table["rows"] = nlohmann::json::array();
  if (!custom_properties_.empty()) {
    table["p"] = custom_properties_;
  }
  for (const TableColumn& col : table_descriptions_) {
    nlohmann::json column_json;
    column_json["id"] = col.id;
    column_json["type"] = col.type;
    column_json["label"] = col.label;
    if (!col.custom_properties.empty()) {
      column_json["p"] = col.custom_properties;
    }
    table["cols"].push_back(column_json);
  }
  for (const auto& row : table_rows_) {
    nlohmann::json row_json;
    row_json["c"] = nlohmann::json::array();
    for (const TableCell* cell : row->GetCells()) {
      nlohmann::json cell_json;
      cell_json["v"] = cell->GetCellValue();
      if (cell->HasFormattedValue()) {
        cell_json["f"] = cell->GetFormattedValue();
      }
      if (!cell->custom_properties.empty()) {
        cell_json["p"] = cell->custom_properties;
      }
      row_json["c"].push_back(cell_json);
    }
    if (!row->GetCustomProperties().empty()) {
      row_json["p"] = row->GetCustomProperties();
    }
    table["rows"].push_back(row_json);
  }
  return table.dump();
}

}  // namespace profiler
}  // namespace tensorflow
